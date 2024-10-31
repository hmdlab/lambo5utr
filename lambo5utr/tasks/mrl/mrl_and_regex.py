import re
import torch
import numpy as np
from zmq import device
from lambo5utr.candidate import StringCandidate
from lambo5utr.tasks.base_task import BaseTask
from lambo5utr.utils import random_sequences
from lambo5utr.utils_mutations import apply_mutation, mutation_list
from lambo5utr.tasks.mrl.prediction_mrl_CNN import predict_mrl_CNN

class MrlRegexTask(BaseTask):
    def __init__(
        self,
        anti_regex_list,
        min_seq_len,
        num_start_examples,
        tokenizer,
        candidate_pool,
        obj_dim,
        transform=lambda x: x,
        path_mrl_predictor=None,
        num_gpu_mrl=1,
        device_id=0,
        **kwargs,
    ):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.anti_regex_list = anti_regex_list
        self.min_seq_len = min_seq_len
        self.min_len = min_seq_len - self.tokenizer.kmer + 1 + 2
        self.num_start_examples = num_start_examples
        self.path_mrl_predictor = path_mrl_predictor
        self.num_gpu_mrl = num_gpu_mrl
        self.device_id = str(device_id)
        self.device = torch.device('cuda:{}'.format(self.device_id))

    def task_setup(self, *args, **kwargs):
        num_examples = 0
        selected_seqs = []
        selected_targets = []
        while num_examples < self.num_start_examples:
            # account for start and stop tokens
            all_seqs = random_sequences(self.tokenizer.non_special_vocab, self.num_start_examples, self.min_seq_len, self.max_seq_len)
            base_candidates = np.array([
                StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
            ]).reshape(-1)
            all_targets = self.score(base_candidates)
            positive_example_mask = (all_targets < 0).sum(-1).astype(bool)
            num_positive = positive_example_mask.astype(int).sum()
            num_negative = all_targets.shape[0] - num_positive
            num_selected = max(num_positive, num_negative)

            selected_seqs.append(all_seqs[positive_example_mask][:num_selected])
            selected_targets.append(all_targets[positive_example_mask][:num_selected])
            selected_seqs.append(all_seqs[~positive_example_mask][:num_selected])
            selected_targets.append(all_targets[~positive_example_mask][:num_selected])
            num_examples += num_selected
            print(num_examples)

        all_seqs = np.concatenate(selected_seqs)[:self.num_start_examples]
        all_targets = np.concatenate(selected_targets)[:self.num_start_examples]

        base_candidates = np.array([
            StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
        ]).reshape(-1)
        base_targets = all_targets.copy()

        return base_candidates, base_targets, all_seqs, all_targets
    
    def make_new_candidates(self, base_candidates, new_seqs):
        assert base_candidates.shape[0] == new_seqs.shape[0]
        new_candidates = []
        for b_cand, n_seq in zip(base_candidates, new_seqs):
            mutation_ops = mutation_list(
                b_cand.mutant_residue_seq,
                n_seq,
                self.tokenizer
            )
            new_candidates.append(b_cand.new_candidate(mutation_ops, self.tokenizer))
        return np.stack(new_candidates)

    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        x_cands, x_seqs, f_vals = [], [], []
        for query_pt in x:
            cand_idx, mut_pos, mut_res_idx, op_idx = query_pt
            op_type = self.op_types[op_idx]
            base_candidate = self.candidate_pool[cand_idx]
            base_seq = base_candidate.mutant_residue_seq
            mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
            mut_seq = apply_mutation(base_seq, mut_pos, mut_res, op_type, self.tokenizer)
            mutation_ops = mutation_list(base_seq, mut_seq, self.tokenizer)
            candidate = base_candidate.new_candidate(mutation_ops, self.tokenizer)
            x_cands.append(candidate)
            x_seqs.append(candidate.mutant_residue_seq)
        x_seqs = np.array(x_seqs).reshape(-1)
        x_cands = np.array(x_cands).reshape(-1)

        out["X_cand"] = x_cands
        out["X_seq"] = x_seqs
        out["F"] = self.transform(self.score(x_cands))

    def score(self, candidates):
        print('calculating the scores of candidates...')
        str_array = np.array([cand.mutant_residue_seq for cand in candidates])
        scores = []
        for regex in self.anti_regex_list:
            scores.append(np.array([
                - len(re.findall(regex, str(x))) for x in str_array
            ]).reshape(-1))
        
        scores = np.stack(scores, axis=-1)

        mrl_score = - predict_mrl_CNN(str_array, self.path_mrl_predictor, 
                                max_seq_len=self.max_seq_len, batch_size=self.batch_size, 
                                num_gpu=self.num_gpu_mrl, device=self.device).reshape(-1,1)
        
        # expect that the optimizer is trying to minimize scores
        scores = - np.concatenate([scores, mrl_score], axis=-1).astype(np.float64)
        return scores

    def is_feasible(self, candidates):
        if self.max_len is None:
            is_feasible = np.ones(candidates.shape).astype(bool)
        else:
            is_feasible = np.array([(len(cand) <= self.max_len) & (len(cand) >= self.min_len) for cand in candidates]).reshape(-1)

        return is_feasible
    
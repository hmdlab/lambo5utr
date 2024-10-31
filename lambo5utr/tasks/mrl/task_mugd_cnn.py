import re, os
import torch

import numpy as np
import pandas as pd

from lambo5utr.candidate import StringCandidate
from lambo5utr.tasks.base_task import BaseTask
from lambo5utr.utils import random_sequences
from lambo5utr.utils_mutations import apply_mutation, mutation_list
from lambo5utr.tasks.mrl.prediction_mrl_CNN import (
    predict_mrl_CNN,
    predict_mrl_CNN_multi,
)
from lambo5utr.tasks.mrl.prediction_g4 import predict_g4, predict_g4_multi
from lambo5utr.tasks.mrl.prediction_degradation import (
    predict_in_vitro_degradation,
    predict_in_vitro_degradation_multi,
)
import logging

logger = logging.getLogger(__name__)

class MrlUlessG4DegradationTask(BaseTask):
    def __init__(
        self,
        seq_seed,
        init_randomize,
        given_filename,
        min_seq_len,
        num_start_examples,
        tokenizer,
        candidate_pool,
        exp_name,
        obj_dim,
        transform=lambda x: x,
        path_mrl_predictor=None,
        num_gpu_mrl=1,
        device_id=0,
        construct=None,
        **kwargs,
    ):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.anti_regex_list = [
            "T",
        ]
        self.seq_seed = seq_seed
        self.init_randomize = init_randomize
        if self.init_randomize is False:
            self.given_filename = given_filename
        self.min_seq_len = min_seq_len
        self.min_len = min_seq_len - self.tokenizer.kmer + 1 + 2
        self.num_start_examples = num_start_examples
        self.path_mrl_predictor = path_mrl_predictor
        self.num_gpu_mrl = num_gpu_mrl
        self.device_id = str(device_id)
        self.device = torch.device("cuda:{}".format(self.device_id))
        self.construct = construct
        self.upstream_seq, self.downstream_seq = self.construct.get_seq()
        self.project_root = ""
        self.exp_name = exp_name

    def task_setup(self, *args, **kwargs):
        num_examples = 0
        selected_seqs = []
        selected_targets = []

        self.project_root = kwargs["project_root"]
        cache_dir = os.path.join(self.project_root, "cache_dir")
        if self.init_randomize:
            filename = "seed{}_numsample{}_{}.csv".format(
                self.seq_seed, self.num_start_examples, self.construct.name
            )
        else:
            filename = "{}_numsample{}_{}.csv".format(
                self.given_filename, self.num_start_examples, self.construct.name
            )
        df_columns = ["sequence", "nonU_cotent", "mrl", "G4", "degradation"]

        if os.path.isfile(os.path.join(cache_dir, filename)):
            cache_path = os.path.join(cache_dir, filename)
            logger.info(
                "loading initial sequences from cache file at: {}".format(cache_path)
            )
            df = pd.read_csv(cache_path)
            all_seqs = np.array(df["sequence"])
            all_targets = np.array(df[df_columns[1:]])
            base_candidates = np.array(
                [
                    StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer)
                    for seq in all_seqs
                ]
            ).reshape(-1)
        else:
            if not os.path.exists(os.path.join(cache_dir)):
                os.makedirs(cache_dir)
            cache_path = os.path.join(cache_dir, filename)

            if self.init_randomize:
                all_seqs = random_sequences(
                    self.tokenizer.non_special_vocab,
                    self.num_start_examples,
                    self.min_seq_len,
                    self.max_seq_len,
                )
            else:
                given_path = os.path.join(
                    cache_dir, "{}.csv".format(self.given_filename)
                )
                logger.info(
                    "scoring initial sequences at : {}".format(given_path)
                )
                # we suppose the file is given in the .csv format
                all_seqs = np.array(
                    pd.read_csv(given_path)["sequence"].iloc[: self.num_start_examples]
                )
                # TODO: format the given sequence into DNA (A, T, G, C) with the given min_seq_len, max_seq_len, num_start_examples
            base_candidates = np.array(
                [
                    StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer)
                    for seq in all_seqs
                ]
            ).reshape(-1)
            all_targets = self.score(base_candidates)

            df = pd.DataFrame(all_seqs, columns=df_columns[0:1])
            df = pd.concat(
                [df, pd.DataFrame(all_targets, columns=df_columns[1:])], axis=1
            )
            logger.info("saving initial sequences to: {}".format(cache_path))
            df.to_csv(cache_path, index=False)

        base_targets = all_targets.copy()

        return base_candidates, base_targets, all_seqs, all_targets

    def make_new_candidates(self, base_candidates, new_seqs):
        assert base_candidates.shape[0] == new_seqs.shape[0]
        new_candidates = []
        for b_cand, n_seq in zip(base_candidates, new_seqs):
            mutation_ops = mutation_list(
                b_cand.mutant_residue_seq, n_seq, self.tokenizer
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
            mut_seq = apply_mutation(
                base_seq, mut_pos, mut_res, op_type, self.tokenizer
            )
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
        logger.info("calculating scores...")
        logger.info(f"exp_name:{self.exp_name}")
        str_array = np.array([cand.mutant_residue_seq for cand in candidates])
        regex_scores = []
        for regex in self.anti_regex_list:
            regex_scores.append(
                np.array([len(re.findall(regex, str(x))) for x in str_array]).reshape(
                    -1
                )
            )

        # lower is better
        regex_scores = np.stack(regex_scores, axis=-1)

        # lower is better
        if self.exp_name == None:
            mrl_score = -predict_mrl_CNN(
                str_array,
                self.path_mrl_predictor,
                max_seq_len=self.max_seq_len,
                batch_size=self.batch_size,
                num_gpu=self.num_gpu_mrl,
                device=self.device,
                project_root=self.project_root,
            ).reshape(-1, 1)

            g4_score = predict_g4(
                str_array,
                max_seq_len=self.max_seq_len,
                upstream_seq=self.upstream_seq,
                downstream_seq=self.downstream_seq,
                project_root=self.project_root,
            )
            # convert to discrete scores so that the G4 oracle is less stringent
            g4_score = np.round(g4_score, decimals=1)

            degradation_score = predict_in_vitro_degradation(
                str_array,
                max_seq_len=self.max_seq_len,
                upstream_seq=self.upstream_seq,
                downstream_seq=self.downstream_seq,
                project_root=self.project_root,
            )
        else:  # When multi experiment, multi ver should be run to avoid conflicting of tmp files
            mrl_score = -predict_mrl_CNN_multi(
                str_array,
                self.path_mrl_predictor,
                max_seq_len=self.max_seq_len,
                batch_size=self.batch_size,
                num_gpu=self.num_gpu_mrl,
                device=self.device,
                exp_name=self.exp_name,
                project_root=self.project_root,
            ).reshape(-1, 1)

            g4_score = predict_g4_multi(
                str_array,
                max_seq_len=self.max_seq_len,
                upstream_seq=self.upstream_seq,
                downstream_seq=self.downstream_seq,
                exp_name=self.exp_name,
                project_root=self.project_root,
            )
            g4_score = np.round(g4_score, decimals=1)

            degradation_score = predict_in_vitro_degradation_multi(
                str_array,
                max_seq_len=self.max_seq_len,
                upstream_seq=self.upstream_seq,
                downstream_seq=self.downstream_seq,
                exp_name=self.exp_name,
                project_root=self.project_root,
            )
        logger.info(f"regex_scores:{regex_scores}")
        logger.info(f"mrl_score:{mrl_score}")
        logger.info(f"g4_score:{g4_score}")
        logger.info(f"degradation:{degradation_score}")
        
        # expect that the optimizer is trying to minimize scores
        scores = np.concatenate(
            [regex_scores, mrl_score, g4_score, degradation_score], axis=-1
        ).astype(np.float64)
        return scores

    def is_feasible(self, candidates):
        if self.max_len is None:
            is_feasible = np.ones(candidates.shape).astype(bool)
        else:
            is_feasible = np.array(
                [
                    (len(cand) <= self.max_len) & (len(cand) >= self.min_len)
                    for cand in candidates
                ]
            ).reshape(-1)
        return is_feasible
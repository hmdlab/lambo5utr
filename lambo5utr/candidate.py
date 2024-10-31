import copy
import uuid
from pathlib import Path
from Bio.SeqUtils import seq1

from lambo5utr.utils_mutations import StringSubstitution, StringDeletion, StringInsertion


"""
def apply_mutation(base_seq, mut_pos, mut_res, tokenizer):
    tokens = tokenizer.decode_lambo(tokenizer.encode_lambo(base_seq)).split(" ")[1:-1]
    mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[(mut_pos + 1):])
    return mut_seq
"""

class StringCandidate:
    def __init__(self, wild_seq, mutation_list, tokenizer, wild_name=None, dist_from_wild=0.):
        self.wild_seq = wild_seq
        self.uuid = uuid.uuid4().hex
        self.wild_name = 'unnamed' if wild_name is None else wild_name
        self.mutant_residue_seq = self.apply_mutations(mutation_list, tokenizer)
        self.dist_from_wild = dist_from_wild
        self.tokenizer = tokenizer

    def __len__(self):
        tok_idxs = self.tokenizer.encode_lambo(self.mutant_residue_seq)
        return len(tok_idxs)

    def apply_mutations(self, mutation_list, tokenizer):
        if len(mutation_list) == 0:
            return self.wild_seq
        mutant_seq = copy.deepcopy(self.wild_seq)
        mutant_seq = tokenizer.encode_lambo(mutant_seq)[1:-1]
        for mutation_op in mutation_list:
            old_tok_idx = mutation_op.old_token_idx
            mut_pos = mutation_op.token_pos
            if mut_pos < len(mutant_seq):
                assert old_tok_idx == mutant_seq[mut_pos], str(mutation_op)
            if isinstance(mutation_op, StringSubstitution):
                new_tok_idx = mutation_op.new_token_idx
                mutant_seq = mutant_seq[:mut_pos] + [new_tok_idx] + mutant_seq[mut_pos + 1:]
            elif isinstance(mutation_op, StringDeletion):
                mutant_seq = mutant_seq[:mut_pos] + mutant_seq[mut_pos + 1:]
            elif isinstance(mutation_op, StringInsertion):
                new_tok_idx = mutation_op.new_token_idx
                mutant_seq = mutant_seq[:mut_pos] + [new_tok_idx] + mutant_seq[mut_pos:]
            else:
                raise RuntimeError('unrecognized mutation op')

        mutant_seq = tokenizer.decode_lambo(mutant_seq, output_tokens=False)
        return mutant_seq

    def new_candidate(self, mutation_list, tokenizer):
        cand_kwargs = dict(
            wild_seq=self.mutant_residue_seq,
            mutation_list=mutation_list,
            tokenizer=tokenizer,
            wild_name=self.wild_name,
            dist_from_wild=self.dist_from_wild + len(mutation_list),
        )
        return StringCandidate(**cand_kwargs)
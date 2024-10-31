from Levenshtein._levenshtein import distance as edit_distance, editops as edit_ops

def mutation_list(src_str, tgt_str, tokenizer):
    # def to_unicode(seq):
    #     return ''.join([chr(int(x)) for x in seq]).encode_lambo('utf-8').decode_lambo('utf-8')

    src_token_id_seq = ''.join([chr(x) for x in tokenizer.encode_lambo(src_str)[1:-1]])
    tgt_token_id_seq = ''.join([chr(x) for x in tokenizer.encode_lambo(tgt_str)[1:-1]])

    if edit_distance(src_token_id_seq, tgt_token_id_seq) == 0:
        return []

    mutations = []
    tmp_tok_id_seq, trans_adj = src_token_id_seq, 0

    # TODO make sure insertion is properly supported, update tests

    for op_name, pos_1, pos_2 in edit_ops(src_token_id_seq, tgt_token_id_seq):
        tmp_pos = pos_1 + trans_adj

        if op_name == "delete":
            char_1 = ord(src_token_id_seq[pos_1])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1]
            else:
                tmp_tok_id_seq = tmp_tok_id_seq[:tmp_pos] + tmp_tok_id_seq[tmp_pos + 1:]

            mutations.append(StringDeletion(char_1, tmp_pos, tokenizer))
            trans_adj -= 1

        if op_name == "replace":
            char_1 = ord(src_token_id_seq[pos_1])
            char_2 = ord(tgt_token_id_seq[pos_2])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1] + chr(char_2)
            else:
                tmp_tok_id_seq = tmp_tok_id_seq[:tmp_pos] + chr(char_2) + tmp_tok_id_seq[tmp_pos + 1:]

            mutations.append(StringSubstitution(char_1, tmp_pos, char_2, tokenizer))

        if op_name == "insert":
            if pos_1 < len(src_token_id_seq):
                char_1 = ord(src_token_id_seq[pos_1])
            else:
                char_1 = None

            char_2 = ord(tgt_token_id_seq[pos_2])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1] + chr(char_2) + tmp_tok_id_seq[-1]
            else:
                tmp_tok_id_seq = tmp_tok_id_seq[:tmp_pos] + chr(char_2) + tmp_tok_id_seq[tmp_pos:]

            mutations.append(StringInsertion(char_1, tmp_pos, char_2, tokenizer))
            trans_adj += 1

    # check output
    tmp_str = tokenizer.decode_lambo([ord(x) for x in tmp_tok_id_seq], output_tokens=False)
    assert tmp_tok_id_seq == tgt_token_id_seq
    assert tmp_str == tgt_str, f'{tgt_str}\n{tmp_str}'

    return mutations

def apply_mutation(base_seq, mut_pos, mut_res, op_type, tokenizer):
    tokens = tokenizer.decode_lambo(tokenizer.encode_lambo(base_seq)).split(" ")[1:-1]

    if op_type == 'sub':
        mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[(mut_pos + 1):])
    elif op_type == 'ins':
        mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[mut_pos:])
    elif op_type == 'del':
        mut_seq = "".join(tokens[:mut_pos] + tokens[(mut_pos + 1):])
    else:
        raise ValueError('unsupported operation')

    return mut_seq

class StringSubstitution:
    def __init__(self, old_token_idx, token_pos, new_token_idx, tokenizer):
        self.old_token_idx = int(old_token_idx)
        self.old_token = tokenizer.decode_lambo(self.old_token_idx)

        self.token_pos = int(token_pos)

        self.new_token_idx = int(new_token_idx)
        self.new_token = tokenizer.decode_lambo(self.new_token_idx)

    def __str__(self):
        prefix = f"{self.token_pos}{self.old_token}-{self.token_pos}{self.old_token}_"
        return prefix + f"sub{self.new_token}"


class StringDeletion:
    def __init__(self, old_token_idx, token_pos, tokenizer):
        self.old_token_idx = int(old_token_idx)
        self.old_token = tokenizer.decode_lambo(self.old_token_idx)

        self.token_pos = int(token_pos)

    def __str__(self):
        prefix = f"{self.token_pos}{self.old_token_idx}-{self.token_pos}{self.old_token_idx}_"
        return prefix + "del"


class StringInsertion:
    def __init__(self, old_token_idx, token_pos, new_token_idx, tokenizer):
        if old_token_idx is None:
            self.old_token_idx = None
            self.old_token = ''
        else:
            self.old_token_idx = int(old_token_idx)
            self.old_token = tokenizer.decode_lambo(self.old_token_idx)

        self.token_pos = int(token_pos)

        self.new_token_idx = int(new_token_idx)
        self.new_token = tokenizer.decode_lambo(self.new_token_idx)

    def __str__(self):
        prefix = f"{self.token_pos}{self.old_token}-{self.token_pos}{self.old_token}_"
        return prefix + f"ins{self.new_token}"
import torch
import numpy as np
import itertools
from collections import OrderedDict
from cachetools import cached, LRUCache

AMINO_ACIDS = ["A","R","N","D","C",
            "E","Q","G","H","I",
            "L","K","M","F","P",
            "S","T","W","Y","V"]
DNA_ALPHABETS = ["A","T","G","C"]
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "0"]

MASK_LIST = {
    1: [],
    2: [1],
    3: [-1, 1],
    4: [-1, 1, 2],
    5: [-2, -1, 1, 2],
    6: [-2, -1, 1, 2, 3]
}

class InitkmerTokenizer:
    def __init__(self, non_special_vocab: list, non_special_kmer_vocab: list, full_vocab: list, padding_token="[PAD]",
                 masking_token="[MASK]", cls_token="[CLS]", sep_token="[SEP]", unk_token="[UNK]", kmer=1, mask_list=[]):
        self.non_special_vocab = non_special_vocab
        self.non_special_kmer_vocab = non_special_kmer_vocab
        self.full_vocab = full_vocab
        self.special_vocab = set(full_vocab) - set(non_special_kmer_vocab)
        if not hasattr(self, 'vocab'):
            self.vocab = OrderedDict({a: i for (i, a) in enumerate(full_vocab)})
        if not hasattr(self, 'ids_to_tokens'):
            self.ids_to_tokens = OrderedDict({i: a for (i, a) in enumerate(full_vocab)})
        if not hasattr(self, '_unk_token'):
            self._unk_token = unk_token
        if not hasattr(self, '_pad_token'):
            self._pad_token = padding_token
        if not hasattr(self, '_mask_token'):
            self._mask_token = masking_token
        if not hasattr(self, '_cls_token'):
            self._cls_token = cls_token
        if not hasattr(self, '_sep_token'):
            self._sep_token = sep_token
        if not hasattr(self, 'kmer'):
            self.kmer = kmer

        self.padding_idx = self.vocab[self._pad_token]
        self.masking_idx = self.vocab[self._mask_token]
        self.cls_idx = self.vocab[self._cls_token]
        self.sep_idx = self.vocab[self._sep_token]
        self.sampling_vocab = non_special_kmer_vocab
        self.non_special_idxs = [self.convert_token_to_id(t) for t in non_special_kmer_vocab]
        self.special_idxs = [self.convert_token_to_id(t) for t in self.special_vocab]
        self.mask_list = mask_list

    @cached(cache=LRUCache(maxsize=int(1e4)))
    def encode_lambo(self, seq: str) -> list:
        tokens = self.convert_seq_to_token(seq)
        tokens = [self._cls_token] + tokens + [self._sep_token]
        return [self.convert_token_to_id(c) for c in tokens]

    def decode_lambo(self, token_ids, output_tokens=True, mask_idx=[]):
        if isinstance(token_ids, int):
            return self.convert_id_to_token(token_ids)
        tokens = []
        for t_id in token_ids:
            token = self.convert_id_to_token(t_id)
            if token in self.special_vocab and token not in [self._mask_token, self._unk_token]:
                continue
            tokens.append(token)
        if output_tokens:
            return ' '.join(tokens)
        else:
            return self.convert_token_to_seq(tokens, mask_idx)

    def convert_id_to_token(self, token_id):
        if torch.is_tensor(token_id):
            token_id = token_id.item()
        assert isinstance(token_id, int)
        return self.ids_to_tokens.get(token_id, self._unk_token)

    def convert_token_to_id(self, token):
        unk_idx = self.vocab[self._unk_token]
        return self.vocab.get(token, unk_idx)
    
    def convert_seq_to_token(self, seq):
        tokens = [seq[x:x+self.kmer] for x in range(len(seq)+1-self.kmer)]
        return tokens
    
    def convert_token_to_seq(self, tokens, mask_idx=[]):
        # print('tokens', tokens)
        # print('mask_idx', mask_idx)
        offset = 1 # considering CLS token
        if len(mask_idx)>0:
            mask_idx = np.array(mask_idx)
            mask_idx -= offset
        if self.kmer==1:
            main_index = 0
        else:
            main_index = self.mask_list.index(1)
        last_token = ''
        seq = ''
        skip_flg = 0
        for idx, token in enumerate(tokens):
            if (token in self.special_vocab and token not in [self._mask_token, self._unk_token]) or skip_flg > 0:
                if skip_flg > 0:
                    skip_flg -= 1
                    last_token = token
                continue
            if len(seq) == 0:
                seq += token[:main_index]
            if idx in mask_idx:
                seq = seq[:-main_index] + token
                skip_flg = max(self.mask_list)
                last_token = token
                continue
            seq += token[main_index]
            last_token = token
        seq += last_token[main_index+skip_flg+1:]
        return seq

    def set_sampling_vocab(self, sampling_vocab=None, max_ngram_size=1):
        if sampling_vocab is None:
            sampling_vocab = []
            for i in range(1, max_ngram_size + 1):
                prod_space = [self.non_special_kmer_vocab] * i
                for comb in itertools.product(*prod_space):
                    sampling_vocab.append("".join(comb))
        else:
            new_tokens = set(sampling_vocab) - set(self.full_vocab)
            self.full_vocab.extend(list(new_tokens))
            self.vocab = OrderedDict({a: i for (i, a) in enumerate(self.full_vocab)})
            self.ids_to_tokens = {i: a for (i, a) in enumerate(self.full_vocab)}

        self.sampling_vocab = sampling_vocab

class ResidueTokenizer(InitkmerTokenizer):
    def __init__(self, kmer=1):
        mask_list = MASK_LIST[kmer]
        super().__init__(non_special_vocab = AMINO_ACIDS, non_special_kmer_vocab = AMINO_ACIDS, full_vocab = AMINO_ACIDS + SPECIAL_TOKENS, kmer=kmer, mask_list=mask_list)

class DNATokenizer(InitkmerTokenizer):
    def __init__(self, kmer=1):
        mask_list = MASK_LIST[kmer]
        super().__init__(non_special_vocab = DNA_ALPHABETS, non_special_kmer_vocab = DNA_ALPHABETS, full_vocab = DNA_ALPHABETS + SPECIAL_TOKENS, kmer=kmer, mask_list=mask_list)

class DNAkmerTokenizer(InitkmerTokenizer):
    def __init__(self, kmer=3):
        kmer_alphabets = generate_kmer_alphabet(alphabets=DNA_ALPHABETS, kmer=kmer)
        mask_list = MASK_LIST[kmer]
        super().__init__(non_special_vocab = DNA_ALPHABETS, non_special_kmer_vocab = kmer_alphabets, full_vocab = kmer_alphabets + SPECIAL_TOKENS, kmer=kmer, mask_list=mask_list)

def generate_kmer_alphabet(alphabets: list, kmer: int) -> list:
    kmer_alphabets = ['']
    while kmer > 0:
        current_alphabets = kmer_alphabets
        kmer_alphabets = []
        for alpha in alphabets:
            for c_alpha in current_alphabets:
                kmer_alphabets.append(alpha+c_alpha)
        kmer -= 1
    return kmer_alphabets

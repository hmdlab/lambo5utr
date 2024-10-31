from typing import Sequence

import torch
import numpy as np
import random

from torch import LongTensor


def padding_collate_fn(batch, padding_value=0.0):
    with torch.no_grad():
        if isinstance(batch[0], tuple):
            k = len(batch[0])
            x = torch.nn.utils.rnn.pad_sequence(
                [b[0] for b in batch], batch_first=True, padding_value=padding_value
            )
            rest = [torch.stack([b[i] for b in batch]) for i in range(1, k)]
            return (x,) + tuple(rest)
        else:
            x = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=padding_value
            )
            return x


class StringToLongTensor:
    def __init__(self, tokenizer, max_len=None):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, x: str):
        tok_idxs = self.tokenizer.encode_lambo(x)
        tok_idxs = torch.LongTensor(tok_idxs)
        num_tokens = tok_idxs.size(0)
        if self.max_len is not None and num_tokens < self.max_len:
            len_diff = self.max_len - num_tokens
            padding = LongTensor(
                [self.tokenizer.padding_idx] * len_diff
            )
            tok_idxs = torch.cat([tok_idxs, padding])
        elif self.max_len is not None and num_tokens > self.max_len:
            tok_idxs = tok_idxs[:self.max_len]

        return tok_idxs


class SequenceTranslation(object):
    """
    Performs a random cycle rotation of a tokenized sequence up to
    `max_shift` tokens either left or right.
    Assumes the sequence has start and stop tokens and NO padding tokens at the end.
    """

    def __init__(self, max_shift: int):
        self.max_shift = max_shift

    def __call__(self, x: LongTensor, shift=None):
        """
        Args:
            x: LongTensor with shape (num_tokens,)
            shift: (optional) magnitude and direction of shift, randomly sampled if None
        """
        if shift is None:
            shift = random.randint(-self.max_shift, self.max_shift)
        else:
            shift = min(shift, self.max_shift)
            shift = max(shift, -self.max_shift)

        num_valid_tokens = x.size(0) - 2
        if shift < 0:
            shift = -(-shift % num_valid_tokens)
        elif shift > 0:
            shift = shift % num_valid_tokens

        if shift == 0:
            return x

        # don't include start/stop tokens in rotation
        trimmed_x = x[1:-1]
        rot_x = x.clone()
        # left shift
        if shift < 0:
            rot_x[1: num_valid_tokens + shift + 1] = trimmed_x[-shift:]
            rot_x[num_valid_tokens + shift + 1: -1] = trimmed_x[:-shift]
        # right shift
        else:
            rot_x[1: shift + 1] = trimmed_x[-shift:]
            rot_x[shift + 1: -1] = trimmed_x[:-shift]

        return rot_x
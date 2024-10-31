import torch
import numpy as np
import math
import random
from collections.__init__ import namedtuple
from scipy.stats import rankdata
from scipy.special import softmax

from lambo5utr.transforms import padding_collate_fn

def random_sequences(alphabet, num, min_seq_len=200, max_seq_len=250):
    sequences = []
    for _ in range(num):
        length = np.random.randint(min_seq_len, max_seq_len + 1)
        idx = np.random.choice(len(alphabet), size=length, replace=True)
        sequences.append("".join([alphabet[i] for i in idx]))
    sequences = np.array(sequences)
    return sequences

fields = ("inputs", "targets")
defaults = (np.array([]), np.array([]))
DataSplit = namedtuple("DataSplit", fields, defaults=defaults)

def weighted_resampling(scores, k=1., num_samples=None):
    """
    Multi-objective ranked resampling weights.
    Assumes scores are being minimized.

    Args:
        scores: (num_rows, num_scores)
        k: softmax temperature
        num_samples: number of samples to draw (with replacement)
    """
    num_rows = scores.shape[0]
    scores = scores.reshape(num_rows, -1)

    ranks = rankdata(scores, method='dense', axis=0)  # starts from 1
    ranks = ranks.max(axis=-1)  # if A strictly dominates B it will have higher weight.

    weights = softmax(-np.log(ranks) / k)

    num_samples = num_rows if num_samples is None else num_samples
    resampled_idxs = np.random.choice(
        np.arange(num_rows), num_samples, replace=True, p=weights
    )
    return ranks, weights, resampled_idxs

def safe_np_cat(arrays, **kwargs):
    if all([arr.size == 0 for arr in arrays]):
        return np.array([])
    cat_arrays = [arr for arr in arrays if arr.size]
    return np.concatenate(cat_arrays, **kwargs)

def str_to_tokens(str_array, tokenizer):
    tokens = [
        torch.tensor(tokenizer.encode_lambo(x)) for x in str_array
    ]
    batch = padding_collate_fn(tokens, tokenizer.padding_idx)
    return batch

def tokens_to_str(tok_idx_array, tokenizer, mask_idxs=None):
    if mask_idxs is not None:
        str_array = np.array([
            tokenizer.decode_lambo(token_ids, output_tokens=False, mask_idx=mask_idx) for token_ids, mask_idx in zip(tok_idx_array, mask_idxs)
        ])
    else:
        str_array = np.array([
            tokenizer.decode_lambo(token_ids, output_tokens=False) for token_ids in tok_idx_array
        ])
    return str_array

def draw_bootstrap(*arrays, bootstrap_ratio=0.632, min_samples=1):
    """
    Returns bootstrapped arrays that (in expectation) have `bootstrap_ratio` proportion
    of the original rows. The size of the bootstrap is computed automatically.
    For large input arrays, the default value will produce a bootstrap
    the same size as the original arrays.

    :param arrays: indexable arrays (e.g. np.ndarray, torch.Tensor)
    :param bootstrap_ratio: float in the interval (0, 1)
    :param min_samples: (optional) instead specify the minimum size of the bootstrap
    :return: bootstrapped arrays
    """

    num_data = arrays[0].shape[0]
    assert all(arr.shape[0] == num_data for arr in arrays)

    if bootstrap_ratio is None:
        num_samples = min_samples
    else:
        assert bootstrap_ratio < 1
        num_samples = int(math.log(1 - bootstrap_ratio) / math.log(1 - 1 / num_data))
        num_samples = max(min_samples, num_samples)

    idxs = random.choices(range(num_data), k=num_samples)
    res = [arr[idxs] for arr in arrays]
    return res


def to_tensor(*arrays, device=torch.device('cpu')):
    tensors = []
    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            tensors.append(arr.to(device))
        else:
            tensors.append(torch.tensor(arr, device=device))

    if len(arrays) == 1:
        return tensors[0]

    return tensors


def batched_call(fn, arg_array, batch_size, *args, **kwargs):
    batch_size = arg_array.shape[0] if batch_size is None else batch_size
    num_batches = max(1, arg_array.shape[0] // batch_size)

    if isinstance(arg_array, np.ndarray):
        arg_batches = np.array_split(arg_array, num_batches)
    elif isinstance(arg_array, torch.Tensor):
        arg_batches = torch.split(arg_array, num_batches)
    else:
        raise ValueError

    return [fn(batch, *args, **kwargs) for batch in arg_batches]

def update_splits(
    train_split: DataSplit,
    val_split: DataSplit,
    test_split: DataSplit,
    new_split: DataSplit,
    holdout_ratio: float = 0.2,
):
    r"""
    This utility function updates train, validation and test data splits with
    new observations while preventing leakage from train back to val or test.
    New observations are allocated proportionally to prevent the
    distribution of the splits from drifting apart.

    New rows are added to the validation and test splits randomly according to
    a binomial distribution determined by the holdout ratio. This allows all splits
    to be updated with as few new points as desired. In the long run the split proportions
    will converge to the correct values.
    """
    train_inputs, train_targets = train_split
    val_inputs, val_targets = val_split
    test_inputs, test_targets = test_split

    # shuffle new data
    new_inputs, new_targets = new_split
    new_perm = np.random.permutation(
        np.arange(new_inputs.shape[0])
    )
    new_inputs = new_inputs[new_perm]
    new_targets = new_targets[new_perm]

    unseen_inputs = safe_np_cat([test_inputs, new_inputs])
    unseen_targets = safe_np_cat([test_targets, new_targets])

    num_rows = train_inputs.shape[0] + val_inputs.shape[0] + unseen_inputs.shape[0]
    num_test = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        unseen_inputs.shape[0],
    )
    num_test = max(test_inputs.shape[0], num_test) if test_inputs.size else max(1, num_test)

    # first allocate to test split
    test_split = DataSplit(unseen_inputs[:num_test], unseen_targets[:num_test])

    resid_inputs = unseen_inputs[num_test:]
    resid_targets = unseen_targets[num_test:]
    resid_inputs = safe_np_cat([val_inputs, resid_inputs])
    resid_targets = safe_np_cat([val_targets, resid_targets])

    # then allocate to val split
    num_val = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        resid_inputs.shape[0],
    )
    num_val = max(val_inputs.shape[0], num_val) if val_inputs.size else max(1, num_val)
    val_split = DataSplit(resid_inputs[:num_val], resid_targets[:num_val])

    # train split gets whatever is left
    last_inputs = resid_inputs[num_val:]
    last_targets = resid_targets[num_val:]
    train_inputs = safe_np_cat([train_inputs, last_inputs])
    train_targets = safe_np_cat([train_targets, last_targets])
    train_split = DataSplit(train_inputs, train_targets)

    return train_split, val_split, test_split
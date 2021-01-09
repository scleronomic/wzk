import numpy as np

from wzk.numpy2 import block_shuffle
from wzk.dicts_lists_tuples import change_tuple_order


def n2train_test(n, split=0.2):

    if isinstance(split, float) and 0 < split < 1:
        n_test = int(np.round(n * split))
    elif isinstance(split, int) and 0 < split < n:
        n_test = split
    else:
        raise ValueError(f"Unknown value for 'split': {split} ; "
                         "must either be a float [0, 1] or an int [0, n_samples]")

    n_train = n - n_test
    return n_train, n_test


def train_test_split(*args,
                     split=0.2, shuffle=False, shuffle_block_size=1, seed=None):
    """
    if split == -1, use the same set for training and testing
    If shuffle=False, than the first n_train elements are used for training and the remaining n_test for testing
    return in same order as keras / TensorFlow
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    -> more arguments: (a_train, b_train, c_train, ...), (a_test, b_test, c_test, ...)
    """

    n = np.shape(args[0])[0]
    for a in args:
        assert np.shape(a)[0] == n

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        idx = block_shuffle(n, block_size=shuffle_block_size)
        args = [a[idx] for a in args]

    if split == -1:
        return change_tuple_order((a, a) for a in args)
    else:
        n_train, n_test = n2train_test(n=n, split=split)

        train_test_tuple = change_tuple_order(np.split(a, [n_train]) for a in args)
        return train_test_tuple  # ttt


def test_train_test_split():
    a = np.ones((10, 1))
    b = np.ones((10, 2)) * 2
    c = np.ones((10, 3)) * 3

    train_test_tuple = train_test_split(a, b, c, split=3)
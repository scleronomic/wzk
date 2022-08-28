import numpy as np


def binary_table(a, b):
    a, b = np.atleast_1d(a, b)
    x = np.zeros((3, 3), dtype=int)
    x[0, 0] = np.logical_and(~a, ~b).sum()
    x[0, 1] = np.logical_and(~a, b).sum()
    x[0, 2] = (~a).sum()
    x[1, 0] = np.logical_and(a, ~b).sum()
    x[1, 1] = np.logical_and(a, b).sum()
    x[1, 2] = a.sum()
    x[2, 0] = (~b).sum()
    x[2, 1] = b.sum()
    x[2, 2] = b.size
    return x


def logical_or(*args):
    b = np.logical_or(args[0], args[1])

    for a in args[2:]:
        b = np.logical_or(b, a)
    return b

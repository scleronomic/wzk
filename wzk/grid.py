import numpy as np

from wzk import np2


def limits2size(limits):
    return limits[:, 1] - limits[:, 0]


def limits2voxel_size(shape, limits, unify=True):
    voxel_size = limits2size(limits) / np.array(shape)
    if unify:
        voxel_size = np2.unify(x=voxel_size)
    return voxel_size


def __mode2offset(voxel_size, mode='c'):
    """Modes
        'c': center
        'b': boundary

    """
    if mode == 'c':
        return voxel_size / 2
    elif mode == 'b':
        return 0
    else:
        raise NotImplementedError(f"Unknown mode: '{mode}'")


def grid_x2i(x, limits, shape):
    """
    Get the indices of the grid cell at the coordinates 'x' in a grid with symmetric cells.
    Always use mode='boundary'
    """

    if x is None:
        return None
    # voxel_size = limits2voxel_size(shape=shape, limits=limits)
    voxel_size = np.diff(limits, axis=-1)[:, 0] / np.array(shape)
    lower_left = limits[:, 0]

    return np.asarray((x - lower_left) / voxel_size, dtype=int)


def grid_i2x(i, limits, shape, mode='c'):
    """
    Get the coordinates of the grid at the index "i" in a grid with symmetric cells.
    borders: 0 | 2 | 4 | 6 | 8 | 10
    centers: | 1 | 3 | 5 | 7 | 9 |
    """

    if i is None:
        return None
    voxel_size = limits2voxel_size(shape=shape, limits=limits)
    lower_left = limits[:, 0]

    offset = __mode2offset(voxel_size=voxel_size, mode=mode)
    return np.asarray(lower_left + offset + i * voxel_size, dtype=float)


def create_grid(ll: (float, float), ur: (float, float), n: (int, int), pad: (float, float)):
    ll, ur, n, pad = np.atleast_1d(ll, ur, n, pad)

    w = ur - ll
    s = (w - pad * (n - 1)) / n

    x = ll[0] + np.arange(n[0]) * (s[0] + pad[0])
    y = ll[1] + np.arange(n[1]) * (s[1] + pad[1])
    return (x, y), s

import numpy as np

from wzk.cpp.MinSphere import MinSphere
from wzk.dicts_lists_tuples import change_tuple_order, depth
from wzk.numpy2 import safe_scalar2array


def __min_sphere(x, r):
    n, d = np.shape(x)
    res = np.zeros(d + 1, dtype='f4', order='c')
    r = safe_scalar2array(r, shape=n)
    if d < 2 or 4 < d:
        raise ValueError
    if d == 2:
        MinSphere.min_sphere2(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    elif d == 3:
        MinSphere.min_sphere3(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    elif d == 4:
        MinSphere.min_sphere4(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    else:
        raise ValueError

    return res[:-1], res[-1]


def min_sphere(x, r):
    d = depth(x)
    if d == 2:
        return __min_sphere(x=x, r=r)
    elif d == 3:
        x2, r2 = change_tuple_order(__min_sphere(x=xx, r=rr) for xx, rr in zip(x, r))
        x2 = np.vstack(x2)
        r2 = np.array(r2)
        return x2, r2
    else:
        raise ValueError

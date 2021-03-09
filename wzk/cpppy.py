import numpy as np
import wzk.cpp.MinSphere.MinSphere as MinSphere


def min_sphere(x, r):
    n, d = np.shape(x)
    res = np.zeros(d + 1, dtype='f4', order='c')
    if d < 2 or 4 < d:
        raise ValueError
    if d == 2:
        MinSphere.min_sphere2(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    elif d == 3:
        MinSphere.min_sphere3(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    elif d == 4:
        MinSphere.min_sphere3(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    else:
        raise ValueError

    return res[:-1], res[-1]




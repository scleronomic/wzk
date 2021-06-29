import numpy as np

from wzk.numpy2 import allclose


def compare_arrays(a, b, axis=None, verbose=0,
                   rtol=1.e-5, atol=1.e-8,
                   title=''):
    eps = 1e-9
    all_equal = allclose(a=a, b=b, axis=axis, rtol=rtol, atol=atol)

    if not np.all(all_equal) or verbose > 0:
        print(title)
        print('shape: ', a.shape)
        print(f"nan: a {int(np.isnan(a).any())} b {int(np.isnan(b).any())}")
        print('maximal difference:', np.abs(a - b).max())
        print('variance difference:', np.std(a-b))
        ratio = a[np.abs(b) > eps] / b[np.abs(b) > eps]
        if np.count_nonzero(ratio) == 0:
            ratio = np.zeros(1)
        print('mean ratio:', ratio.mean())
        if not np.all(all_equal) and verbose >= 2:
            print(all_equal.astype(int))

    return np.all(all_equal)

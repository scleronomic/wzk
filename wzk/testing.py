import numpy as np

from wzk.numpy2 import allclose


def compare_arrays(a, b, axis=None, verbose=0,
                   rtol=1.e-5, atol=1.e-8,
                   title=''):

    all_equal = allclose(a=a, b=b, axis=axis, rtol=rtol, atol=atol)

    if not np.all(all_equal) or verbose > 0:
        print(title)
        print('shape: ', a.shape)
        print('maximal difference:', np.abs(a - b).max())
        if not np.all(all_equal) and verbose >= 2:
            print(all_equal.astype(int))

    return np.all(all_equal)
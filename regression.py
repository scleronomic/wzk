import numpy as np
from scipy import optimize


def leastsq(*, x, y,
            fit_fun=None, p0=None, err_fun=None):

    # quadratic function
    if fit_fun is None:
        def fit_fun(_p, _x):
            return _p[0] + _p[1]*_x + _p[2]*_x**2

    # squared error
    if err_fun is None:
        def err_fun(_p, _x, _y):
            return 0.5 * (fit_fun(_p=_p, _x=_x) - _y) ** 2

    if p0 is None:
        p0 = [1.0, 1.0, 1.0, 1.0, 1.0]

    p1, success = optimize.leastsq(err_fun, p0[:], args=(x, y))

    xx = np.linspace(start=x.min(), stop=x.max(), num=100)
    yy = fit_fun(p1, xx)

    return (p1, success), (xx, yy)

import numpy as np  # noqa

from wzk import np2, multiprocessing2, object2
from wzk.gd.optimizer import *


class GradientDescent(object2.CopyableObject):
    __slots__ = ('n_steps',                        # int                 | Number of iterations
                 'stepsize',                       # float               |
                 'opt',                            # Optimizer           | Adam, RMSProp, ...
                 'clip',                           # float[n_steps]      |
                 'clip_mode',                      # str                 | 'jump', 'clip', 'ignore'
                 'callback',                       # fun()               |
                 'limits',                         # fun()               |
                 'limits_mode',                    # str                 |
                 'n_processes',                    # int                 |
                 'use_loop_instead_of_processes',  # bool                |
                 'hesse_inv',                      # float[n_var][n_var] |
                 'hesse_weighting',                # float[n_steps]      |
                 'return_x_list',                  # bool                |  is this a suitable parameter? not really
                 'active_dims'                     # bool[n_var]         |
                 )

    def __init__(self, n_steps=100, stepsize=1, opt=Naive(), clip=0.1, n_processes=1):
        self.n_steps = n_steps
        self.stepsize = stepsize
        self.clip = clip
        self.clip_mode = 'value'
        self.limits_mode = 'clip'
        self.opt = opt
        self.active_dims = None

        self.n_processes = n_processes
        self.use_loop_instead_of_processes = False

        self.callback = None

        self.hesse_inv = None
        self.hesse_weighting = 0

        self.limits = lambda x: x
        # Logging
        self.return_x_list = False


# Gradient Descent
def gradient_descent_mp(x, fun, grad, gd):

    def gd_wrapper(xx):
        return gradient_descent(x=xx, fun=fun, grad=grad, gd=gd)

    return multiprocessing2.mp_wrapper(x, fun=gd_wrapper, n_processes=gd.n_processes)


def gradient_descent(x, fun, grad, gd):
    x = __x_wrapper(x)

    # b = np.zeros(len(x), dtype=bool)
    gd.opt.fun = fun
    if gd.active_dims is not None:
        active_dims = gd.active_dims
    else:
        active_dims = slice(x.shape[-1])

    # If the parameters aren't given for all steps, expand them
    if np.size(gd.clip) == 1:
        gd.clip = np.full(gd.n_steps, fill_value=float(gd.clip))

    if np.size(gd.hesse_weighting) == 1:
        gd.hesse_weighting = np.full(gd.n_steps, fill_value=float(gd.hesse_weighting))

    # grad_max_evolution = []
    if gd.return_x_list:
        x_list = np.zeros((gd.n_steps,) + x.shape)
        f_list = np.zeros((gd.n_steps, len(x)))
    else:
        x_list = None
        f_list = None

    # Gradient Descent Loop
    for i in range(gd.n_steps):
        j = grad(x=x, i=i)

        # Correct via an approximated hesse function
        if gd.hesse_inv is not None and gd.hesse_weighting[i] > 0:
            h = (gd.hesse_inv[np.newaxis, ...] @ j.reshape(-1, gd.hesse_inv.shape[-1], 1)).reshape(j.shape)
            j = j * (1 - gd.hesse_weighting[i]) + h * gd.hesse_weighting[i]
            pass

        if gd.callback is not None:
            j = gd.callback(x=x.copy(), jac=j.copy())  # , count=o) -> callback function handles count

        v = gd.opt.update(x=x, v=j)
        v = np2.clip2(v, clip=gd.clip[i], mode=gd.clip_mode)
        x[..., active_dims] += v[..., active_dims]

        x = gd.limits(x)

        if gd.return_x_list:
            x_list[i] = x
            f_list[i] = fun(x)  # only for debugging, is inefficient to call separately

    f = fun(x=x)

    if gd.return_x_list:
        return x, f, (x_list, f_list)
    else:
        return x, f


def __x_wrapper(x):
    x = x.copy()
    if x.ndim == 1:
        x = x[np.newaxis, :]

    return x

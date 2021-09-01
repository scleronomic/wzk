import numpy as np

from wzk.math2 import angle2minuspi_pluspi


def full2inner(x):
    return x[..., 1:-1, :]


def inner2full(inner, start=None, end=None):
    n_samples = inner.shape[0]

    def __repeat(x):
        x = np.atleast_2d(x)
        if inner.ndim != 3:
            return x
        else:
            if x.ndim == 2:
                x = x[:, np.newaxis, :]

            return x.repeat(n_samples // x.shape[0], axis=0)

    if start is not None:
        start = __repeat(start)
        inner = np.concatenate((start, inner), axis=-2)

    if end is not None:
        end = __repeat(end)
        inner = np.concatenate((inner, end), axis=-2)

    return inner


def full2start_end(x, squeeze=True):
    if squeeze:
        return x[..., 0, :], x[..., -1, :]
    else:
        return x[..., :1, :], x[..., -1:, :]


def x_mode(x, mode=None):  # TODO better name
    if mode is None or mode == 'full':
        return x
    elif mode == 'inner':
        return x[..., 1:-1, :]
    elif mode == 'wo_start':
        return x[..., 1:, :]
    elif mode == 'wo_end':
        return x[..., :-1, :]
    else:
        raise ValueError


def flat2full(flat, n_dof):
    if flat.ndim == 1:
        flat = flat.reshape(-1, n_dof)
    elif flat.ndim == 2:
        flat = flat.reshape(len(flat), -1, n_dof)
    return flat


def full2flat(x):
    if x.ndim == 2:
        x = x.reshape(-1)
    elif x.ndim == 3:
        x = x.reshape(len(x), -1)
    return x


def periodic_dof_wrapper(x,
                         is_periodic=None):
    if is_periodic is not None and any(is_periodic):
        x[..., is_periodic] = angle2minuspi_pluspi(x[..., is_periodic])
    return x


def get_steps(q,
              is_periodic=None):
    return periodic_dof_wrapper(np.diff(q, axis=-2), is_periodic=is_periodic)


def get_steps_norm(q,
                   is_periodic=None):
    return np.linalg.norm(get_steps(q=q, is_periodic=is_periodic), axis=-1)


def get_substeps(x, n,
                 is_periodic=None, include_start=True):

    *shape, m, d = x.shape

    # only fill in substeps if the the number is greater 1,
    if n <= 1 or m <= 1:
        if include_start:
            return x
        else:  # How often does this happen? once?: fill_linear_connection
            return x[..., 1:, :]

    steps = get_steps(q=x, is_periodic=is_periodic)
    delta = (np.arange(n-1, -1, -1)/n) * steps[..., np.newaxis]
    x_ss = x[..., 1:, :, np.newaxis] - delta
    x_ss = np.swapaxes(x_ss, -2, -1)
    x_ss = x_ss.reshape(shape + [(m-1) * n, d])

    if include_start:
        x_ss = np.concatenate((x[..., :1, :], x_ss), axis=-2)

    x_ss = periodic_dof_wrapper(x_ss, is_periodic=is_periodic)
    return x_ss


def get_substeps_adjusted(x, n,
                          is_periodic=None, weighting=None):

    *shape, m, d = x.shape

    if shape:
        raise NotImplementedError

    m1 = m - 1
    steps = get_steps(q=x, is_periodic=is_periodic)

    if weighting is not None:
        steps *= weighting

    # Distribute the waypoints equally along the linear sequences of the initial path
    steps_length = np.linalg.norm(steps, axis=-1)
    if np.sum(steps_length) == 0:
        relative_steps_length = np.full(m1, fill_value=1 / m1)
    else:
        relative_steps_length = steps_length / np.sum(steps_length)

    # Adjust the number of waypoints for each step to make the initial guess as equally spaced as possible
    n_sub_exact = relative_steps_length * (n - 1)
    n_sub = np.round(n_sub_exact).astype(int)

    # If the number of points do not match, change the substeps where the rounding was worst
    n_diff = (n-1) - np.sum(n_sub)
    if n_diff != 0:
        n_sub_acc = n_sub_exact - n_sub
        n_sub_acc = 0.5 + np.sign(n_diff) * n_sub_acc
        idx = np.argsort(n_sub_acc)[-np.abs(n_diff):]
        n_sub[idx] += np.sign(n_diff)

    n_sub_cs = np.hstack((0, n_sub.cumsum())) + 1

    # Add the linear interpolation between the random waypoints step by step for each dimension
    x_n = np.empty((n, d))
    x_n[0, :] = x[0, :]
    for i in range(m1):
        x_n[n_sub_cs[i]:n_sub_cs[i + 1], :] = \
            get_substeps(x=x[i:i + 2, :], n=n_sub[i], is_periodic=is_periodic, include_start=False)

    x_n = periodic_dof_wrapper(x=x_n, is_periodic=is_periodic)
    return x_n


def order_path(x, start=None, end=None, infinity_joints=None, weights=1.):
    """
    Order the points given by 'x' [2d: (n, d)] according to a weighted euclidean distance
    so that always the nearest point comes next.
    Start with the first point in the array and end with the last if 'x_start' or 'x_end' aren't given.
    """

    # Handle different input combinations
    d = None
    if start is None:
        start = x[..., 0, :]
        x = np.delete(x, 0, axis=-2)
    else:
        d = np.size(start)

    if x is None:
        n = 0
    else:
        n, d = x.shape

    if end is None:
        x_o = np.zeros((n + 1, d))
    else:
        x_o = np.zeros((n + 2, d))
        x_o[-1, :] = end.ravel()

    # Order the points, so that always the nearest is visited next, according to the euclidean distance
    x_o[0, :] = start.ravel()
    for i in range(n):
        x_diff = np.linalg.norm(periodic_dof_wrapper(x - start, is_periodic=infinity_joints) * weights, axis=-1)
        i_min = np.argmin(x_diff)
        x_o[1 + i, :] = x[i_min, :]
        start = x[i_min, :]
        x = np.delete(x, i_min, axis=-2)

    return x_o

#
# DERIVATIVES


def d_substeps__dx(n, order=0):
    """
    Get the dependence of substeps (') on the outer way points (x).
    The substeps are placed linear between the waypoints.
    To prevent counting points double one step includes only one of the two endpoints
    This gives a symmetric number of steps but ignores either the start or the end in
    the following calculations.

         x--'--'--'--x---'---'---'---x---'---'---'---x--'--'--'--x--'--'--'--x
    0:  {>         }{>            } {>            } {>         }{>         }
    1:     {         <} {            <} {            <}{         <}{         <}

    Ordering of the waypoints into a matrix:
    0:
    s00 s01 s02 s03 -> step 0 (with start)
    s10 s11 s12 s13
    s20 s21 s22 s23
    s30 s31 s32 s33
    s40 s41 s42 s43 -> step 4 (without end)

    1: (shifting by one, and reordering)
    s01 s02 s03 s10 -> step 0 (without start)
    s11 s12 s13 s20
    s21 s22 s23 s30
    s31 s32 s33 s40
    s41 s42 s43 s50 -> step 4 (with end)


    n          # way points
    n-1        # steps
    n_s        # intermediate points per step
    (n-1)*n_s  # substeps (total)

    n_s = 5
    jac0 = ([[1. , 0.8, 0.6, 0.4, 0.2],  # following step  (including way point (1.))
             [0. , 0.2, 0.4, 0.6, 0.8]]) # previous step  (excluding way point (1.))

    jac1 = ([[0.2, 0.4, 0.6, 0.8, 1. ],  # previous step  (including way point (2.))
             [0.8, 0.6, 0.4, 0.2, 0. ]])  # following step  (excluding way point (2.))

    """

    if order == 0:
        jac = (np.arange(n) / n)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]
    else:
        jac = (np.arange(start=n - 1, stop=-1, step=-1) / n)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]

    # jac /= n_substeps  # FIXME sum = 1, the influence of the substeps on the waypoints is independent from its numbers
    # Otherwise the center would always be 1, no matter how many substeps -> better this way
    # print('substeps normalization')
    return jac


def combine_d_substeps__dx(d_dxs, n):
    # Combine the jacobians of the sub-way-points (joints) to the jacobian for the optimization variables
    if n <= 1:
        return d_dxs

    if d_dxs.ndim == 3:
        n_samples, n_wp_ss, n_dof = d_dxs.shape
        d_dxs = d_dxs.reshape(n_samples, n_wp_ss//n, n, n_dof)
        ss_jac = d_substeps__dx(n=n, order=1)[np.newaxis, np.newaxis, ..., np.newaxis]
        d_dx = np.einsum('ijkl, ijkl -> ijl', d_dxs, ss_jac[:, :, 0, :, :])
        d_dx[:, :-1, :] += np.einsum('ijkl, ijkl -> ijl', d_dxs[:, 1:, :, :], ss_jac[:, :, 1, :, :])
        return d_dx
    else:
        raise ValueError

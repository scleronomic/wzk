import numpy as np
from scipy.stats import norm

from wzk import np2, limits as limits2, math2, grid


def p_normal_skew(x, loc=0.0, scale=1.0, a=0.0):
    t = (x - loc) / scale
    return 2 * norm.pdf(t) * norm.cdf(a*t)


def normal_skew_int(loc=0.0, scale=1.0, a=0.0, low=None, high=None, size=1):
    if low is None:
        low = loc-10*scale
    if high is None:
        high = loc+10*scale+1

    p_max = p_normal_skew(x=loc, loc=loc, scale=scale, a=a)

    samples = np.zeros(np.prod(size))

    for i in range(int(np.prod(size))):
        while True:
            x = np.random.randint(low=low, high=high)
            if np.random.rand() <= p_normal_skew(x, loc=loc, scale=scale, a=a) / p_max:
                samples[i] = x
                break

    samples = samples.astype(int)
    if size == 1:
        samples = samples[0]
    return samples


def random_uniform_ndim(low, high, shape=None):
    n_dim = np.shape(low)[0]
    return np.random.uniform(low=low, high=high, size=np2.shape_wrapper(shape) + (n_dim,))


def noise(shape, scale, mode="normal"):
    shape = np2.shape_wrapper(shape)

    if mode == "constant":  # could argue that this is no noise
        return np.full(shape=shape, fill_value=+scale)
    if mode == "plusminus":
        return np.where(np.random.random(shape) < 0.5, -scale, +scale)
    if mode == "uniform":
        return np.random.uniform(low=-scale, high=+scale, size=shape)
    elif mode == "normal":
        return np.random.normal(loc=0, scale=scale, size=shape)
    else:
        raise ValueError(f"Unknown mode '{mode}'")


def get_n_in2(n_in, n_out,
              n_total, n_current,
              safety_factor=1.01,
              max_factor=128):

    if n_out == 0:
        n_in2 = n_in*2
    else:
        n_in2 = (n_total - n_current) * n_in / n_out
    # n_in2 = int(n_in2)
    # print(f"total:{n_total} | current:{n_current} | new:{n_out}/{n_in} -> {n_in2}")

    n_in2 = min(n_total * max_factor, n_in2)  # otherwise it can grow up to 2**maxiter
    n_in2 = max(int(np.ceil(safety_factor * n_in2)), 1)
    return n_in2


def fun2n(fun, n,
          max_iter=100, max_factor=128, verbose=0):

    x = x_new = fun(n)

    n_in = n
    for i in range(max_iter):

        n_in = get_n_in2(n_in=n_in, n_out=len(x_new), n_total=n, n_current=len(x), max_factor=max_factor)

        x_new = fun(n_in)
        x = np.concatenate([x, x_new], axis=0)

        if verbose > 0:
            print(f"{i}: total:{n} | current:{len(x)} | new:{len(x_new)}/{n_in}")

        if len(x) >= n:
            return x[:n]

    else:
        Warning(f"Maximum number of iterations reached! Only {len(x)} samples could be generated")
        return x


def choose_from_sections(n_total, n_sections, n_choose_per_section, flatten=True):
    np.random.seed()   # TODO somewhere was something not random
    n_i = np.array_split(np.arange(n_total), n_sections)

    n_choose_per_section = np2.scalar2array(n_choose_per_section, shape=n_sections)
    i = [np.random.choice(arr, size=m) for arr, m in zip(n_i, n_choose_per_section)]
    if flatten:
        i = np.concatenate(i, axis=0)
    return i


def choose_from_uniform_grid(x, n):
    n_samples, n_dim = x.shape

    limits = limits2.x2limits(x=x, axis=1)
    limits = limits2.make_limits_symmetrical(limits=limits)

    def fun(_s):
        _shape = (_s,) * n_dim
        _i = grid.grid_x2i(x=x, limits=limits, shape=_shape)
        _u = np.unique(_i, axis=0)
        return len(_u) - n

    s = math2.bisection(f=fun, a=10, b=100, tol=0, verbose=0)
    shape = (int(np.ceil(s)),) * n_dim

    ix = grid.grid_x2i(x=x, limits=limits, shape=shape)
    u, inv = np.unique(ix, axis=0, return_inverse=True)
    iu = np.random.choice(np.arange(len(u)), n, replace=False)

    i = [np.random.choice(np.nonzero(inv == j)[0]) for j in iu]
    return np.array(i, dtype=int)

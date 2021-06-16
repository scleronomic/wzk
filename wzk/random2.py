import numpy as np
from scipy.stats import norm

from wzk.numpy2 import shape_wrapper


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
    return np.random.uniform(low=low, high=high, size=shape_wrapper(shape) + (n_dim,))


def noise(shape, scale, mode='normal'):
    shape = shape_wrapper(shape)

    if mode == 'constant':  # could argue that this is no noise
        return np.full(shape=shape, fill_value=+scale)
    if mode == 'plusminus':
        return np.where(np.random.random(shape) < 0.5, -scale, +scale)
    if mode == 'uniform':
        return np.random.uniform(low=-scale, high=+scale, size=shape)
    elif mode == 'normal':
        return np.random.normal(loc=0, scale=scale, size=shape)
    else:
        raise ValueError(f"Unknown mode {mode}")

import numpy as np
from itertools import product

from wzk.numpy2 import axis_wrapper, insert
from wzk.dicts_lists_tuples import atleast_tuple

# a/b = (a+b) / a -> a / b =
golden_ratio = (np.sqrt(5.0) + 1) / 2


def number2digits(num):
    return [int(x) for x in str(num)]


# Normalize
def normalize_01(x, low=None, high=None, axis=None):
    if low is None:
        low = np.min(x, axis=axis, keepdims=True)

    if high is None:
        high = np.max(x, axis=axis, keepdims=True)

    return (x-low) / (high-low)


def denormalize_01(x, low, high):
    return x * (high - low) + low


def normalize_11(x, low, high):
    """
    Normalize [low, high] to [-1, 1]
    low and high should either be scalars or have the same dimension as the last dimension of x
    """
    return 2 * (x - low) / (high - low) - 1


def denormalize_11(x, low, high):
    """
    Denormalize [-1, 1] to [low, high]
    low and high should either be scalars or have the same dimension as the last dimension of x
    """
    return (x + 1) * (high - low)/2 + low


def euclidean_norm(arr, axis=-1, squared=False):
    if squared:
        return (arr**2).sum(axis=axis)
    else:
        return np.sqrt((arr**2).sum(axis=axis))


def discretize(x, step):

    if np.isinf(step) or np.isnan(step):
        return x

    difference = x % step  # distance to the next discrete value

    if isinstance(x, (int, float)):
        if difference > step / 2:
            return x - (difference - step)
        else:
            return x - difference

    else:
        difference[difference > step / 2] -= step  # round correctly
        return x - difference


def dnorm_dx(x, x_norm=None):
    """ ∂ |x| / ∂ x
     normalization over last dimension
     """
    if x_norm is None:
        x_norm = np.linalg.norm(x, axis=-1)

    dn_dx = x.copy()
    i = x_norm != 0  # All steps where there is movement between t, t+1
    dn_dx[i, :] = dn_dx[i, :] / x_norm[i][..., np.newaxis]
    return dn_dx


def dxnorm_dx(x, return_norm=False):
    """
    ∂ (x/|x|) / ∂ x
    normalization over last dimension

    Calculate Jacobian
      xn       =           x * (x^2 + y^2 + z^2)^(-1/2)
    d xn / d x = (y^2 + z^2) * (x^2 + y^2 + z^2)^(-3/2)
    d yn / d y = (x^2 + y^2) * (x^2 + y^2 + z^2)^(-3/2)
    d zn / d z=  (x^2 + z^2) * (x^2 + y^2 + z^2)^(-3/2)

    Pattern of numerator
    X123
    0X23
    01X3
    012X

    d xn / d y = -(x*y) * (x^2 + y^2 + z^2)^(-3/2)
    d xn / d z = -(x*z) * (x^2 + y^2 + z^2)^(-3/2)

    jac = [[dxn/dx, dxn/dy, dxn/dz]
           [dyn/dx, dyn/dy, dyn/dz]
           [dzn/dx, dzn/dy, dzn/dz]

    """

    n_dim = x.shape[-1]

    off_diag_idx = [[j for j in range(n_dim) if i != j] for i in range(n_dim)]

    dxn_x = np.empty(x.shape + x.shape[-1:])
    x_squared = x**2

    # Diagonal
    dxn_x[:, np.arange(n_dim), np.arange(n_dim)] = x_squared[..., off_diag_idx].sum(axis=-1)

    # Off-Diagonal
    dxn_x[:, np.arange(n_dim)[:, np.newaxis], off_diag_idx] = -x[..., np.newaxis] * x[:, off_diag_idx]

    dxn_x *= (x_squared.sum(axis=-1, keepdims=True)**(-3/2))[..., np.newaxis]

    if return_norm:
        x /= np.sqrt(x_squared.sum(axis=-1, keepdims=True))
        return x, dxn_x
    else:
        return dxn_x


# Smooth
def smooth_step(x):
    """
    https://en.wikipedia.org/wiki/Smoothstep
    Interpolation which has zero 1st-order derivatives at x = 0 and x = 1,
     ~ cubic Hermite interpolation with clamping.
    """
    res = -2 * x**3 + 3 * x**2
    return np.clip(res, 0, 1)


def smoother_step(x):
    """
    https://en.wikipedia.org/wiki/Smoothstep+
    Ken Perlin suggests an improved version of the smooth step function,
    which has zero 1st- and 2nd-order derivatives at x = 0 and x = 1
    """
    res = +6 * x**5 - 15 * x**4 + 10 * x**3
    return np.clip(res, 0, 1)


# Divisors
def divisors(n, with_1_and_n=False):
    """
    https://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number#171784
    """

    # Get factors and their counts
    factors = {}
    nn = n
    i = 2
    while i*i <= nn:
        while nn % i == 0:
            if i not in factors:
                factors[i] = 0
            factors[i] += 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = 1

    primes = list(factors.keys())

    # Generates factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k+1)
            prime = primes[k]
            for _factor in rest:
                prime_to_i = 1
                # Prime_to_i iterates prime**o values, o being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield _factor * prime_to_i
                    prime_to_i *= prime

    if with_1_and_n:
        return list(generate(0))
    else:
        return list(generate(0))[1:-1]


def get_mean_divisor_pair(n):
    """
    Calculate the 'mean' pair of divisors. The two divisors should be as close as possible to the sqrt(n).
    The smaller divisor is the first value of the output pair
    10 -> 2, 5
    20 -> 4, 5
    24 -> 4, 6
    25 -> 5, 5
    30 -> 5, 6
    40 -> 5, 8
    """
    assert isinstance(n, int)
    assert n >= 1

    div = divisors(n)
    if len(div) == 0:
        return 1, n

    div.sort()

    # if numbers of divisors is odd -> n = o * o : power number
    if len(div) % 2 == 1:
        idx_center = len(div) // 2
        return div[idx_center], div[idx_center]

    # else get the two numbers at the center
    else:
        idx_center_plus1 = len(div) // 2
        idx_center_minus1 = idx_center_plus1 - 1
        return div[idx_center_minus1], div[idx_center_plus1]


def get_divisor_safe(numerator, denominator):
    divisor = numerator / denominator
    divisor_int = int(divisor)

    assert divisor_int == divisor
    return divisor_int


def doubling_factor(small, big):
    return np.log2(big / small)


def modulo(x, low, high):
    return (x - low) % (high - low) + low


def angle2minuspi_pluspi(x):
    return modulo(x=x, low=-np.pi, high=+np.pi)
    # modulo is faster for larger arrays, for small ones they are similar but arctan is faster in this region
    #  -> as always you have to make an trade-off
    # return np.arctan2(np.sin(x), np.cos(x))


def log_b(x, base=np.e):
    return np.log(x) / np.log(base)


def assimilate_orders_of_magnitude(a, b, base=10):
    a_mean = np.abs(a).mean()
    b_mean = np.abs(b).mean()
    np.log1p()
    a_mean_log = np.log(a_mean)
    b_mean_log = np.log1(b_mean)

    c = np.power(base, (a_mean_log + b_mean_log) / 2)

    aa = a * c / a_mean
    bb = b * c / b_mean

    return aa, bb, c


# Derivative
def numeric_derivative(fun, x, eps=1e-5, axis=-1, mode='central',
                       diff=None,
                       **kwargs_fun):
    """
    Use central, forward or backward difference scheme to calculate the
    numeric derivative of function at point x.
    'axis' indicates the dimensions of the free variables.
    The result has the shape f(x).shape + x.shape[axis]
    """
    axis = axis_wrapper(axis=axis, n_dim=np.ndim(x))

    f_x = fun(x, **kwargs_fun)
    fun_shape = np.shape(f_x)
    var_shape = atleast_tuple(np.array(np.shape(x))[(axis,)])
    eps_mat = np.empty_like(x, dtype=float)

    derv = np.empty(fun_shape + var_shape)

    if diff is None:
        def diff(a, b):
            return a - b

    def update_eps_mat(_idx):
        eps_mat[:] = 0
        insert(eps_mat, val=eps, idx=_idx, axis=axis)

    for idx in product(*(range(s) for s in var_shape)):
        update_eps_mat(_idx=idx)

        if mode == 'central':
            derv[(Ellipsis,) + idx] = diff(fun(x + eps_mat, **kwargs_fun),
                                           fun(x - eps_mat, **kwargs_fun)) / (2 * eps)

        elif mode == 'forward':
            derv[(Ellipsis, ) + idx] = diff(fun(x + eps_mat, **kwargs_fun), f_x) / eps

        elif mode == 'backward':
            derv[(Ellipsis, ) + idx] = diff(f_x, fun(x - eps_mat, **kwargs_fun)) / eps

    return derv


# Magic
def magic(n):
    """
    Equivalent of the MATLAB function:
    M = magic(n) returns an n-by-n matrix constructed from the integers 1 through n2 with equal row and column sums.
    https://stackoverflow.com/questions/47834140/numpy-equivalent-of-matlabs-magic
    """

    n = int(n)

    if n < 1:
        raise ValueError('Size must be at least 1')
    if n == 1:
        return np.array([[1]])
    elif n == 2:
        return np.array([[1, 3], [4, 2]])
    elif n % 2 == 1:
        p = np.arange(1, n+1)
        return n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
    elif n % 4 == 0:
        j = np.mod(np.arange(1, n+1), 4) // 2
        k = j[:, None] == j
        m = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
        m[k] = n*n + 1 - m[k]
    else:
        p = n//2
        m = magic(p)
        m = np.block([[m, m+2*p*p], [m+3*p*p, m+p*p]])
        i = np.arange(p)
        k = (n-2)//4
        j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
        m[np.ix_(np.concatenate((i, i+p)), j)] = m[np.ix_(np.concatenate((i+p, i)), j)]
        m[np.ix_([k, k+p], [0, k])] = m[np.ix_([k+p, k], [0, k])]
    return m


# Clustering
def k_farthest_neighbors(x, k, weighting=None):
    n = len(x)

    m_dist = x[np.newaxis, :, :] - x[:, np.newaxis, :]
    weighting = np.ones(x.shape[-1]) if weighting is None else weighting
    m_dist = ((m_dist * weighting)**2).sum(axis=-1)

    cum_dist = m_dist.sum(axis=-1)

    idx = [np.argmax(cum_dist)]

    for i in range(k-1):
        m_dist_cur = m_dist[idx]
        m_dist_cur_sum = m_dist_cur.sum(axis=0)
        # m_dist_cur_std = np.std(m_dist_cur, axis=0)
        obj = m_dist_cur_sum   # + (m_dist_cur_std.max() - m_dist_cur_std) * 1000
        idx_new = np.argsort(obj)[::-1]
        for j in range(n):
            if idx_new[j] not in idx:
                idx.append(idx_new[j])
                break

    return np.array(idx)


def test_k_farthest_neighbors():
    x = np.random.random((200, 2))
    k = 10
    idx = k_farthest_neighbors(x=x, k=k)

    from wzk import new_fig
    fig, ax = new_fig(aspect=1)
    ax.plot(*x.T, ls='', marker='o', color='b', markersize=5, alpha=0.5)
    ax.plot(*x[idx, :].T, ls='', marker='x', color='r', markersize=10)


# Combinatorics
def binomial(n, k):
    return np.math.factorial(n) // np.math.factorial(k) // np.math.factorial(n - k)


def random_subset(n, k, m, dtype=np.uint16):
    assert n == np.array(n, dtype=dtype)
    return np.array([np.random.choice(n, k, replace=False) for _ in range(m)]).astype(np.uint16)

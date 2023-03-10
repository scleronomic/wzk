import numpy as np
from scipy.sparse import csr_matrix
from itertools import product
from wzk import dtypes

np.core.arrayprint._line_width = 80


class DummyArray:
    """Allows indexing but always returns the same 'dummy' value"""

    def __init__(self, arr, shape):
        self.arr = arr
        self.shape = shape

    def __assert_int(self, item, i):
        assert item in range(-self.shape[i], self.shape[i])

    def __assert_slice(self, item, i):
        pass

    def __assert_ellipsis(self, item, i):
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            self.__assert_int(item=item, i=0)
        elif isinstance(item, slice):
            self.__assert_slice(item=item, i=0)
        elif isinstance(item, type(...)):
            self.__assert_ellipsis(item=item, i=0)

        else:
            assert len(item) == len(self.shape), f"Incompatible index {item} for array with shape {self.shape}"
            for i, item_i in enumerate(item):
                if isinstance(item_i, int):
                    self.__assert_int(item=item_i, i=i)
                elif isinstance(item_i, slice):
                    self.__assert_slice(item=item_i, i=i)
                elif isinstance(item, type(...)):
                    self.__assert_ellipsis(item=item, i=i)
                else:
                    raise ValueError

        return self.arr


def initialize_array(shape, mode="zeros", dtype=None, order="c"):
    if mode == "zeros":
        return np.zeros(shape, dtype=dtype, order=order)
    elif mode == "ones":
        return np.ones(shape, dtype=dtype, order=order)
    elif mode == "empty":
        return np.empty(shape, dtype=dtype, order=order)
    elif mode == "random":
        return np.random.random(shape).astype(dtype=dtype, order=order)
    else:
        raise ValueError(f"Unknown initialization method {mode}")


# Checks
def np_isinstance(o, c):
    """
    # Like isinstance if o is not np.ndarray
    np_isinstance(('this', 'that'), tuple)  # True
    np_isinstance(4.4, int)                 # False
    np_isinstance(4.4, float)               # True

    # else
    np_isinstance(np.ones(4, dtype=int), int)    # True
    np_isinstance(np.ones(4, dtype=int), float)  # False
    np_isinstance(np.full((4, 4), 'bert'), str)  # True
    """

    if isinstance(o, np.ndarray):
        c = (dtypes.c2np[cc] for cc in c) if isinstance(c, tuple) else dtypes.c2np[c]
        return isinstance(o.flat[0], c)

    else:
        return isinstance(o, c)


# object <-> numeric
def object2numeric_array(arr):
    s = np.shape(arr)
    arr = np.array([v for v in np.ravel(arr)])
    arr = np.reshape(arr, s + np.shape(arr)[1:])
    return arr


def numeric2object_array(arr):
    n = arr.shape[0]
    arr_obj = np.zeros(n, dtype=object)
    for i in range(n):
        arr_obj[i] = arr[i]

    return arr_obj


def array2array(a, shape):
    a = np.atleast_1d(a)
    if np.size(a) == 1:
        return scalar2array(a.item(), shape=shape)

    b = np.empty(shape, dtype=a.dtype)
    s = align_shapes(shape, a.shape)
    s = tuple([slice(None) if ss == 1 else np.newaxis for ss in s])
    b[:] = a[s]
    return b


def max_size(*args):
    return int(np.max([np.size(a) for a in args]))


def max_len(*args):
    return int(np.max([len(a) for a in args]))


# scalar <-> matrix
def scalar2array(*val_or_arr, shape, squeeze=True, safe=True):
    # standard numpy broadcasting rules apply
    shape = shape_wrapper(shape)

    res = []
    for voa in val_or_arr:
        try:
            res_i = np.empty(shape=shape, dtype=type(voa))
            res_i[:] = np.array(voa).copy()
            res.append(res_i)

        except ValueError:
            if safe:
                assert np.all(np.shape(voa) == shape), f"{np.shape(voa)} != {shape}"
            res.append(voa)

    if len(res) == 1 and squeeze:
        return res[0]
    else:
        return res


# def try_to_make_equal(*args):
#     pass
    # TODO find a good general approach to make apples and pears the same size
    # x = np.ones((10, 3))
    # y = "red"
    # -> y = ["red"]*10

    # x = np.ones((10, 3))
    # y = ["red", "blue"]
    # -> fails

    # x = np.ones(3)
    # y = ["blue"]
    # -> y = ["blue"]*3

    # x = np.ones(1, 3)
    # y = ["blue"]
    # -> works

    # args = np.atleast_1d(*args)
    #
    # n = np2.max_len(*args)
    # args = np2.scalar2array(*args)
    # return args


def unify(x):
    x = np.atleast_1d(x)
    assert np.allclose(x - x.mean(), np.zeros_like(x))
    x_mean = np.mean(x)
    return x_mean.astype(x.dtype)


def flatten_without_last(x):
    return np.reshape(x, (-1, np.shape(x)[-1]))


def flatten_without_first(x):
    return np.reshape(x, (np.shape(x)[0], -1))


def args2arrays(*args):
    return [np.array(a) for a in args]


# Shapes and Axis
def axis_wrapper(axis, n_dim, invert=False):
    if axis is None:
        axis = np.arange(n_dim)

    axis = np.atleast_1d(axis)
    axis %= n_dim
    np.sort(axis)

    if invert:
        return tuple(np.setxor1d(np.arange(n_dim), axis).astype(int))
    else:
        return tuple(axis)


def shape_wrapper(shape=None) -> tuple:
    """
    Note the inconsistent usage of shape / shape as function arguments in numpy.
    https://stackoverflow.com/questions/44804965/numpy-size-vs-shape-in-function-arguments
    -> use shape
    """
    if shape is None:
        return ()

    elif isinstance(shape, (int, np.int_)):
        return int(shape),

    elif isinstance(shape, tuple):
        return shape

    elif isinstance(shape, (list, np.ndarray)):
        return tuple(shape)

    else:
        raise ValueError(f"Unknown 'shape': {shape}")


def get_max_shape(*args):
    shapes = [-1 if a is None else np.shape(a) for a in args]
    sizes = [np.prod(shape) for shape in shapes]
    return shapes[np.argmax(sizes)]


def get_subshape(shape, axis):
    return tuple(np.array(shape)[np.array(axis)])


def align_shapes(a, b):
    """
    a = np.array((2, 3, 4, 3, 5, 1))
    b = np.array((3, 4, 3))
    -> array([-1, 1, 1, 1, -1, -1])
    """
    idx = find_subarray(a=a, b=b).item()
    aligned_shape = np.full(shape=len(a), fill_value=-1, dtype=int)
    aligned_shape[idx:idx + len(b)] = 1
    return aligned_shape


def repeat2new_shape(img, new_shape):
    reps = np.ceil(np.array(new_shape) / np.array(img.shape)).astype(int)
    for i in range(img.ndim):
        img = np.repeat(img, repeats=reps[i], axis=i)

    img = img[slicen(end=new_shape)]
    return img


def change_shape(arr, mode="even"):
    s = np.array(arr.shape)

    if mode == "even":
        s_new = (s + s % 2)
    elif mode == "odd":
        s_new = (s // 2) * 2 + 1
    else:
        raise ValueError(f"Unknown mode {mode}")

    arr_odd = np.zeros(s_new, dtype=arr.dtype)
    fill_with_air_left(arr=arr, out=arr_odd)
    return arr_odd


def fill_with_air_left(arr, out):
    assert arr.ndim == out.ndim
    out[slicen(end=arr.shape)] = arr


# replace numpy defaults with more general versions

def __argfun(a, axis, fun):
    axis = axis_wrapper(axis=axis, n_dim=a.ndim)
    if len(axis) == 1:
        return fun(a, axis=axis)

    elif len(axis) == a.ndim:
        np.unravel_index(fun(a), shape=a.shape)
        # TODO, what happens here?
    else:
        axis_inv = axis_wrapper(axis=axis, n_dim=a.ndim, invert=True)
        shape_inv = get_subshape(shape=a.shape, axis=axis_inv)
        shape = get_subshape(shape=a.shape, axis=axis)

        a2 = np.transpose(a, axes=np.hstack((axis_inv, axis))).reshape(shape_inv + (-1,))
        idx = fun(a2, axis=-1)
        idx = np.array(np.unravel_index(idx, shape=shape))

        return np.transpose(idx, axes=np.roll(np.arange(idx.ndim), -1))


def min_size(*args):
    return int(np.min([np.size(a) for a in args]))


def argmax(a, axis=None):
    return __argfun(a=a, axis=axis, fun=np.argmax)


def argmin(a, axis=None):
    return __argfun(a=a, axis=axis, fun=np.argmin)


def allclose(a, b, rtol=1.e-5, atol=1.e-8, axis=None):
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    axis = np.array(axis_wrapper(axis=axis, n_dim=a.ndim))
    assert len(axis) <= len(a.shape)
    if np.isscalar(a) and np.isscalar(b):
        return np.allclose(a, b)
    shape = np.array(a.shape)[axis]
    bool_arr = np.zeros(shape, dtype=bool)
    for i in product(*(range(s) for s in shape)):
        bool_arr[i] = np.allclose(extract(a, idx=i, axis=axis),
                                  extract(b, idx=i, axis=axis),
                                  rtol=rtol, atol=atol)

    return bool_arr


def __wrapper_pair2list_fun(*args, fun):
    assert len(args) >= 2
    res = fun(args[0], args[1])
    for a in args[2:]:
        res = fun(res, a)
    return res


def minimum(*args):
    return __wrapper_pair2list_fun(*args, fun=np.minimum)


def maximum(*args):
    return __wrapper_pair2list_fun(*args, fun=np.maximum)


def logical_or(*args):
    return __wrapper_pair2list_fun(*args, fun=np.logical_or)


def logical_and(*args):
    return __wrapper_pair2list_fun(*args, fun=np.logical_and)


# -------

def delete_args(*args, obj, axis=None):
    return tuple(np.delete(a, obj=obj, axis=axis) for a in args)


def remove_outside_limits(x, limits, safety_factor=None, return_idx=False):
    if safety_factor is not None:
        limits = add_safety_limits(limits=limits, factor=safety_factor)

    below_lower = np.sum(x < limits[:, 0], axis=-1) > 0
    above_upper = np.sum(x > limits[:, 1], axis=-1) > 0
    outside_limits = np.logical_or(below_lower, above_upper)
    inside_limits = ~outside_limits
    x = x[inside_limits]
    if return_idx:
        return x, inside_limits
    return x


def __fill_index_with(idx, axis, shape, mode="slice"):
    """
    orange <-> orth-range
    sorry but 'orange', 'slice' was just to delicious
    """
    axis = axis_wrapper(axis=axis, n_dim=len(shape))
    if mode == "slice":
        idx_with_ = [slice(None) for _ in range(len(shape) - len(axis))]

    elif mode == "orange":
        idx_with_ = np.ogrid[[range(s) for i, s in enumerate(shape) if i not in axis]]

    elif mode is None:
        return idx
    else:
        raise ValueError(f"Unknown mode {mode}")

    idx = np.array(idx)
    for i, ax in enumerate(axis):
        idx_with_.insert(ax, idx[..., i])

    return tuple(idx_with_)


def insert(a, val, idx, axis, mode="slice"):
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    a[idx] = val


def extract(a, idx, axis, mode="slice"):
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    return a[idx]


# Combine
def interleave(arrays, axis=0, out=None):
    """
    https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    """

    shape = list(np.asanyarray(arrays[0]).shape)
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape), "'axis' is out of bounds"
    if out is not None:
        out = out.reshape(shape[:axis + 1] + [len(arrays)] + shape[axis + 1:])
    shape[axis] = -1
    return np.stack(arrays, axis=axis + 1, out=out).reshape(shape)


# Functions
def digitize_group(x, bins, right=False):
    """
    https://stackoverflow.com/a/26888164/7570817
    Similar to scipy.stats.binned_statistic but just return the indices corresponding to each bin.
    Same signature as numpy.digitize()
    """
    idx_x = np.digitize(x=x, bins=bins, right=right)
    n, m = len(x), len(bins) + 1
    s = csr_matrix((np.arange(n), [idx_x, np.arange(n)]), shape=(m, n))
    return [group for group in np.split(s.data, s.indptr[1:-1])]


def sort_args(idx, *args):
    return [a[idx] for a in args]


def rolling_window(a, window):
    """https://stackoverflow.com/a/6811241/7570817"""
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_subarray(a, b):
    """
    Find b in a
    Return the index where the overlap begins.

    # a = np.array((2, 3, 4, 3, 5, 1))
    # b = np.array((3, 4, 3))

    # a = np.array((27, 3))
    # b = np.array((3,))
    # -> array([1])
    """
    a, b = np.atleast_1d(a, b)

    window = len(b)
    a_window = rolling_window(a=a, window=window)
    idx = np.nonzero((a_window == b).sum(axis=-1) == window)[0]
    return idx


def find_values(arr, values):
    res = np.zeros_like(arr, dtype=bool)
    for v in values:
        res[~res] = arr[~res] == v
    return res


def find_array_occurrences(a, o):
    assert a.ndim == o.ndim
    assert a.shape[-1] == o.shape[-1]

    if a.ndim == 2:
        b = a[:, np.newaxis, :] == o[np.newaxis, :, :]
        b = np.sum(b, axis=-1) == o.shape[-1]
        i = np.array(np.nonzero(b)).T

    else:
        raise ValueError

    return i


def get_element_overlap(arr1, arr2=None, verbose=0):
    """
    arr1 is a 2D array (n, matrix)
    arr2 is a 2D array (l, k)
    along the first dimension are different samples and the second dimension are different features of one sample

    return a 2D int array (n, l) where each element o, j shows how many of the elements
    of arr1[o] are also present in arr2[j], without regard of specific position in the arrays
    """
    if arr2 is None:
        arr2 = arr1

    overlap = np.zeros((len(arr1), len(arr2)), dtype=int)
    for i, arr_i in enumerate(arr1):
        if verbose > 0:
            print(f"{i} / {len(arr1)}")
        for j, arr_j in enumerate(arr2):
            for k in arr_i:
                if k in arr_j:
                    overlap[i, j] += 1

    return overlap


def create_constant_diagonal(n, m, v, k):
    diag = np.eye(N=n, M=m, k=k) * v[0]
    for i in range(1, len(v)):
        diag += np.eye(N=n, M=m, k=k + i) * v[i]
    return diag


def banded_matrix(v_list, k0):
    m = np.diag(v_list[0], k=k0)
    for i, v in enumerate(v_list[1:], start=1):
        m += np.diag(v, k=k0 + i)

    return m


def get_first_row_occurrence(bool_arr):
    """

    array([[ True,  True, False,  True,  True,  True],
           [False,  True, False,  True,  True, False],
           [False, False,  True, False, False, False],
           [False, False, False, False, False, False]])

    -> array([ 0,  1,  2, -1])
    """
    nz_i, nz_j = np.nonzero(bool_arr)
    u, idx = np.unique(nz_i, return_index=True)
    res = np.full(bool_arr.shape[0], fill_value=-1)
    res[u] = nz_j[idx]
    return res


def fill_interval_indices(interval_list, n):
    if isinstance(interval_list, np.ndarray):
        interval_list = interval_list.tolist()

    if np.size(interval_list) == 0:
        return np.array([[1, n]])

    if interval_list[0][0] != 0:
        interval_list.insert(0, [0, interval_list[0][0]])

    if interval_list[-1][1] != n:
        interval_list.insert(len(interval_list), [interval_list[-1][1], n])

    i = 1
    while i < len(interval_list):
        if interval_list[i - 1][1] != interval_list[i][0]:
            interval_list.insert(i, [interval_list[i - 1][1], interval_list[i][0]])
        i += 1

    return np.array(interval_list)


def get_interval_indices(bool_array, expand=False):
    """
    Get list of start and end indices, which indicate the sections of True values in the array.

    Array is converted to bool first

    [0, 0, 0, 0]  ->  [[]]
    [0, 0, 0, 1]  ->  [[3, 4]]
    [0, 1, 1, 0]  ->  [[1, 3]]
    [1, 0, 0, 0]  ->  [[0, 1]]
    [1, 0, 0, 1]  ->  [[0, 1], [3, 4]]
    [1, 1, 0, 1]  ->  [[0, 2], [3, 4]]
    [1, 1, 1, 1]  ->  [[0, 4]]

    """
    bool_array = bool_array.astype(bool)
    assert bool_array.ndim == 1
    interval_list = np.where(np.diff(bool_array) != 0)[0] + 1
    if bool_array[0]:
        interval_list = np.concatenate([[0], interval_list])
    if bool_array[-1]:
        interval_list = np.concatenate([interval_list, bool_array.shape])
    interval_list = interval_list.reshape(-1, 2)

    if expand:
        interval_list = [list(range(i0, i1)) for (i0, i1) in interval_list]

    return interval_list


def get_cropping_indices(pos, shape_small, shape_big, mode="lower_left"):
    """
    Adjust the boundaries to fit small array in a larger image.
    pos:  idx where the small image should be set in the bigger picture, option A
    mode:  mode how to position theta smaller array in the larger:
                  "center": pos describes the center of the small array inside the big array (shape_small must be odd)
                  "lower_left":
                  "upper_right":
    shape_small:  Size of the small image (=2*sm-1) in (number of pixels in each dimension)
    shape_big:  Size of the large image in (number of pixels in each dimension)
    :return:
    """

    shape_small, shape_big = args2arrays(shape_small, shape_big)

    if mode == "center":
        assert np.all(np.array(shape_small) % 2 == 1), shape_small
        shape_small2 = (np.array(shape_small) - 1) // 2

        ll_big = pos - shape_small2
        ur_big = pos + shape_small2 + 1

    elif mode == "lower_left":
        ll_big = pos
        ur_big = pos + shape_small

    elif mode == "upper_right":
        ll_big = pos - shape_small
        ur_big = pos

    else:
        raise ValueError(f"Invalid position mode {mode}")

    ll_small = np.where(ll_big < 0,
                        -ll_big,
                        0)
    ur_small = np.where(shape_big - ur_big < 0,
                        shape_small + (shape_big - ur_big),
                        shape_small)

    ll_big = np.where(ll_big < 0, 0, ll_big)
    ur_big = np.where(shape_big - ur_big < 0, shape_big, ur_big)

    return ll_big, ur_big, ll_small, ur_small


def add_small2big(idx, small, big, mode_crop="center", mode_add="add"):
    """
    Insert a small picture into the complete picture at the position 'idx'
    Assumption: all dimension of the small_img are odd, and idx indicates the center of the image,
    if this is not the case, there are zeros added at the end of each dimension to make the image shape odd
    """

    idx = flatten_without_last(idx)
    n_samples, n_dim = idx.shape
    ll_big, ur_big, ll_small, ur_small = get_cropping_indices(pos=idx, mode=mode_crop,
                                                              shape_small=small.shape[-n_dim:],
                                                              shape_big=big.shape)

    if small.ndim > n_dim:
        for ll_b, ur_b, ll_s, ur_s, s in zip(ll_big, ur_big, ll_small, ur_small, small):
            big[slicen(ll_b, ur_b)] += s[slicen(ll_s, ur_s)]
    else:
        for ll_b, ur_b, ll_s, ur_s in zip(ll_big, ur_big, ll_small, ur_small):

            try:
                if mode_add == "add":
                    big[slicen(ll_b, ur_b)] += small[slicen(ll_s, ur_s)]
                elif mode_add == "replace":
                    big[slicen(ll_b, ur_b)] = small[slicen(ll_s, ur_s)]
            except ValueError:
                print("idx", idx)
                print("big", ll_b, ur_b)  # TODO sometimes this fails, check
                print("big", big[slicen(ll_b, ur_b)].shape)
                print("small", ll_s, ur_s)
                print("small", small[slicen(ll_s, ur_s)].shape)


def get_exclusion_mask(a, exclude_values):
    bool_a = np.ones_like(a, dtype=bool)
    for v in exclude_values:
        bool_a[a == v] = False

    return bool_a


def matmul(a, b, axes_a=(-2, -1), axes_b=(-2, -1)):
    if axes_a == (-2, -1) and axes_b == (-2, -1):
        return a @ b

    if axes_a == (-3, -2) and axes_b == (-2, -1) and np.ndim(a) == np.ndim(b) + 1:
        return np.concatenate([(a[..., i] @ b)[..., np.newaxis]
                               for i in range(a.shape[-1])], axis=-1)
    else:
        raise NotImplementedError


def matsort(mat, order_j=None):
    """
    mat = np.where(np.random.random((500, 500)) > 0.01, 1, 0)
    ax[0].imshow(mat, cmap='gray_r')
    ax[1].imshow(matsort(mat)[0], cmap='gray_r')
    """

    def idx2interval(idx):
        idx = np.sort(idx)
        interval = np.zeros((len(idx) - 1, 2), dtype=int)
        interval[:, 0] = idx[:-1]
        interval[:, 1] = idx[1:]
        return interval

    n, m = mat.shape

    if order_j is None:
        order_j = np.argsort(np.sum(mat, axis=0))[::-1]

    order_i = np.argsort(mat[:, order_j[0]])[::-1]

    interval_idx = np.zeros(2, dtype=int)
    interval_idx[1] = n
    for i in range(0, m - 1):
        interval_idx = np.unique(np.hstack([interval_idx,
                                            get_interval_indices(mat[order_i, order_j[i]]).ravel()]))

        for j, il in enumerate(idx2interval(idx=interval_idx)):
            slice_j = slice(il[0], il[1])
            if j % 2 == 0:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])[::-1]]
            else:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])]

    return mat[order_i, :][:, order_j], order_i, order_j


def idx2boolmat(idx, n=100):
    """
    The last dimension of idx contains the indices. n-1 is the maximal possible index
    Returns matrix with shape np.shape(idx)[:-1] + (n,)

    """

    s = np.shape(idx)[:-1]

    mat = np.zeros(s + (n,), dtype=bool)

    for i, idx_i in enumerate(idx.reshape(-1, idx.shape[-1])):
        print(i, (np.unravel_index(i, shape=s)))
        mat[np.unravel_index(i, shape=s)][idx_i] = True
    return mat


def tile_offset(a, reps, offsets=None):
    s = shape_wrapper(a.shape)
    b = np.tile(a, reps)

    if offsets is not None:
        r = np.array(b.shape) // np.array(a.shape)
        if np.size(offsets) == 1:
            o = scalar2array(offsets, shape=len(s))
        else:
            o = offsets

        assert len(o) == len(s)
        offsets = [np.repeat(np.arange(rr), ss) * oo for ss, rr, oo in zip(s, r, o)]
        b += sum(np.meshgrid(*offsets, indexing='ij') + [0])  # noqa
    return b


def construct_array(shape, val, idx, init_mode="zeros", dtype=None,
                    axis=None, insert_mode=None):
    a = initialize_array(shape=shape, mode=init_mode, dtype=dtype)
    insert(a=a, val=val, idx=idx, axis=axis, mode=insert_mode)
    return a


def get_limits(x, axis=-1):
    axis = axis_wrapper(axis=axis, n_dim=x.ndim, invert=True)
    limits = np.stack([x.min(axis=axis), x.max(axis=axis)], axis=1)
    return limits


def get_closest(x, y):
    d = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=-1)
    i_x = np.argmin(d, axis=0)
    i_y = np.argmin(d, axis=1)
    return i_x, i_y


# Block lists
def block_view(a, shape, aslist=False, require_aligned_blocks=True):
    """
    Return a 2N-D view of the given N-D array, rearranged so each ND block (tile)
    of the original array is indexed by its block address using the first N
    indexes of the output array.
    Note: This function is nearly identical to ``skimage.util.view_as_blocks()``, except:
          - "imperfect" block shapes are permitted (via require_aligned_blocks=False)
          - only contiguous arrays are accepted.  (This function will NOT silently copy your array.)
            As a result, the return value is *always* a view of the input.
    Args:
        a: The ND array
        shape: The tile shape
        aslist: If True, return all blocks as a list of ND blocks
                instead of a 2D array indexed by ND block coordinate.
        require_aligned_blocks: If True, check to make sure no data is "left over"
                                in each row/column/etc. of the output view.
                                That is, the blockshape must divide evenly into the full array shape.
                                If False, "leftover" items that cannot be made into complete blocks
                                will be discarded from the output view.
    Here's a 2D example (this function also works for ND):
    # >>> a = np.arange(1,21).reshape(4,5)
    # >>> print(a)
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]
    # >>> view = blockwise_view(a, (2,2), require_aligned_blocks=False)
    # >>> print(view)
    [[[[ 1  2]
       [ 6  7]]
    <BLANKLINE>
      [[ 3  4]
       [ 8  9]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[11 12]
       [16 17]]
    <BLANKLINE>
      [[13 14]
       [18 19]]]]

    Inspired by the 2D example shown here: https://stackoverflow.com/a/8070716/162094
    """
    assert a.flags["C_CONTIGUOUS"], "This function relies on the memory layout of the array."
    shape = tuple(shape)
    outershape = tuple(np.array(a.shape) // shape)
    view_shape = outershape + shape

    if require_aligned_blocks:
        assert (np.mod(a.shape, shape) == 0
                ).all(), "blockshape {} must divide evenly into array shape {}".format(shape, a.shape)

    # inner strides: strides within each block (same as original array)
    intra_block_strides = a.strides

    # outer strides: strides from one block to another
    inter_block_strides = tuple(a.strides * np.array(shape))

    # This is where the magic happens.
    # Generate a view with our new strides (outer+inner).
    view = np.lib.stride_tricks.as_strided(a, shape=view_shape, strides=(inter_block_strides + intra_block_strides))

    if aslist:
        return list(map(view.__getitem__, np.rndindex(outershape)))
    return view


def expand_block_indices(idx_block, block_size, squeeze=True):
    """
    Expand the indices to get an index for each element

           block_size: |   1  |     2    |       3      |         4        |           5          |
    block_idx:          --------------------------------------------------------------------------|
              0        |   0  |   0,  1  |   0,  1,  2  |   0,  1,  2,  3  |   0,  1,  2,  3,  4  |
              1        |   1  |   2,  3  |   3,  4,  5  |   4,  5,  6,  7  |   5,  6,  7,  8,  9  |
              2        |   2  |   4,  5  |   6,  7,  8  |   8,  9, 10, 11  |  10, 11, 12, 13, 14  |
              3        |   3  |   6,  7  |   9, 10, 11  |  12, 13, 14, 15  |  15, 16, 17, 18, 19  |
              4        |   4  |   8,  9  |  12, 13, 14  |  16, 17, 18, 19  |  20, 21, 22, 23, 24  |
              5        |   5  |  10, 11  |  15, 16, 17  |  20, 21, 22, 23  |  25, 26, 27, 28, 29  |
              6        |   6  |  12, 13  |  18, 19, 20  |  24, 25, 26, 27  |  30, 31, 32, 33, 34  |
              7        |   7  |  14, 15  |  21, 22, 23  |  28, 29, 30, 31  |  35, 36, 37, 38, 39  |
              8        |   8  |  16, 17  |  24, 25, 26  |  32, 33, 34, 35  |  40, 41, 42, 43, 44  |
              9        |   9  |  18, 19  |  27, 28, 29  |  36, 37, 38, 39  |  45, 46, 47, 48, 49  |
              10       |  10  |  20, 21  |  30, 31, 32  |  40, 41, 42, 43  |  50, 51, 52, 53, 54  |
    """

    idx_block = np.atleast_1d(idx_block)
    if np.size(idx_block) == 1:
        return np.arange(block_size * int(idx_block), block_size * (int(idx_block) + 1))
    else:
        idx2 = np.array([expand_block_indices(i, block_size=block_size, squeeze=squeeze) for i in idx_block])
        if squeeze:
            return idx2.flatten()
        else:
            return idx2


def replace(arr, r_dict, copy=True, dtype=None):
    if copy:
        arr2 = arr.copy()
        if dtype is not None:
            arr2 = arr2.astype(dtype=dtype)
        for key in r_dict:
            arr2[arr == key] = r_dict[key]
        return arr2
    else:
        for key in r_dict:
            arr[arr == key] = r_dict[key]


def block_shuffle(arr, block_size, inside=False):
    """
    Shuffle the array along the first dimension,
    if block_size > 1, keep as many elements together and shuffle the n // block_size blocks
    """

    if isinstance(arr, int):
        n = arr
        arr = np.arange(n)
    else:
        n = arr.shape[0]

    if block_size == 1:
        np.random.shuffle(arr)
        return arr

    assert block_size > 0
    assert isinstance(block_size, int)
    assert n % block_size == 0
    n_blocks = n // block_size

    if inside:
        idx = np.arange(n)
        for i in range(0, n, block_size):
            np.random.shuffle(idx[i:i + block_size])
        return arr[idx]

    else:
        idx_block = np.arange(n_blocks)
        np.random.shuffle(idx_block)
        idx_ele = expand_block_indices(idx_block=idx_block, block_size=block_size, squeeze=True)
        return arr[idx_ele]


def replace_tail_roll(a, b):
    """
    Replace the last elements of the array with the new array and roll the new ones to the start
    So that a repeated call of this function cycles through the array
    replace_tail_roll(a=[ 1,  2,  3,  4,  5,  6, 7, 8], b=[77, 88])   -->   [77, 88,  1,  2,  3,  4,  5,  6]
    replace_tail_roll(a=[77, 88,  1,  2,  3,  4, 5, 6], b=[55, 66])   -->   [55, 66, 77, 88,  1,  2,  3,  4]
    replace_tail_roll(a=[55, 66, 77, 88,  1,  2, 3, 4], b=[33, 44])   -->   [33, 44, 55, 66, 77, 88,  1,  2]
    replace_tail_roll(a=[33, 44, 55, 66, 77, 88, 1, 2], b=[11, 22])   -->   [11, 22, 33, 44, 55, 66, 77, 88]
    """
    n_a, n_b = np.shape(a)[0], np.shape(b)[0]
    assert n_a > n_b

    a[-n_b:] = b
    return np.roll(a, n_b, axis=0)


def replace_tail_roll_list(arr_list, arr_new_list):
    assert len(arr_list) == len(arr_new_list)
    return (replace_tail_roll(a=arr, b=arr_new) for (arr, arr_new) in zip(arr_list, arr_new_list))


def find_block_shuffled_order(a, b, block_size, threshold, verbose=1):
    n = len(a)
    m = len(b)
    assert n == m
    assert n % block_size == 0

    nn = n // block_size
    idx = np.empty(nn)

    for i in range(nn):
        for j in range(nn):
            d = (a[i * block_size:(i + 1) * block_size] -
                 b[j * block_size:(j + 1) * block_size])

            d = np.abs(d).max()
            if d < threshold:
                idx[i] = j
                if verbose > 0:
                    print(i, j, d)

    return idx


def get_stats(x, axis=None, return_array=False):
    stats = {"mean": np.mean(x, axis=axis),
             "std": np.std(x, axis=axis),
             "median": np.median(x, axis=axis),
             "min": np.min(x, axis=axis),
             "max": np.max(x, axis=axis)}

    if return_array:
        return np.array([stats["mean"], stats["std"], stats["median"], stats["min"], stats["max"]])

    return stats


def verbose_reject_x(title, x, b):
    if b.size == 0:
        mean = 0
    else:
        mean = b.mean()
    print(f"{title}: {b.sum()}/{b.size} ~ {np.round(mean * 100, 3)}%")
    return x[b]


def add_safety_limits(limits, factor):
    limits = np.atleast_1d(limits)
    diff = np.diff(limits, axis=-1)[..., 0]
    return np.array([limits[..., 0] - factor * diff,
                     limits[..., 1] + factor * diff]).T


def get_points_inbetween(x, extrapolate=False):
    assert x.ndim == 1

    delta = x[1:] - x[:-1]
    x_new = np.zeros(np.size(x) + 1)
    x_new[1:-1] = x[:-1] + delta / 2
    if extrapolate:
        x_new[0] = x_new[1] - delta[0]
        x_new[-1] = x_new[-2] + delta[-1]
        return x_new
    else:
        return x_new[1:-1]


def find_consecutives(x, n):
    if n == 1:
        return np.arange(len(x))
    assert n > 1
    return np.nonzero(np.convolve(np.abs(np.diff(x)), v=np.ones(n - 1), mode="valid") == 0)[0]


def find_largest_consecutives(x):
    i2 = np.nonzero(np.convolve(np.abs(np.diff(x)), v=np.ones(2 - 1), mode="valid") == 0)[0]
    i2 -= np.arange(len(i2))
    _, c2 = np.unique(i2, return_counts=True)
    if c2.size == 0:
        n = 1
    else:
        n = c2.max() + 1

    return n, find_consecutives(x, n=n)


def squeeze_all(*args):
    return [np.squeeze(a) for a in args]


def round2(x,
           decimals=None):
    # noinspection PyProtectedMember
    try:
        return np.round(x, decimals=decimals)

    except (TypeError, np.core._exceptions.UFuncTypeError):
        return np.array(x)


def clip2(x, clip, mode, axis=-1):
    if mode:
        if mode == "value":
            return np.clip(x, a_min=-clip, a_max=+clip)

        elif mode == "norm":
            x = x.copy()
            n = np.linalg.norm(x, axis=axis, keepdims=True)
            b = n > clip
            x[b] = x[b] * (clip / n[b])
            return x

        elif mode == "norm-force":
            n = np.linalg.norm(x, axis=axis, keepdims=True)
            return x * (clip / n)

        else:
            raise ValueError(f"Unknown mode: '{mode}'")
    return x


def load_dict(file: str) -> dict:
    d = np.load(file, allow_pickle=True)
    try:
        d = d.item()
    except AttributeError:
        d = d["arr"]
        d = d.item()

    assert isinstance(d, dict)
    return d


def round_dict(d, decimals=None):
    for key in d.keys():
        value = d[key]
        if isinstance(value, dict):
            d[key] = round_dict(d=value, decimals=decimals)
        else:
            d[key] = round2(x=value, decimals=decimals)

    return d


def arangen(start=None, end=None, step=None):
    """n dimensional slice"""
    n = max_size(start, end, step)
    start, end, step = scalar2array(start, end, step, shape=n)
    return [np.arange(start[i], end[i], step[i]) for i in range(n)]


# Slice Ellipsis Range
# Slice and range
def slicen(start=None, end=None, step=None):
    """n dimensional slice"""
    n = max_size(start, end, step)
    start, end, step = scalar2array(start, end, step, shape=n)
    return tuple(map(slice, start, end, step))


def test_slicen():
    sl = slicen([1], [2])
    assert sl == (slice(1, 2),)


def range2slice(r):
    return slice(r.start, r.stop, r.step)


def slice2range(s):
    return range(0 if s.start is None else s.start,
                 s.stop,
                 1 if s.step is None else s.step)


def __slice_or_range2tuple(sor, type2):
    if type2 == "slice":
        default1, default2, default3 = None, None, None
    elif type2 == "range":
        default1, default2, default3 = 0, 1, 1
    else:
        raise ValueError(f"Unknown {type2}")

    if isinstance(sor, (slice, range)):
        return sor.start, sor.stop, sor.step

    elif isinstance(sor, int):
        return default1, sor, default3

    elif sor is None:
        return default1, default2, default3

    elif isinstance(sor, tuple):
        if len(sor) == 1:
            return default1, sor[0], default3
        elif len(sor) == 2:
            return sor[0], sor[1], default3
        elif len(sor) == 3:
            return sor
        else:
            raise ValueError("tuple must be have length={1, 2, 3}")
    else:
        raise TypeError("r must be {slice, range, int, tuple}")


def slice2tuple(s):
    return __slice_or_range2tuple(s, "slice")


def range2tuple(r):
    return __slice_or_range2tuple(r, "range")


def slice_add(a, b):
    a = slice2tuple(a)
    b = slice2tuple(b)
    return slice(a[0] + b[0], a[1] + b[1], max(a[2], b[2]))


def range_add(a, b):
    a = range2tuple(a)
    b = range2tuple(b)
    return range(a[0] + b[0], a[1] + b[1], max(a[2], b[2]))

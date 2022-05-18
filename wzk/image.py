import zlib
import numpy as np
from scipy.signal import convolve2d, convolve
from skimage.io import imread, imsave  # noqa
from skimage import measure
from skimage.morphology import flood_fill

from wzk.dicts_lists_tuples import tuple_extract
from wzk.numpy2 import (align_shapes, get_cropping_indices, flatten_without_last, initialize_array, limits2cell_size,
                        grid_x2i, grid_i2x)
from wzk.trajectory import get_substeps_adjusted
from wzk.geometry import discretize_triangle_mesh, ConvexHull
from wzk.math2 import make_even_odd


def imread_bw(file, threshold):

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    img = imread(file)
    img = rgb2gray(img)
    obstacle_image = img < threshold  # 60
    return obstacle_image


def combine_shape_n_dim(shape, n_dim=None):
    if np.size(shape) == 1:
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)
        shape *= n_dim

    else:
        shape = tuple(shape)

    return shape


def image_array_shape(shape, n_samples=None, n_dim=None, n_channels=None):
    """
    Helper to set the shape for an image array.
    n_samples=100,  shape=64,          n_dim=2,    n_channels=None  ->  (100, 64, 64)
    n_samples=100,  shape=64,          n_dim=3,    n_channels=2     ->  (100, 64, 64, 64, 2)
    n_samples=None, n_voxel=(10, 11, 12), n_dim=None, n_channels=None  ->  (10, 11, 12)
    """

    shape = combine_shape_n_dim(shape=shape, n_dim=n_dim)

    if n_samples is not None:
        shape = (n_samples,) + shape
    if n_channels is not None:
        shape = shape + (n_channels,)

    return shape


def initialize_image_array(shape: tuple, n_dim: int = None, n_samples: int = None, n_channels: int = None,
                           dtype=None, initialization: str = 'zeros') -> np.ndarray:
    shape = image_array_shape(shape=shape, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels)
    return initialize_array(shape=shape, mode=initialization, dtype=dtype)


def reshape_img(img, n_dim=2, sample_dim=True, channel_dim=True,
                n_samples=None,  # either
                n_channels=None):  # or. infer the other

    shape = img.shape[1]  # Can go wrong for 1D

    if n_samples is None and n_channels is None:
        n_channels = 1
        n_samples = 1

    elif n_samples is None:
        n_samples = img.size // (shape ** n_dim) // n_channels
    elif n_channels is None:
        n_channels = img.size // (shape ** n_dim) // n_samples

    if n_samples == 1 and not sample_dim:
        n_samples = None

    if n_channels == 1 and not channel_dim:
        n_channels = None

    return img.reshape(image_array_shape(shape=shape, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels))


def concatenate_images(*imgs, axis=-1):
    """
    could add parameters which allow stacking horizontal, vertical, channel_wise, and sample wise
    """
    n_dim = np.max([i.ndim for i in imgs])
    if axis == +1:
        imgs = [i if i.ndim == n_dim else i[np.newaxis, ...] for i in imgs]
    elif axis == -1:
        imgs = [i if i.ndim == n_dim else i[..., np.newaxis] for i in imgs]
    else:
        raise NotImplementedError

    return np.concatenate(imgs, axis=axis)


def add_padding(img, padding, value):
    shape = np.array(img.shape)
    padding_arr = np.zeros((img.ndim, 2), dtype=int)

    if isinstance(padding, str):
        padding_arr[:, 0] = make_even_odd(shape, mode=padding, rounding=+1) - shape

    elif isinstance(padding, int):
        padding_arr[:] = padding

    elif isinstance(padding, (np.ndarray, tuple, list)):
        if padding.ndim == 2:
            padding_arr[:] = padding
        elif len(padding) == len(shape):
            padding_arr[:] = np.expand_dims(padding, axis=1)

    else:
        raise ValueError

    new_shape = shape + padding_arr.sum(axis=1)
    new_img = np.full(shape=new_shape, fill_value=value, dtype=img.dtype)
    new_img[tuple(map(slice, padding_arr[:, 0], padding_arr[:, 0]+shape))] = img
    return new_img


def block_collage(*, img_arr, inner_border=None, outer_border=None, fill_boarder=0, dtype=float):

    assert img_arr.ndim == 4
    n_rows, n_cols, n_x, n_y = img_arr.shape

    bv_i, bh_i = tuple_extract(inner_border, default=(0, 0), mode='repeat')
    bv_o, bh_o = tuple_extract(outer_border, default=(0, 0), mode='repeat')

    img = np.full(shape=(n_x * n_rows + bv_i * (n_rows - 1) + 2*bv_o,
                         n_y * n_cols + bh_i * (n_cols - 1) + 2*bh_o), fill_value=fill_boarder, dtype=dtype)

    for r in range(n_rows):
        for c in range(n_cols):
            img[bv_o + r * (n_y + bv_i):bv_o + (r + 1) * (n_y + bv_i) - bv_i,
                bh_o + c * (n_x + bh_i):bh_o + (c + 1) * (n_x + bh_i) - bh_i] = img_arr[r, c]

    return img


def reduce_shape(img, shape, n_dim, n_channels, kernel, pooling_type='average', n_samples=None,
                 sample_dim=False, channel_dim=False):
    # TODO use scipy method, + clean up
    # https://stackoverflow.com/questions/59988649/indexing-numpy-array-with-list-of-slices
    # shape_new = 3
    # for o in range(shape_new):
    #     for j in range(shape_new):
    #         for k in range(shape_new):
    #             print(o,j,k)
    #
    # import itertools

    # for ijk in itertools.product(range(shape_new), repeat=n_dim):

    # if you keep using this method rewrite the difference between 2D and 3d cleaner with map
    # assert shape % kernel == 0
    if pooling_type == 'average':
        pool = np.mean
        dtype = float
    else:  # == 'max'
        pool = np.max
        dtype = bool

    shape_new = np.array(shape) // np.array(kernel)
    shape_new = shape_new.astype(int)
    img_new = initialize_image_array(shape=shape_new, n_dim=n_dim, n_channels=n_channels, n_samples=n_samples,
                                     dtype=dtype)
    img = reshape_img(img=img, n_dim=n_dim, n_channels=n_channels, channel_dim=True, n_samples=n_samples,
                      sample_dim=True)
    img_new = reshape_img(img=img_new, n_dim=n_dim, n_channels=n_channels, channel_dim=True, n_samples=n_samples,
                          sample_dim=True)

    if n_dim == 2:
        for i in range(shape_new[0]):
            for j in range(shape_new[1]):
                img_new[:, i, j, :] = pool(img[:,
                                           kernel * i: kernel * (i + 1),
                                           kernel * j: kernel * (j + 1), :], axis=(1, 2))
    else:  # n_dim == 3
        for i in range(shape_new[0]):
            for j in range(shape_new[1]):
                for k in range(shape_new[2]):
                    img_new[:, i, j, k, :] = pool(img[:,
                                                  kernel * i: kernel * (i + 1),
                                                  kernel * j: kernel * (j + 1),
                                                  kernel * k: kernel * (k + 1), :], axis=(1, 2, 3))

    return reshape_img(img=img_new, n_dim=n_dim, n_channels=n_channels, channel_dim=channel_dim, n_samples=n_samples,
                       sample_dim=sample_dim)


def pooling(mat, kernel, method='max', pad=False):
    """
    Non-overlapping pooling on 2D or 3D Measurements.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel shape in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has shape
           n//f, n being <mat> shape, f being kernel shape.
           if pad output has shape ceil(n/f).

    Return <result>: pooled matrix.
    """

    print("# TODO write general function with reshaping -> much faster")
    m, n = mat.shape[:2]
    ky, kx = kernel

    def _ceil(x, y):
        return int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[:ny * ky, :nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result


def get_outer_edge(img):
    n_dim = np.ndim(img)
    kernel = np.ones((3,)*n_dim)
    edge_img = convolve(img, kernel, mode='same', method='direct') > 0
    return np.logical_xor(edge_img, img)


def tile_2d(*, pattern, v_in_row, v_to_next_row, offset=(0, 0),
            shape):
    """

    Examples:

    # Point A
    pattern = np.ones((1, 1))
    shape = (11, 11)
    offset = (0, 0)
    v_in_row = 4
    v_to_next_row = (1, 2)

    # [[1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]]

    # Point B
    pattern = np.ones((1, 1))
    shape = (11, 11)
    offset = (0, 0)
    v_in_row = 5
    v_to_next_row = (1, 2)

    # [[1 0 0 0 0 1 0 0 0 0 1]
    #  [0 0 1 0 0 0 0 1 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 1 0]
    #  [0 1 0 0 0 0 1 0 0 0 0]
    #  [0 0 0 1 0 0 0 0 1 0 0]
    #  [1 0 0 0 0 1 0 0 0 0 1]
    #  [0 0 1 0 0 0 0 1 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 1 0]
    #  [0 1 0 0 0 0 1 0 0 0 0]
    #  [0 0 0 1 0 0 0 0 1 0 0]
    #  [1 0 0 0 0 1 0 0 0 0 1]]

    # Dumbbell
    pattern = np.ones((2, 2))
    pattern[0, 1] = 0
    pattern[1, 0] = 0
    shape = (16, 16)
    offset = (0, 0)
    v_in_row = 7
    v_to_next_row = (1, 3)

    # [[1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0]
    #  [0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1]
    #  [0 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0]
    #  [0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1]
    #  [0 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]]

    # Triangle
    pattern = np.ones((2, 2))
    pattern[0, 1] = 0
    shape = (10, 10)
    offset = (0, 0)
    v_in_row = 8
    v_to_next_row = (1, 3)

    # [[1 0 0 0 0 1 1 0 1 0]
    #  [1 1 0 1 0 0 0 0 1 1]
    #  [0 0 0 1 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 1 0 1]
    #  [0 1 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 1 0]
    #  [0 0 1 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 1 0 1 0]
    #  [1 1 0 1 0 0 0 0 1 1]]

    # Cross
    pattern = np.zeros((3, 3))
    pattern[1, :] = 1
    pattern[:, 1] = 1
    shape = (13, 13)
    offset = (1, 1)
    v_in_row = 4
    v_to_next_row = (3, 2)

    # [[1 1 0 1 1 1 0 1 1 1 0 1 1]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [0 1 1 1 0 1 1 1 0 1 1 1 0]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [1 1 0 1 1 1 0 1 1 1 0 1 1]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [0 1 1 1 0 1 1 1 0 1 1 1 0]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [1 1 0 1 1 1 0 1 1 1 0 1 1]]
    """
    nodes = np.zeros((shape[0]+v_to_next_row[0], shape[1]+v_in_row))

    for ii, i in enumerate(range(0, nodes.shape[0], v_to_next_row[0])):
        nodes[i, range((ii*v_to_next_row[1]) % v_in_row, nodes.shape[1], v_in_row)] = 1

    img = convolve2d(nodes, pattern, mode='full')

    ll = (v_to_next_row[0] + offset[0],
          v_to_next_row[1] + offset[1])

    return img[ll[0]:ll[0]+shape[0],
               ll[1]:ll[1]+shape[1]]


def check_overlap(a, b, return_arr=False):
    """
    Boolean indicating if the two arrays have an overlap.
    Sum over the axis that do not match.
    """

    a, b = (a, b) if a.n_dim > b.n_dim else (b, a)

    aligned_shape = align_shapes(a=a, b=b)
    summation_axis = np.arange(a.n_dim)[aligned_shape == -1]
    a = a.sum(axis=summation_axis)

    if return_arr:
        return np.logical_and(a, b)
    else:
        return np.logical_and(a, b).any()


def safe_add_small2big(idx, small_img, big_img, mode='center'):
    """
    Insert a small picture into the complete picture at the position 'idx'
    Assumption: all dimension of the small_img are odd, and idx indicates the center of the image,
    if this is not the case, there are zeros added at the end of each dimension to make the image shape odd
    Not both 'big_img' and 'shape' can be None. One is needed to calculate the other.
    """

    idx = flatten_without_last(idx)
    n_samples, n_dim = idx.shape
    ll_big, ur_big, ll_small, ur_small = get_cropping_indices(pos=idx, mode=mode,
                                                              shape_small=small_img.shape[-n_dim:],
                                                              shape_big=big_img.shape)

    if small_img.ndim > n_dim:
        for ll_b, ur_b, ll_s, ur_s, img_s in zip(ll_big, ur_big, ll_small, ur_small, small_img):
            big_img[tuple(map(slice, ll_b, ur_b))] += img_s[tuple(map(slice, ll_s, ur_s))]
    else:
        for ll_b, ur_b, ll_s, ur_s in zip(ll_big, ur_big, ll_small, ur_small):
            big_img[tuple(map(slice, ll_b, ur_b))] += small_img[tuple(map(slice, ll_s, ur_s))]


def sample_from_img(img, range_, n, replace=False):
    b = np.logical_and(range_[0] < img, img < range_[1])
    idx = np.array(np.nonzero(b)).T
    i = np.random.choice(a=np.arange(len(idx)), size=n, replace=replace)
    return idx[i]


# Image Compression <-> Decompression
def img2compressed(img, n_dim=-1, level=9):
    """
    Compress the given image with the zlib routine to a binary string.
    Level of compression can be adjusted. A timing with respect to different compression levels for decompression showed
    no difference, so the highest level is default, this corresponds to the largest compression.
    For compression, it is slightly slower but this happens just once and not during keras training, so the smaller
    needed memory was favoured.

    Alternative:
    <-> use numpy sparse for the world images, especially in 3d  -> zlib is more effective and more general
    """

    shape = img.shape[:-n_dim]
    if n_dim == -1 or shape == ():
        return zlib.compress(img.tobytes(), level=level)

    else:
        img_cmp = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            img_cmp[idx] = zlib.compress(img[idx, ...].tobytes(), level=level)
        return img_cmp


def compressed2img(img_cmp, shape, n_dim=None, n_channels=None, dtype=None):
    """
    Decompress the binary string back to an image of given shape
    """

    shape2 = image_array_shape(shape=shape, n_dim=n_dim, n_channels=n_channels)
    if np.shape(img_cmp):
        n_samples = np.size(img_cmp)
        img_arr = initialize_image_array(shape=shape, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels,
                                         dtype=dtype)
        for i in range(n_samples):
            img_arr[i, ...] = np.frombuffer(zlib.decompress(img_cmp[i]), dtype=dtype).reshape(shape2)
        return img_arr

    else:
        return np.frombuffer(zlib.decompress(img_cmp), dtype=dtype).reshape(shape2)


def bool_img2surf(img, limits):
    lower_left = limits[:, 0]
    voxel_size = limits2cell_size(shape=img.shape, limits=limits)
    if img.sum() == 0:
        verts = np.zeros((3, 3))
        faces = np.zeros((1, 3), dtype=int)
        faces[:] = np.arange(3)

    else:
        verts, faces, _, _ = measure.marching_cubes(img, level=0, spacing=(voxel_size,) * img.ndim)
        verts = verts + lower_left

    return verts, faces


def mesh2bool_img(p, shape, limits, f=None):

    img = np.zeros(shape, dtype=int)

    voxel_size = limits2cell_size(shape=shape, limits=limits)
    if img.ndim == 2:
        p2 = np.concatenate((p, p[:1]), axis=0)
        p2 = get_substeps_adjusted(x=p2, n=2 * len(p) * max(shape))
        i2 = grid_x2i(x=p2, limits=limits, shape=shape)
        img[i2[:, 0], i2[:, 1]] = 1

    elif img.ndim == 3:
        if f is None:
            ch = ConvexHull(p)
            p = ch.points
            f = ch.simplices  # noqa
        p2 = discretize_triangle_mesh(p=p, f=f, voxel_size=voxel_size)
        i2 = grid_x2i(x=p2, limits=limits, shape=shape)
        img[i2[:, 0], i2[:, 1], i2[:, 2]] = 1

    else:
        raise ValueError

    img = flood_fill(img, seed_point=(0,)*img.ndim, connectivity=1, new_value=2)
    img = np.array(img != 2)

    return img


def test_mesh2bool_img():

    p = np.random.random((10, 2))
    limits = np.array([[0, 1],
                       [0, 1]])
    img = mesh2bool_img(p=p, shape=(64, 64), limits=limits)

    from wzk.mpl import new_fig, imshow

    fig, ax = new_fig()
    ax.plot(*p.T, ls='', marker='o')
    imshow(ax=ax, img=img, mask=~img, limits=limits)


if __name__ == '__main__':
    test_mesh2bool_img()

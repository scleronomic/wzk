import numpy as np
from scipy.signal import convolve
from skimage import measure
from skimage.morphology import flood_fill

from wzk.geometry import discretize_triangle_mesh, ConvexHull, rectangle, cube
from wzk.numpy2 import (limits2cell_size, grid_x2i, grid_i2x, scalar2array, flatten_without_last, safe_add_small2big)
from wzk.printing import print_progress
from wzk.trajectory import get_substeps_adjusted


__eps = 1e-9


def __closest_boundary_rel_idx(half_side):
    idx_rel = np.arange(start=-half_side + 1, stop=half_side + 2, step=1)
    idx_rel[half_side:] = idx_rel[half_side - 1:-1]
    return idx_rel


def closest_grid_boundary(*, x, half_side, limits, shape, idx=None):
    """
    Given the coordinates of a point 'x', the radius of a sphere and dimensions of a grid.
    Calculate for that coordinate the closest boundary to the point 'x' for a cell defined by the relative idx j.
    4 cases:

           ----- ----- ----- ----- -----
          |     |     |     |     |     |
          |     |     |     |     |     |
          |    o|    o|  j  |o    |o    |
           ----- ----- ----- ----- -----
          |     |ooooo|ooooo|ooooo|     |
          |     |ooooo|ooooo|ooooo|     |
          |    o|ooooo|ooooo|ooooo|o    |
           ----- ----- ----- ----- -----
          |     |ooooo|     |ooooo|     |
          |    j|ooooo|  x  |ooooo|j    |
          |     |ooooo|     |ooooo|     |
           ----- ----- ----- ----- -----
          |    o|ooooo|ooooo|ooooo|o    |
          |     |ooooo|ooooo|ooooo|     |
          |     |ooooo|ooooo|ooooo|     |
           ----- ----- ----- ----- -----
          |    o|    o|  j  |o    |o    |
          |     |     |     |     |     |
          |     |     |     |     |     |
           ----- ----- ----- ----- -----


    The four cases at the intersection of cross with the grid
        __j__
        j_x_j
        __j__
    """

    if idx is None:
        idx = grid_x2i(x=x, limits=limits, shape=shape)

    idx = flatten_without_last(x=idx)
    rel_idx = __closest_boundary_rel_idx(half_side=half_side)

    idx = idx[:, np.newaxis, :] + rel_idx[np.newaxis, :, np.newaxis]
    x_closest = grid_i2x(i=idx, limits=limits, shape=shape, mode='b')

    x_closest[:, half_side, :] = x

    return x_closest


def __get_centers(voxel_size, n_dim):
    limits0 = np.zeros((n_dim, 2))
    limits0[:, 0] = __eps
    limits0[:, 1] = voxel_size - __eps

    if n_dim == 2:
        v, e = rectangle(limits=limits0)

    elif n_dim == 3:
        v, e, f = cube(limits=limits0)

    else:
        raise ValueError

    return v


# Helper
def __compare_dist_against_radius(x_a, x_b, r):
    dist = x_a - x_b
    dist = (dist**2).sum(axis=-1)
    return dist < r ** 2 - 5 * __eps


def get_max_occupied_cells(length, voxel_size):
    return np.asarray(np.ceil(length / voxel_size), dtype=int) + 1


def get_outer_edge(img):
    n_dim = np.ndim(img)
    kernel = np.ones((3,)*n_dim)
    edge_img = convolve(img, kernel, mode='same', method='direct') > 0
    return np.logical_xor(edge_img, img)


def get_sphere_stencil(r: float, voxel_size: float, n_dim: int = 2) -> (np.ndarray, np.ndarray):
    half_side = get_max_occupied_cells(length=r, voxel_size=voxel_size) - 1

    if half_side == 0:
        return np.ones((1,) * n_dim, dtype=bool), np.zeros((1,) * n_dim, dtype=bool)

    x_center = __get_centers(voxel_size=voxel_size, n_dim=n_dim)

    limits = np.zeros((n_dim, 2))
    limits[:, 1] = voxel_size
    shape = np.ones(n_dim)
    x_closest = closest_grid_boundary(x=x_center, half_side=half_side, limits=limits, shape=shape)

    img = np.zeros((len(x_center),) + (2 * half_side + 1,) * n_dim, dtype=bool)
    for i in range(len(x_center)):
        x_closest_i = np.array(np.meshgrid(*x_closest[i].T)).T
        img[i, ...] = __compare_dist_against_radius(x_a=x_center[i], x_b=x_closest_i, r=r)

    inner = img.sum(axis=0) == img.shape[0]
    outer = get_outer_edge(inner)

    return inner, outer


def get_stencil_list(r, n,
                     voxel_size, n_dim):
    if np.size(r) > 1:
        assert np.size(r) == n
        r_unique, stencil_idx = np.unique(r, return_inverse=True)
        stencil_list = [get_sphere_stencil(r=r_, voxel_size=voxel_size, n_dim=n_dim) for r_ in r_unique]
    else:
        stencil_list = [get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)]
        stencil_idx = np.zeros(n, dtype=int)
        r_unique = np.array([r])

    return r_unique, stencil_list, stencil_idx


def create_stencil_dict(voxel_size, n_dim):
    stencil_dict = dict()
    n = int(5*(1//voxel_size))
    for i, r in enumerate(np.linspace(voxel_size/10, 2, num=n)):
        print_progress(i=i, n=n, prefix='create_stencil_dict')
        l = int((r // voxel_size) * 2 + 3)
        if l not in stencil_dict.keys():
            stencil = np.logical_or(*get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim))
            assert l == stencil.shape[0]
            stencil_dict[l] = stencil
    return stencil_dict


def bimg2surf(img, limits):
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


def mesh2bimg(p, shape, limits, f=None):
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

    img = flood_fill(img, seed_point=(0,) * img.ndim, connectivity=1, new_value=2)
    img = np.array(img != 2)

    return img


def spheres2bimg(x, r, shape, limits,
                 stencil_dict=None):
    x = np.atleast_2d(x)
    n, n_dim = x.shape
    assert len(shape) == n_dim

    r = scalar2array(r, shape=n)
    img = np.zeros(shape, dtype=bool)
    voxel_size = limits2cell_size(shape=shape, limits=limits)

    for i in range(n):
        j = grid_x2i(x[i], limits=limits, shape=shape)
        l = int((r[i] // voxel_size) * 2 + 3)
        if stencil_dict:
            stencil = stencil_dict[l]
        else:
            stencil = np.logical_or(*get_sphere_stencil(r=r[i], voxel_size=voxel_size, n_dim=n_dim))
        # img[tuple(map(slice, j - (l - 1) // 2, j + (l - 1) // 2 + 1))] += stencil
        safe_add_small2big(idx=j, small=stencil, big=img)

    return img


def sample_bimg_i(img, n, replace=True):
    i = np.array(np.nonzero(img)).T
    j = np.random.choice(a=np.arange(len(i)), size=n, replace=replace)
    return i[j]


def sample_bimg_x(img, limits, n, replace=True):

    i = sample_bimg_i(img=img, n=n, replace=replace)
    x = grid_i2x(i=i, limits=limits, shape=img.shape)
    voxel_size2 = limits2cell_size(shape=img.shape, limits=limits) / 2
    cell_noise = np.random.uniform(low=-voxel_size2, high=+voxel_size2, size=(n, img.ndim))

    x += cell_noise
    return x


def sample_spheres_bimg_x(x, r, shape, limits, n,):
    img = spheres2bimg(x=x, r=r, shape=shape, limits=limits)
    x = sample_bimg_x(img=img, limits=limits, n=n, replace=True)
    return x


def test_get_sphere_stencil():
    from wzk import new_fig
    r = 10
    voxel_size = 0.3
    n_dim = 2

    inner, outer = get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)
    fig, ax = new_fig()
    ax.imshow(inner)

    n_dim = 3
    inner, outer = get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)
    fig, ax = new_fig()
    ax.imshow(inner.sum(axis=-1), cmap='gray_r')


def test_mesh2bimg():
    p = np.random.random((10, 2))
    limits = np.array([[0, 1],
                       [0, 1]])
    img = mesh2bimg(p=p, shape=(64, 64), limits=limits)

    from wzk.mpl import new_fig, imshow

    fig, ax = new_fig()
    ax.plot(*p.T, ls='', marker='o')
    imshow(ax=ax, img=img, mask=~img, limits=limits)


def test_spheres2bimg():
    n = 10
    shape = (256, 256, 256)
    
    limits = np.array([[-1, 2],
                       [-1, 2],
                       [-1, 2]])
    x = np.random.random((n, 3))
    r = np.random.uniform(low=0.1, high=0.2, size=n)
    img = spheres2bimg(x=x, r=r, shape=shape, limits=limits)
    
    from wzk.pv.plotting import plot_bimg, pv
    p = pv.Plotter()
    plot_bimg(p=p, img=img, limits=limits)
    p.show()
    
    
if __name__ == '__main__':
    test_spheres2bimg()
    # test_get_sphere_stencil()
    # test_mesh2bimg()

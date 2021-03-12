import numpy as np

from scipy.spatial import ConvexHull


def projection_line_point(x0, x1, x2):
    """
    http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    Projection from x0 to the line defined by {x1 + a*x2}
    """
    x21 = x2-x1
    t = np.sum((x1 - x0)*(x21), axis=-1) / np.linalg.norm(x21, axis=-1)**2
    x0_p = x1 - t * x21
    return x0_p


def distance_line_point(x0, x1, x2):
    """
    http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    Distance between x0 and the line defined by {x1 + a*x2}
    """
    return np.linalg.norm(np.cross(x0-x1, x0-x2), axis=-1) / np.linalg.norm(x2-x1, axis=-1)


def circle_circle_intersection(xy0, r0, xy1, r1):
    """
    https://stackoverflow.com/a/55817881/7570817
    https://mathworld.wolfram.com/Circle-CircleIntersection.html

    circle 1: (x0, y0), radius r0
    circle 2: (x1, y1), radius r1
    """

    d = np.linalg.norm(xy1 - xy0)

    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(r0 ** 2 - a ** 2)
        d01 = (xy1 - xy0) / d

        xy2 = xy0 + a * d01[::+1] * [+1, +1]
        xy3 = xy2 + h * d01[::-1] * [+1, -1]
        xy4 = xy2 + h * d01[::-1] * [-1, +1]

        return xy3, xy4


def ray_sphere_intersection(rays, spheres):
    """
    :param rays: n_rays x 2 x 3    (axis=1: origin, target)
    :param spheres: n_spheres x 4  (axis=1: x, y, z, r)
    :return: n_rays x n_spheres (boolean array) with res[o, j] = True if ray o intersects with sphere j
    Formula from: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    """

    o = rays[:, 0]
    u = np.diff(rays, axis=1)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    c = spheres[:, :3]
    r = spheres[:, 3:].T
    co = (o[:, np.newaxis, :] - c[np.newaxis, :, :])
    res = (u * co).sum(axis=-1)**2 - (co**2).sum(axis=-1) + r**2
    return res >= 0


def ray_sphere_intersection_2(rays, spheres, r):
    """
    :param rays: n x n_rays x 2 x 3    (axis=2: origin, target)
    :param spheres: n x n_spheres x 3  (axis=2: x, y, z)
    :param r: n_spheres
    :return: n x n_rays x n_spheres (boolean array) with res[:, o, j] = True if ray o intersects with sphere j
    Formula from: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

    rays = np.random.random((10, 4, 2, 3))
    spheres = np.random.random((10, 5, 3))
    r = np.ones(5) * 0.1
    res = ray_sphere_intersection_2(rays=rays, spheres=spheres, r=r)
    """

    o = rays[:, :, 0]
    u = np.diff(rays, axis=-2)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    c = spheres[..., :3]
    co = (o[:, :, np.newaxis, :] - c[:, np.newaxis, :, :])
    res = (u * co).sum(axis=-1)**2 - (co**2).sum(axis=-1) + r**2
    return res >= 0


def rotation_between_vectors(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    s = np.linalg.norm(v, axis=-1)
    c = (a * b).sum(axis=-1)

    vx = np.zeros(a.shape[:-1] + (3, 3))
    vx[..., 0, 1] = -v[..., 2]
    vx[..., 0, 2] = v[..., 1]
    vx[..., 1, 0] = v[..., 2]
    vx[..., 1, 2] = -v[..., 0]
    vx[..., 2, 0] = -v[..., 1]
    vx[..., 2, 1] = v[..., 0]

    i = np.zeros(a.shape[:-1] + (3, 3))
    i[..., :, :] = np.eye(3)

    r = i + vx + ((1 - c) / s ** 2)[..., np.newaxis, np.newaxis]*(vx @ vx)
    return r


def test_rotation_between_vectors():
    a = np.array([1, 0, 0], dtype=np.float64)
    b = np.array([0, 0, 1], dtype=np.float64)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx)*(1-c)/(s**2)
    print(r)


def sample_points_on_disc(radius, size=None):
    rho = np.sqrt(np.random.uniform(0, radius**2, size=size))
    theta = np.random.uniform(0, 2*np.pi, size)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return x, y


def sample_points_on_sphere_3d(size):
    size = shape_wrapper(shape=size)
    x = np.empty(tuple(size) + (3,))
    theta = np.random.uniform(low=0, high=2*np.pi, size=size)
    phi = np.arccos(1-2*np.random.uniform(low=0, high=1, size=size))
    sin_phi = np.sin(phi)
    x[..., 0] = sin_phi * np.cos(theta)
    x[..., 1] = sin_phi * np.sin(theta)
    x[..., 2] = np.cos(phi)

    return x


def sample_points_on_sphere_nd(size, n_dim, ):

    # if np.shape(shape) < 2:
    #     safety = 100
    # else:

    safety = 1.2

    size = shape_wrapper(shape=size)
    volume_sphere = hyper_sphere_volume(n_dim)
    volume_cube = 2**n_dim
    safety_factor = int(np.ceil(safety * volume_cube/volume_sphere))

    size_w_ndim = size + (n_dim,)
    size_sample = (safety_factor,) + size_w_ndim

    x = np.random.uniform(low=-1, high=1, size=size_sample)
    x_norm = np.linalg.norm(x, axis=-1)
    bool_keep = x_norm < 1
    n_keep = bool_keep.sum()
    # print(n_keep / np.shape(shape))
    assert n_keep > np.size(size)
    raise NotImplementedError


def hyper_sphere_volume(n_dim, r=1.):
    """https: // en.wikipedia.org / wiki / Volume_of_an_n - ball"""
    n2 = n_dim//2
    if n_dim % 2 == 0:
        return (np.pi ** n2) / np.math.factorial(n2) * r**n_dim
    else:
        return 2*(np.math.factorial(n2)*(4*np.pi)**n2) / np.math.factorial(n_dim) * r**n_dim


def get_points_on_circle(x, r, n=10, endpoint=False):
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=endpoint)
    sc = np.stack((np.sin(theta), np.cos(theta))).T
    points = x[..., np.newaxis, :] + r[..., np.newaxis, np.newaxis] * sc
    return points


def get_points_on_multicircles(x, r, n=10, endpoint1=False, endpoint2=True):
    points = get_points_on_circle(x, r, n=n, endpoint=endpoint1)
    hull = ConvexHull(points.reshape(-1, 2))
    if endpoint2:
        i = np.concatenate((hull.vertices, hull.vertices[:1]))
    else:
        i = hull.vertices
    hull = points.reshape(-1, 2)[i]
    return points, hull
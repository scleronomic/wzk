import numpy as np

from itertools import combinations
from scipy.spatial import ConvexHull

from wzk.numpy2 import shape_wrapper
from wzk.dicts_lists_tuples import change_tuple_order


def projection_point_line(p, x0, x1, clip=False):
    """
    http:#mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    Projection from point p to the line defined by {x0 + mu*x1}
    """
    x21 = x1 - x0
    mu = -((x0 - p) * x21).sum(axis=-1) / (x21 * x21).sum(axis=-1)
    if clip:
        mu = np.clip(mu, 0, 1)
    x0_p = x0 + mu * x21
    return x0_p


def distance_point_line(p, x0, x1, clip=False):
    pp = projection_point_line(p=p, x0=x0, x1=x1, clip=clip)
    return np.linalg.norm(pp - p, axis=-1)


def __flip_and_clip_mu(mu):
    #  x | x | x
    #    0   1
    #  x | 0 | x - 1

    diff_mu = np.zeros_like(mu)
    b = mu < 0
    diff_mu[b] = mu[b]
    b = mu > 1
    diff_mu[b] = mu[b] - 1
    return diff_mu


def __clip_ppp(o, u, v, uu, vv):

    n = np.cross(u, v)
    nn = (n*n).sum(axis=-1)
    mua = (+n * np.cross(v, o)).sum(axis=-1) / nn
    mub = (-n * np.cross(u, o)).sum(axis=-1) / nn

    uv = (u*v).sum(axis=-1)
    mua2 = mua + uv / uu * __flip_and_clip_mu(mu=mub)
    mub2 = mub + uv / vv * __flip_and_clip_mu(mu=mua)
    return np.clip(mua2, 0, 1), np.clip(mub2, 0, 1)


def projection_point_plane(p, o, u, v, clip=False):
    """
    Projection of point p on the plane ouv.
    Defined by its origin o and two vectors spanning the plane u and v.
              p .
                |
                v
         v ^  - - -
          /        /
         o ----->
               u

    If clip is True, the projection is clipped / projected on the sheet o, o+u, o+u+v, o+v
    """

    o = o - p
    if clip:
        mua, mub = __clip_ppp(o=o, u=u, v=v, uu=(u*u).sum(axis=-1), vv=(v*v).sum(axis=-1))
        return o + mua * u + mub * v + p

    else:
        n = np.cross(u, v)
        p0 = n * (n*o).sum(axis=-1) / (n*n).sum(axis=-1)
        return p0 + p


def line_line(line_a, line_b, return_mu=False):
    """
    (x1-x3) --- (x1-x4)
       |           |
       |           |
    (x2-x3) --- (x2-x4)

     O ---> V
     |
     v
     U

     U = (x2-x3) - (x1-x3) = x2 - x1
     U = (x1-x4) - (x1-x3) = x3 - x4
    """

    x1, x2 = line_a
    x3, x4 = line_b
    o = x1 - x3
    u = x2 - x1
    v = x4 - x3

    mua, mub = __clip_ppp(o=o, u=u, v=-v, uu=(u*u).sum(axis=-1), vv=(v*v).sum(axis=-1))  # attention sign change for v
    xa = x1 + mua * u
    xb = x3 + mub * v

    if return_mu:
        return (xa, xb), (mua, mub)
    else:
        return xa, xb


def line_line_pairs(lines, pairs, __return_mu=False):
    a, b = pairs.T
    x1, x3 = lines[..., a, 0, :], lines[..., b, 0, :]
    uv_temp = lines[..., :, 1, :] - lines[..., :, 0, :]
    u = uv_temp[..., a, :]
    v = uv_temp[..., b, :]
    o = x1 - x3

    uuvv_temp = (uv_temp * uv_temp).sum(axis=-1)
    mua, mub = __clip_ppp(o=o, u=u, v=-v, uu=uuvv_temp[..., a], vv=uuvv_temp[..., b])  # attention sign change for v
    xa = x1 + mua[..., np.newaxis] * u
    xb = x3 + mub[..., np.newaxis] * v

    if __return_mu:
        return (xa, xb), (mua, mub)
    else:
        return xa, xb

# x = T * y
# ∂x/∂q = ∂T/∂q * y
# d = b - a
# d2 = (b - a)^2
# a = a(a0, a1), a0 = Ta * â0, a1 = Ta * â1
# b = b(b0, b1), b0 = Tb * b^0, b1 = Tb * b^1
# ∂a/∂q = ∂a/∂a0 ∂a0/∂q + ∂a/∂a1 ∂a1/∂q
# ∂a/∂q = (mua-1) ∂Ta/∂q * â0 + (-mua) ∂Ta/∂q * â1
# ∂a/∂q =  ∂Ta/∂q * [(mua-1)*â0 - mua*â1]
# ∂b/∂q =  ∂Tb/∂q * [(1-mub)*b^0 + mub*b^1]

# ∂d2/∂q = ∂d2/∂a ∂a/dq + ∂d2/∂b ∂b/dq


def test_jac_full():
    from Kinematic.Robots import Justin19
    robot = Justin19()

    q = robot.sample_q(10)
    f, j = robot.get_frames_jac(q)
    pairs = np.array([[13, 22],
                      [13, 4],
                      [22, 4],
                      [10, 20]])
    capsule_pos = np.random.random((27, 2, 4))
    capsule_pos[..., -1] = 1

    x_capsules = (f[:, :, np.newaxis, :, :] @ capsule_pos[np.newaxis, :, :, :,  np.newaxis])[..., 0]
    line_line_pairs_d2_jac(lines=x_capsules, pairs=pairs)


def line_line_pairs_d2_jac(lines, pairs):
    (xa, xb), (mua, mub) = line_line_pairs(lines, pairs, __return_mu=False)

    dxaxb_dx = np.zeros(mua.shape + (2, 2))
    dxaxb_dx[..., 0, 0] = +mua - 1
    dxaxb_dx[..., 0, 1] = -mua
    dxaxb_dx[..., 1, 0] = -mub + 1
    dxaxb_dx[..., 0, 1] = +mub

    d = xb - xa
    d2 = (d*d).sum(axis=-1)
    dd2_dx = 2*d * dxaxb_dx
    return d2, dd2_dx

def d2_mink(x):
    x1, x2, x3, x4 = x
    xa, xb = line_line(line_a=np.array((x1, x2)),
                       line_b=np.array((x3, x4)))
    d = xb - xa
    d2 = (d*d).sum(axis=-1)
    return d2


def aas():
    pass


def d2_mink_jac(x):
    x1, x2, x3, x4 = x

    u = x2 - x1
    v = x4 - x3

    (xa, xb), (mua,mub) = line_line(line_a=np.array((x1, x2)),
                                    line_b=np.array((x3, x4)), return_mu=True)
    d = xb - xa
    d2 = (d*d).sum(axis=-1)

    dxaxb_dx = np.zeros((4, 1))
    dxaxb_dx[0] = +mua - 1
    dxaxb_dx[1] = -mua
    dxaxb_dx[2] = -mub + 1
    dxaxb_dx[3] = +mub

    _d = x3 - x1 + mub*v - mua*u
    dd2_dx = 2 * _d * dxaxb_dx
    return dd2_dx



def capsule_capsule(line_a, line_b, radius_a, radius_b):
    xa, xb = line_line(line_a, line_b)
    d = xb - xa
    d_n = d / np.linalg.norm(d)
    xa = xa + d_n * radius_a
    xb = xb - d_n * radius_b
    return xa, xb


def capsule_capsule_pairs(lines, pairs, radii):
    xa, xb = line_line_pairs(lines=lines, pairs=pairs)
    d = xb - xa
    d_n = d / np.linalg.norm(d, axis=-1)
    xa = xa + d_n * radii[pairs[:, 0]]
    xb = xb - d_n * radii[pairs[:, 1]]
    return xa, xb


def distance_point_plane(p, o, u, v, clip=False):
    pp = projection_point_plane(p=p, o=o, u=u, v=v, clip=clip)
    return np.linalg.norm(pp - p, axis=-1)


def circle_circle_intersection(xy0, r0, xy1, r1):
    """
    https:#stackoverflow.com/a/55817881/7570817
    https:#mathworld.wolfram.com/Circle-CircleIntersection.html

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
    Formula from: https:#en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
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
    Formula from: https:#en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

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
    rho = np.sqrt(np.random.uniform(low=0, high=radius**2, size=size))
    theta = np.random.uniform(low=0, high=2*np.pi, size=size)
    xy = np.empty(np.shape(theta)+(2,))
    xy[..., 0] = rho * np.cos(theta)
    xy[..., 1] = rho * np.sin(theta)

    return xy


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
    """https: # en.wikipedia.org / wiki / Volume_of_an_n - ball"""
    n2 = n_dim#2
    if n_dim % 2 == 0:
        return (np.pi ** n2) / np.math.factorial(n2) * r**n_dim
    else:
        return 2*(np.math.factorial(n2)*(4*np.pi)**n2) / np.math.factorial(n_dim) * r**n_dim


def get_points_on_circle(x, r, n=10, endpoint=False):
    r = np.atleast_1d(r)
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=endpoint)
    sc = np.stack((np.sin(theta), np.cos(theta))).T
    points = x[..., np.newaxis, :] + r[..., np.newaxis, np.newaxis] * sc
    return points


def get_points_on_multicircles(x, r, n=10, endpoint1=False, endpoint2=True):
    points = get_points_on_circle(x=x, r=r, n=n, endpoint=endpoint1)
    hull = ConvexHull(points.reshape(-1, 2))
    if endpoint2:
        i = np.concatenate((hull.vertices, hull.vertices[:1]))
    else:
        i = hull.vertices
    hull = points.reshape(-1, 2)[i]
    return points, hull


def get_points_on_sphere(x, r, n=100, mode='fibonacci', squeeze=True):
    x = np.atleast_2d(x)
    r = np.atleast_1d(r)
    if mode == 'fibonacci':
        x = x[:, np.newaxis, :] + r[..., np.newaxis, np.newaxis]*fibonacci_sphere(n=n)[np.newaxis, :, :]
    else:
        raise ValueError

    if squeeze and np.size(r) == 1:
        x = x[0]

    return x


def get_points_on_multisphere(x, r, n):
    if isinstance(r, float):
        r = np.full(len(x), r)
    points = get_points_on_sphere(x=x, r=r, n=n)
    hull = ConvexHull(points.reshape(-1, 3))
    return points, hull


def fibonacci_sphere(n=100):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = np.linspace(1, -1, n)
    r = np.sqrt(1 - y*y)              # radius at y
    theta = phi * np.arange(n)
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    return np.array((x, y, z)).T


def test_lines(n=2):
    from Util.Visualization.pyvista2 import plot_connections
    import pyvista as pv

    lines = np.random.random((n, 2, 3))
    pairs = np.array(list(combinations(np.arange(n), 2)))
    pairs2 = np.arange(2*len(pairs)).reshape(len(pairs), 2)
    lines2 = np.array(line_line_pairs(lines=lines, pairs=pairs)).swapaxes(0, -2)

    p = pv.Plotter()
    h_lines = plot_connections(x=lines, pairs=pairs, p=p, opacity=1, color='blue')
    h_lines2 = plot_connections(x=lines2, pairs=pairs2, p=p, opacity=1, color='red')

    def on_drag(point, i):
        print(i)
        lines.reshape(n * 2, 3)[i] = point
        lines2[:] = np.array(line_line_pairs(lines=lines, pairs=pairs)).swapaxes(0, -2)

        plot_connections(x=lines, pairs=pairs, p=p, opacity=1, color='blue', h=h_lines)
        plot_connections(x=lines2, pairs=pairs2, p=p, opacity=1, color='red', h=h_lines2)


    p.add_sphere_widget(on_drag, center=lines.reshape(n*2, 3), radius=0.01, color='yellow')
    p.show()


def test_capsules():
    from Util.Visualization.pyvista2 import plot_connections, plot_convex_hull
    from wzk import get_points_on_multisphere
    import pyvista as pv

    n = 2
    lines = np.random.random((n, 2, 3))
    radii = np.random.random(n)
    pairs = np.array(list(combinations(np.arange(n), 2)))
    pairs2 = np.arange(2*len(pairs)).reshape(len(pairs), 2)
    lines2 = np.array(capsule_capsule_pairs(lines=lines, pairs=pairs, radii=radii)).swapaxes(0, -2)

    hulls = [get_points_on_multisphere(x=xx, r=rr, n=100)[1] for xx, rr in zip(lines, radii)]

    p = pv.Plotter()
    h_lines = plot_connections(x=lines, pairs=pairs, p=p, opacity=1, color='blue')
    h_lines2 = plot_connections(x=lines2, pairs=pairs2, p=p, opacity=1, color='red')
    h_hulls = [plot_convex_hull(x=None, hull=h, p=p, opacity=0.5) for h in hulls]

    def on_drag(point, i):
        lines.reshape(n * 2, 3)[i] = point

        hulls[:] = [get_points_on_multisphere(x=xx, r=rr, n=100)[1] for xx, rr in zip(lines, radii)]
        lines2[:] = np.array(capsule_capsule_pairs(lines=lines, pairs=pairs, radii=radii)).swapaxes(0, -2)

        plot_connections(x=lines, pairs=pairs, p=p, opacity=1, color='blue', h=h_lines)
        plot_connections(x=lines2, pairs=pairs2, p=p, opacity=1, color='red', h=h_lines2)
        [plot_convex_hull(x=None, hull=h, p=p, opacity=0.5, h=hh) for hh, h in zip(h_hulls, hulls)]

    p.add_sphere_widget(on_drag, center=lines.reshape(n*2, 3), radius=0.1, color='yellow')
    p.show()


def test_speed():
    from wzk import tic, toc
    n = 12
    m = 50
    lines = np.random.random((m, n, 2, 3))
    pairs = np.array(list(combinations(np.arange(n), 2)))

    tic()
    n = 40
    spheres = np.random.random((m, n, 3))
    pairs2 = np.array(list(combinations(np.arange(n), 2)))

    tic()
    for i in range(100):
        res = line_line_pairs(lines=lines, pairs=pairs)
    toc('mink')

    tic()
    for i in range(100):
        res = line_line_pairs(lines=lines, pairs=pairs)
    toc('d1234')


def line_line33(u, v, w):
    # TODO, use in Cpp
    raise NotImplementedError("http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment()")
    # a = (u*u).sum(axis=-1)        # always >= 0
    # b = (u*v).sum(axis=-1)
    # c = (v*v).sum(axis=-1)        # always >= 0
    # d = (u*w).sum(axis=-1)
    # e = (v*w).sum(axis=-1)
    # D = a*c - b*b        # always >= 0
    # # sc, sN  # sc = sN / sD, default sD = D >= 0
    # # tc, tN  # tc = tN / tD, default tD = D >= 0
    # sD = D
    # tD = D
    # eps = 1e-6
    # # compute the line parameters of the two closest points
    #
    # i = d > eps
    # sN = np.where(i, b*e - c*d, 0)
    # sD = np.where(i, a*e - b*d, 1)
    #
    # i = sN < 0
    # sN[i] = 0
    # tD[i] = e
    #
    # tN[i] = e
    #
    # if D < 1e-9:  # the lines are almost parallel
    #     sN = 0.0       # force using point P0 on segment S1
    #     sD = 1.0       # to prevent possible division by 0.0 later
    #     tN = e
    #     tD = c
    #
    # else:                 # get the closest points on the infinite lines
    #     sN = (b*e - c*d)
    #     tN = (a*e - b*d)
    #     if sN < 0.0:  #        # sc < 0 => the s=0 edge is visible
    #         sN = 0.0
    #         tN = e
    #         tD = c
    #     elif sN > sD:  # sc > 1  => the s=1 edge is visible
    #         sN = sD
    #         tN = e + b
    #         tD = c
    #
    # if tN < 0.0:            # tc < 0 => the t=0 edge is visible
    #     tN = 0.0
    #     # recompute sc for this edge
    #     if -d < 0.0:
    #         sN = 0.0
    #     elif -d > a:
    #         sN = sD
    #     else:
    #         sN = -d
    #         sD = a
    #
    #
    # elif tN > tD:     # tc > 1  => the t=1 edge is visible
    #     tN = tD
    #     # recompute sc for this edge
    #     if (-d + b) < 0.0:
    #         sN = 0
    #     elif (-d + b) > a:
    #         sN = sD
    #     else:
    #         sN = (-d +  b)
    #         sD = a
    #
    # # finally do the division to get sc and tc
    # sc = (abs(sN) < SMALL_NUM ? 0.0 : sN / sD)
    # tc = (abs(tN) < SMALL_NUM ? 0.0 : tN / tD)
    #
    # # get the difference of the two closest points
    # dP = w + (sc * u) - (tc * v)  # =  S1(sc) - S2(tc)
    #
    # return norm(dP);   # return the closest distance


def test_jac_mink():
    from wzk import numeric_derivative, print_progress, tic, toc

    n = 1000
    for i in range(n):
        print_progress(i, n=n)
        x = np.random.random((4, 3))
        j = d2_mink_jac(x)
        j_true = numeric_derivative(fun=d2_mink, x=x.copy(), axis=(0, 1))
        if not np.allclose(j, j_true):
            print(i)
            raise ValueError('Wrong Derivative')

    print(f'All {n} tests were equal to the numeric derivative')



if __name__ == '__main__':
    pass

    test_jac_mink()
    # test_speed()
    # for i in range(1000):
    #     la, lb = np.random.random((2, 2, 3))
    #     res = line_line(la, lb)
    #     d = np.linalg.norm(res[1] - res[0])
    #     res_mink = line_line_mink(la, lb)
    #     d_mink = np.linalg.norm(res_mink[1] - res_mink[0])
    #     b = np.allclose(d, d_mink) or d > d_mink
    #     if not b:
    #         print(d, d_mink)


    # test_capsules()
    # test_lines(n=2)
    # symbolic_jac()

import numpy as np
from scipy.spatial.transform import Rotation

from wzk import ltd, np2, random2, geometry, trajectory
from wzk.math2 import angle2minuspi_pluspi  # noqa
from wzk.spatial.util import initialize_frames, fill_frames_trans
from wzk.spatial.transform_2d import theta2dcm

# angle axis representation is like an onion, the singularity is the boarder to the next 360 shell
# 0 is 1360 degree away from the next singularity -> nice  # what???

# Nomenclature
# dcm ~ SE3 matrix (3x3)
# frame ~ (4x4) homogeneous matrix, SE3 + translation


# vectorized versions of scipy's Rotation.from_x().to_y()
def euler2dcm(euler, seq="ZXZ"):
    """ZXZ = roll pitch yaw"""
    return Rotation.from_euler(seq, angles=euler.reshape((-1, 3)),
                               ).as_matrix().reshape(euler.shape[:-1] + (3, 3))


def quaternions2dcm(quat):
    return Rotation.from_quat(quat.reshape((-1, 4))
                              ).as_matrix().reshape(quat.shape[:-1] + (3, 3))


def rotvec2dcm(rotvec):
    return Rotation.from_rotvec(rotvec.reshape((-1, 3))
                                ).as_matrix().reshape(rotvec.shape[:-1] + (3, 3))


def dcm2euler(dcm, seq="ZXZ"):
    return Rotation.from_matrix(dcm.reshape((-1, 3, 3))
                                ).as_euler(seq=seq).reshape(dcm.shape[:-2] + (3,))


def dcm2quaternions(dcm):
    return Rotation.from_matrix(dcm.reshape((-1, 3, 3))
                                ).as_quat().reshape(dcm.shape[:-2] + (4,))


def dcm2rotvec(dcm):
    return Rotation.from_matrix(dcm.reshape((-1, 3, 3))
                                ).as_rotvec().reshape(dcm.shape[:-2] + (3,))


# frame2rotation
# ----------------------------------------------------------------------------------------------------------------------
def frame2dcm(f):
    return f[..., :3, :3]


def frame2quat(f):
    return dcm2quaternions(f[..., :3, :3])


def frame2euler(f, seq="ZXZ"):
    return dcm2euler(f[..., :3, :3], seq=seq)


def frame2rotvec(f):
    return dcm2rotvec(f[..., :3, :3])


def frame2rotz(f):
    z_axis = f[..., :3, 2]
    a = np.arctan2(z_axis[..., 1], z_axis[..., 0])
    return a
    

# frame2trans_rot
# ----------------------------------------------------------------------------------------------------------------------
def frame2trans(f):
    return f[..., :-1, -1]


def frame2trans_dcm(f):
    return frame2trans(f), frame2dcm(f)


def frame2trans_rotvec(f):
    return frame2trans(f), frame2rotvec(f=f)


def frame2trans_quat(f):
    return frame2trans(f), frame2quat(f=f)


def frame2trans_euler(f, seq="ZXZ"):
    return frame2trans(f), frame2euler(f=f, seq=seq)


# 2frame
# ----------------------------------------------------------------------------------------------------------------------
def rotx2frame(alpha):
    if isinstance(alpha, (np.ndarray, tuple, list)):
        return np.array([rotx2frame(a) for a in alpha])

    return np.array([[1, 0, 0, 0],
                     [0, +np.cos(alpha), -np.sin(alpha), 0],
                     [0, +np.sin(alpha), +np.cos(alpha), 0],
                     [0, 0, 0, 1]])


def roty2frame(beta):
    if isinstance(beta, (np.ndarray, tuple, list)):
        return np.array([roty2frame(b) for b in beta])

    return np.array([[+np.cos(beta), 0, +np.sin(beta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(beta), 0, +np.cos(beta), 0],
                     [0, 0, 0, 1]])


def rotz2frame(gamma):
    if isinstance(gamma, (np.ndarray, tuple, list)):
        return np.array([rotz2frame(g) for g in gamma])

    return np.array([[+np.cos(gamma), -np.sin(gamma), 0, 0],
                     [+np.sin(gamma), +np.cos(gamma), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def trans2frame(xyz=None, x=None, y=None, z=None):
    if xyz is not None:
        f = initialize_frames(shape=xyz.shape[:-1], n_dim=xyz.shape[-1], mode="eye")
        f[..., :-1, -1] = xyz
        return f

    n = np2.max_size(*ltd.remove_nones([x, y, z]))
    f = initialize_frames(shape=n, n_dim=3, mode="eye")
    if x is not None:
        f[..., 0, -1] = x
    if y is not None:
        f[..., 1, -1] = y
    if z is not None:
        f[..., 2, -1] = z

    return f


def trans_rot2frame(x, a, axis, squeeze=True):
    """translation + rotation around one axis -> frame"""
    x = np.atleast_2d(x)
    n = len(x)

    rv = np.zeros((n, 3))
    rv[:, axis] = a
    f = trans_rotvec2frame(trans=x, rotvec=rv)

    if squeeze and n == 1:
        f = np.squeeze(f)
    return f
    

def trans_quat2frame(trans=None, quat=None):
    s = np2.get_max_shape(trans, quat)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = quaternions2dcm(quat=quat)
    return f


def trans_rotvec2frame(trans=None, rotvec=None):
    s = np2.get_max_shape(trans, rotvec)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = rotvec2dcm(rotvec=rotvec)
    return f


def trans_euler2frame(trans=None, euler=None, seq="ZXZ"):
    s = np2.get_max_shape(trans, euler)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = euler2dcm(euler=euler, seq=seq)
    return f


def vrotxyz2dcm(v_rotx, v_roty, v_rotz):
    s = np2.get_max_shape(v_rotx, v_roty, v_rotz)
    dcm = initialize_frames(shape=s[:-1], n_dim=2, mode="eye")

    if v_rotx is not None:
        dcm[..., :, 0] = v_rotx
    if v_roty is not None:
        dcm[..., :, 1] = v_roty
    if v_rotz is not None:
        dcm[..., :, 2] = v_rotz

    return dcm


def trans_vrotxyz2frame(trans, v_rotx, v_roty, v_rotz):
    dcm = vrotxyz2dcm(v_rotx=v_rotx, v_roty=v_roty, v_rotz=v_rotz)
    f = trans_dcm2frame(trans=trans, dcm=dcm)
    return f


def trans_dcm2frame(trans=None, dcm=None):
    s = np2.get_max_shape(trans, dcm)
    if dcm is not None:
        s = s[:-1]

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    if dcm is not None:
        f[..., :-1, :-1] = dcm
    return f


# Sampling 
# ----------------------------------------------------------------------------------------------------------------------
def sample_quaternions(shape=None):
    """
    Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning, James J. Kuffner (2004)
    https://ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
    """
    s = np.random.random(shape)
    sigma1 = np.sqrt(1 - s)
    sigma2 = np.sqrt(s)

    theta1 = np.random.uniform(0, 2 * np.pi, shape)
    theta2 = np.random.uniform(0, 2 * np.pi, shape)

    w = np.cos(theta2) * sigma2
    x = np.sin(theta1) * sigma1
    y = np.cos(theta1) * sigma1
    z = np.sin(theta2) * sigma2
    return np.stack([w, x, y, z], axis=-1)


def sample_dcm(shape=None):
    quat = sample_quaternions(shape=shape)
    return quaternions2dcm(quat=quat)


def sample_dcm_noise(shape=None, scale=0.01, mode="normal", n_dim=3):
    """
    samples rotation dcm with the absolute value of the rotation relates to 'scale' in rad
    """

    if n_dim == 3:
        rv = geometry.sample_points_on_sphere_3d(shape)
        rv *= random2.noise(shape=rv.shape[:-1], scale=scale, mode=mode)[..., np.newaxis]
        return rotvec2dcm(rotvec=rv)

    elif n_dim == 2:
        theta = np.random.uniform(low=0, high=2*np.pi, size=shape)
        if shape[-1] == 1:
            theta = theta[..., np.newaxis]

        return theta2dcm(theta=theta)

    else:
        raise ValueError(f"n_dim={n_dim} not supported, only [2, 3]")


def round_dcm(dcm, decimals=0):
    """Round dcm to degrees
    See numpy.round for more information
    decimals=+2: 123.456 -> 123.45
    decimals=+1: 123.456 -> 123.4
    decimals= 0: 123.456 -> 123.0
    decimals=-1: 123.456 -> 120.0
    decimals=-2: 123.456 -> 100.0
    """
    euler = dcm2euler(dcm)
    euler = np.rad2deg(euler)
    euler = np.round(euler, decimals=decimals)
    euler = np.deg2rad(euler)
    return euler2dcm(euler)


def sample_frames(x_low=np.zeros(3), x_high=np.ones(3), shape=None):
    assert len(x_low) == 3  # n_dim == 3
    return trans_quat2frame(trans=random2.random_uniform_ndim(low=x_low, high=x_high, shape=shape),
                            quat=sample_quaternions(shape=shape))


def apply_noise(f, trans, rot, mode="normal"):
    n_dim = f.shape[-1] - 1
    s = tuple(np.array(np.shape(f))[:-2])

    f2 = f.copy()
    f2[..., :-1, -1] += random2.noise(shape=s + (n_dim,), scale=trans, mode=mode)
    f2[..., :-1, :-1] = f2[..., :-1, :-1] @ sample_dcm_noise(shape=s, scale=rot, mode=mode, n_dim=n_dim)
    return f2


def sample_around_f(f, trans, rot, mode="normal", shape=None):
    shape = np2.shape_wrapper(shape)
    f0 = np.zeros(shape+f.shape)
    f0[:] = f.copy()
    return apply_noise(f=f0, trans=trans, rot=rot, mode=mode)


def sample_frame_noise(trans, rot, shape=None, mode="normal"):
    f = initialize_frames(shape=shape, n_dim=3, mode="eye")
    return apply_noise(f=f, trans=trans, rot=rot, mode=mode)


def sample_frames_on_noisy_grid(x_grid, y_grid, z_grid,
                                f0, noise_trans, noise_rot,
                                n_samples):
    f_list = []
    n_total = len(x_grid) * len(y_grid) * len(z_grid)
    n_per_cell = int(np.ceil(n_samples / n_total))

    for x in x_grid:
        for y in y_grid:
            for z in z_grid:
                for n_noise in range(n_per_cell):
                    f = f0.copy()
                    f[..., :3, -1] = [x, y, z]
                    f = apply_noise(f, trans=noise_trans, rot=noise_rot)
                    f_list.append(f)

    f_list = np.array(f_list)
    f_list = f_list[np.random.choice(n_total*n_per_cell, n_samples, replace=False), :, :]
    return f_list


# linalg
# ----------------------------------------------------------------------------------------------------------------------
def get_frames_between(f0, f1, n):

    if np.ndim(f0) == 3 and np.ndim(f1) == 3:
        return np.array([get_frames_between(f0=f0_i, f1=f1_i, n=n) for (f0_i, f1_i) in zip(f0, f1)])

    x = trajectory.get_substeps(x=np.concatenate([f0[:-1, -1:], f1[:-1, -1:]], axis=1).T, n=n-1)

    dm = f0[:-1, :-1].T @ f1[:-1, :-1]
    rv = dcm2rotvec(dm)
    a = np.linalg.norm(rv)
    if np.allclose(a, 0):
        dm2 = np.zeros((n, 3, 3))
        dm2[:] = np.eye(3)
    else:
        rvn = rv/a
        a2 = np.linspace(0, a, n)
        rv2 = a2[:, np.newaxis] * rvn[np.newaxis, :]
        dm2 = rotvec2dcm(rv2)

    m = f0[:-1, :-1] @ dm2
    f = trans_dcm2frame(trans=x, dcm=m)
    return f


def offset_frame(f, i=None, vm=None,
                 offset=0.01):
    """
    offset: in [m]
    i: idx along which axis to offset
    """
    assert (i is None) ^ (vm is None)

    f = f.copy()
    if i is not None:
        d = f[..., :-1, i]

    elif vm is None:
        d = f[..., :-1, :-1] @ vm

    else:
        raise RuntimeError("i and vm can not both be None")

    d = d * offset / np.linalg.norm(d, axis=-1, keepdims=True)
    f[..., :3, -1] -= d
    return f


def apply_eye_wrapper(f, possible_eye):
    if possible_eye is None or np.allclose(possible_eye, np.eye(possible_eye.shape[0])):
        return f
    else:
        return possible_eye @ f


def invert(f):
    """
    Create the inverse of an array of hm f
    Assume n x n are the last two dimensions of the array
    """

    n_dim = f.shape[-1] - 1
    t = f[..., :n_dim, -1]  # Translation

    # Apply the inverse rotation on the translation
    f_inv = f.copy()
    f_inv[..., :n_dim, :n_dim] = np.swapaxes(f_inv[..., :n_dim, :n_dim], axis1=-1, axis2=-2)
    f_inv[..., :n_dim, -1:] = -f_inv[..., :n_dim, :n_dim] @ t[..., np.newaxis]
    return f_inv


def Ax(A, x):
    if x.shape[-1] != A.shape[-1]:
        x = np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)

    if A.ndim == 2 and x.ndim == 2:
        A = A[np.newaxis, :, :]

    if A.ndim == 3 and x.ndim == 1:
        x = x[np.newaxis, :]

    b = np.sum(A * x[..., np.newaxis, :], axis=-1)[:, :-1]
    return b


def AxBxC(a, b, c):
    """
    a x (b x c) = (a.c) b  - (a.b) c
    x: cross product
    .: dot product
    """
    return (np.sum(a * c, axis=-1, keepdims=True) * b -
            np.sum(a * b, axis=-1, keepdims=True) * c)


def VxDCM(v, dcm):
    dcmT = np.cross(v[..., np.newaxis, :], np.swapaxes(dcm, -1, -2))
    return np.swapaxes(dcmT, -1, -2)


def try_cross_order():
    j0, j1, r = np.random.random((3, 3))

    r0 = np.cross(np.cross(j0, j1), r)
    r1 = np.cross(np.cross(j0, r), j1)
    r2 = np.cross(np.cross(j1, r), j0)
    r0 - r1 + r2

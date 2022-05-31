import numpy as np
from scipy.spatial.transform import Rotation

from wzk.numpy2 import shape_wrapper
from wzk.random2 import noise, random_uniform_ndim
from wzk.geometry import sample_points_on_sphere_3d
from wzk.trajectory import get_substeps


# angle axis representation is like an onion, the singularity is the boarder to the next 360 shell
# 0 is 1360 degree away from the next singularity -> nice  # what???

# Nomenclature
# matrix ~ SE3 Matrix (3x3)
# frame ~ (4x4) homogeneous matrix, SE3 + translation


# vectorized versions of scipy's Rotation.from_x().to_y()
def euler2matrix(euler, seq='ZXZ'):
    return Rotation.from_euler(seq, angles=euler.reshape((-1, 3)),
                               ).as_matrix().reshape(euler.shape[:-1] + (3, 3))


def quaternions2matrix(quat):
    return Rotation.from_quat(quat.reshape((-1, 4))
                              ).as_matrix().reshape(quat.shape[:-1] + (3, 3))


def rotvec2matrix(rotvec):
    return Rotation.from_rotvec(rotvec.reshape((-1, 3))
                                ).as_matrix().reshape(rotvec.shape[:-1] + (3, 3))


def matrix2euler(matrix, seq='ZXZ'):
    return Rotation.from_matrix(matrix.reshape((-1, 3, 3))
                                ).as_euler(seq=seq).reshape(matrix.shape[:-2] + (3,))


def matrix2quaternions(matrix):
    return Rotation.from_matrix(matrix.reshape((-1, 3, 3))
                                ).as_quat().reshape(matrix.shape[:-2] + (4,))


def matrix2rotvec(matrix):
    return Rotation.from_matrix(matrix.reshape((-1, 3, 3))
                                ).as_rotvec().reshape(matrix.shape[:-2] + (3,))


# frames2rotation
def frame2quat(f):
    return matrix2quaternions(f[..., :3, :3])


def frame2euler(f, seq='ZXZ'):
    return matrix2euler(f[..., :3, :3], seq=seq)


def frame2rotvec(f):
    return matrix2rotvec(f[..., :3, :3])


def frame2trans_rotvec(f):
    return f[..., :-1, -1], frame2rotvec(f=f)


def frame2trans_quat(f):
    return f[..., :-1, -1], frame2quat(f=f)


# 2frame
def __shape_wrapper(a, b):
    return a.shape if a is not None else b.shape


def fill_frames_trans(f, trans=None):
    if trans is not None:
        f[..., :-1, -1] = trans


def trans_quat2frame(trans=None, quat=None):
    s = __shape_wrapper(trans, quat)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = quaternions2matrix(quat=quat)
    return f


def trans_rotvec2frame(trans=None, rotvec=None):
    s = __shape_wrapper(trans, rotvec)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = rotvec2matrix(rotvec=rotvec)
    return f


def trans_euler2frame(trans=None, euler=None):
    s = __shape_wrapper(trans, euler)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = euler2matrix(euler=euler)
    return f


def trans_matrix2frame(trans=None, matrix=None):
    s = __shape_wrapper(trans, matrix)
    if trans is None:
        s = s[:-1]

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    if matrix is not None:
        f[..., :-1, :-1] = matrix
    return f


def initialize_frames(shape, n_dim, mode='hm', dtype=None, order=None):
    f = np.zeros((shape_wrapper(shape) + (n_dim+1, n_dim+1)), dtype=dtype, order=order)
    if mode == 'zero':
        pass
    elif mode == 'eye':
        for i in range(f.shape[-1]):
            f[..., i, i] = 1
    elif mode == 'hm':
        f[..., -1, -1] = 1
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return f


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


def apply_eye_wrapper(f, possible_eye):
    if possible_eye is None or np.allclose(possible_eye, np.eye(possible_eye.shape[0])):
        return f
    else:
        return possible_eye @ f


# Sampling matrix and quaternions
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


def sample_matrix(shape=None):
    quat = sample_quaternions(shape=shape)
    return quaternions2matrix(quat=quat)


def sample_matrix_noise(shape=None, scale=0.01, mode='normal'):
    """
    samples rotation matrix with the absolute value of the rotation relates to 'scale' in rad
    """

    rv = sample_points_on_sphere_3d(shape)
    rv *= noise(shape=rv.shape[:-1], scale=scale, mode=mode)[..., np.newaxis]
    return rotvec2matrix(rotvec=rv)


def round_matrix(matrix, decimals=0):
    """Round matrix to degrees
    See numpy.round for more information
    decimals=+2: 123.456 -> 123.45
    decimals=+1: 123.456 -> 123.4
    decimals= 0: 123.456 -> 123.0
    decimals=-1: 123.456 -> 120.0
    decimals=-2: 123.456 -> 100.0
    """
    euler = matrix2euler(matrix)
    euler = np.rad2deg(euler)
    euler = np.round(euler, decimals=decimals)
    euler = np.deg2rad(euler)
    return euler2matrix(euler)


def sample_frames(x_low=np.zeros(3), x_high=np.ones(3), shape=None):
    assert len(x_low) == 3  # n_dim == 3
    return trans_quat2frame(trans=random_uniform_ndim(low=x_low, high=x_high, shape=shape),
                            quat=sample_quaternions(shape=shape))


def apply_noise(f, trans, rot, mode='normal'):
    s = tuple(np.array(np.shape(f))[:-2])

    f2 = f.copy()
    f2[..., :3, 3] += noise(shape=s + (3,), scale=trans, mode=mode)
    f2[..., :3, :3] = f2[..., :3, :3] @ sample_matrix_noise(shape=s, scale=rot, mode=mode)
    return f2


def sample_frame_noise(trans, rot, shape, mode='normal'):
    f = initialize_frames(shape=shape, n_dim=3, mode='eye')
    return apply_noise(f=f, trans=trans, rot=rot, mode=mode)


def rot_x(alpha):
    return np.array([[1, 0, 0, 0],
                     [0, +np.cos(alpha), -np.sin(alpha), 0],
                     [0, +np.sin(alpha), +np.cos(alpha), 0],
                     [0, 0, 0, 1]])


def rot_y(beta):
    return np.array([[+np.cos(beta), 0, +np.sin(beta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(beta), 0, +np.cos(beta), 0],
                     [0, 0, 0, 1]])


def rot_z(gamma):  # theta
    return np.array([[+np.cos(gamma), -np.sin(gamma), 0, 0],
                     [+np.sin(gamma), +np.cos(gamma), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def get_frames_between(f0, f1, n):

    x = get_substeps(x=np.concatenate((f0[:-1, -1:], f1[:-1, -1:]), axis=1).T, n=n-1,)

    dm = f0[:-1, :-1].T @ f1[:-1, :-1]
    rv = matrix2rotvec(dm)
    a = np.linalg.norm(rv)
    if np.allclose(a, 0):
        dm2 = np.zeros((n, 3, 3))
        dm2[:] = np.eye(3)
    else:
        rvn = rv/a
        a2 = np.linspace(0, a, n)
        rv2 = a2[:, np.newaxis] * rvn[np.newaxis, :]
        dm2 = rotvec2matrix(rv2)

    m = f0[:-1, :-1] @ dm2
    f = trans_matrix2frame(trans=x, matrix=m)
    return f


def test_get_frames_between():


    n = 10
    f0 = sample_frames()
    f1 = sample_frames()

    f = get_frames_between(f0=f0, f1=f1, n=n)

    assert np.allclose(f0, f[0])
    assert np.allclose(f1, f[-1])
    from wzk.pv.plotting import plot_frames, pv
    p = pv.Plotter()
    plot_frames(p=p, f=f, scale=0.2)
    p.show()

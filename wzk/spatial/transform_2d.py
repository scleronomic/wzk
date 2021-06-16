import numpy as np

from wzk.spatial.transform import initialize_frames, fill_frames_trans


def fill_frames_2d_sc(f, sin, cos):
    f[..., 0, 0] = cos
    f[..., 0, 1] = -sin
    f[..., 1, 0] = sin
    f[..., 1, 1] = cos


def __theta_wrapper(theta):
    if np.isscalar(theta):
        shape = ()
    else:
        shape = np.shape(theta)
        if shape[-1] == 1:
            shape = shape[:-1]
            theta = theta[..., 0]
    return theta, shape


def trans_theta2frame(trans=None, theta=None):
    theta, shape = __theta_wrapper(theta)
    sin, cos = np.sin(theta), np.cos(theta)

    f = initialize_frames(shape=shape, n_dim=2)
    fill_frames_trans(f=f, trans=trans)
    fill_frames_2d_sc(f=f, sin=sin, cos=cos)
    return f


def dframe_dtheta(theta):
    theta, shape = __theta_wrapper(theta)
    sin, cos = np.sin(theta), np.cos(theta)

    j = initialize_frames(shape=shape, n_dim=2, mode='zero')
    fill_frames_2d_sc(f=j, sin=cos, cos=-sin)
    return j


def from_2d_to_3d(f_2d):
    shape = np.array(f_2d.shape)
    shape[-2:] += 1
    frames_3d = np.zeros(shape)

    frames_3d[..., :-2, :-2] = f_2d[..., :-1, :-1]
    frames_3d[..., :-2, -1] = f_2d[..., :-1, -1]
    frames_3d[..., -2, -2] = 1
    frames_3d[..., -1, -1] = 1

    return frames_3d


def frame2trans_theta(f_2d):
    trans = f_2d[..., :-1, -1]
    #                 sin              cos
    theta = np.arctan2(f_2d[..., 1, 0], f_2d[..., 0, 0])
    return trans, theta

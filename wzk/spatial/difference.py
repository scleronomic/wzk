import numpy as np
from wzk.spatial.transform import dcm2rotvec


# Location
def location_difference(loc_a, loc_b):
    return np.linalg.norm(loc_a - loc_b, axis=-1)


def location_difference_cost(loc_a, loc_b):
    loc_diff = loc_a - loc_b
    lco_diff_sqrd = 0.5 * (loc_diff ** 2).sum(axis=-1)
    return loc_diff, lco_diff_sqrd


# Rotation
def rotation_cost(rot):
    """
    Use cos(alpha) as cost metric, it is symmetric around zero, and the derivative of arccos is not needed.

    # cos(theta) = (np.trace(rot, axis1=-2, axis2=-1) - n_dim + 2) / 2
    # cost = 1 - cos(theta)

    # theta = arccos(1-cost)
    # cost = 1 - (np.trace(rot, axis1=-2, axis2=-1) - n_dim + 2) / 2
    # cost = 1 - (np.trace(rot, axis1=-2, axis2=-1) - n_dim) / 2 + 1
    # cost = - (np.trace(rot, axis1=-2, axis2=-1) - n_dim) / 2
    # cost = (n_dim - np.trace(rot, axis1=-2, axis2=-1)) / 2
    """
    n_dim = rot.shape[-1]

    cost = (n_dim - np.trace(rot, axis1=-2, axis2=-1)) / 2
    return cost


def rotation_cost2dist(cost):
    return np.arccos(1 - cost)


def rotation_dist2cost(dist):
    return 1 - np.cos(dist)


def rotation_dist(rot):
    return rotation_cost2dist(cost=rotation_cost(rot=rot))


def rotation_difference(rot_a, rot_b):
    rot = rot_b @ rot_a.swapaxes(-2, -1)
    if rot.shape[-1] == 3:
        return np.linalg.norm(dcm2rotvec(rot), axis=-1)  # This is numerically more stable
    else:
        return np.arccos(rot[..., 0, 0])


def rotation_difference_cost(rot_a, rot_b):
    rot = rot_b @ rot_a.swapaxes(-2, -1)
    return rotation_cost(rot=rot)


# Combined
def frame_difference(f_a, f_b, unit_trans="m", unit_rot="rad", verbose=0):
    loc = location_difference(loc_a=f_a[..., :-1, -1],
                              loc_b=f_b[..., :-1, -1])

    rot = rotation_difference(rot_a=f_a[..., :-1, :-1],
                              rot_b=f_b[..., :-1, :-1])

    if verbose > 0:
        print(f"Max : {np.max(loc*1000):.3f}mm, {np.max(np.rad2deg(rot)):.3f}deg")
        print(f"Mean: {np.mean(loc*1000):.3f}mm, {np.mean(np.rad2deg(rot)):.3f}deg")

    if unit_trans == "mm":
        loc *= 1000

    if unit_rot == "deg":
        rot = np.rad2deg(rot)

    return loc, rot


def frame_difference_cost(f_a, f_b):
    loc = location_difference_cost(loc_a=f_a[..., :-1, -1],
                                   loc_b=f_b[..., :-1, -1])[1]

    rot = rotation_difference_cost(rot_a=f_a[..., :-1, :-1],
                                   rot_b=f_b[..., :-1, :-1])
    return loc, rot

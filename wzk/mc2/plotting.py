import os

import numpy as np

from wzk import grid, bimage, mpl2, spatial, geometry, np2, uuid4
import meshcat
from meshcat import geometry as mg, transformations as mt

Visualizer = meshcat.Visualizer

MeshGeometry_DICT = dict(stl=mg.StlMeshGeometry,
                         obj=mg.ObjMeshGeometry,
                         dae=mg.DaeMeshGeometry)


def get_material(color="gray", alpha=1.0):
    # Set material color from URDF, converting for triplet of doubles to a single int.
    material = mg.MeshPhongMaterial()

    if color is not None:
        color = mpl2.colors.to_rgba(c=color, alpha=alpha)

        material.color = int(color[0] * 255) * 256 ** 2 + int(color[1] * 255) * 256 + int(color[2] * 255)
        if float(color[3]) != 1.0:
            material.transparent = True
            material.opacity = float(color[3])

    else:
        material.color = None

    return material


def wrapper_handle(handle=None, default=""):
    if handle is None:
        handle = f"{default}-{uuid4()}"
    return handle


def plot_points(vis, h,
                x, color=None, alpha=1.):
    h = wrapper_handle(handle=h, default="points")

    assert x.ndim == 2 and x.shape[1] == 3
    color = mpl2.colors.to_rgba(c=color, alpha=alpha)

    vis[h].set_object(mg.PointCloud(position=x.T, color=color))


def plot_bimg_voxel(vis, h,
                    bimg, limits, color=None, alpha=1.0):
    h = wrapper_handle(handle=h, default="bimg")

    material = get_material(color=color, alpha=alpha)
    voxel_size = grid.limits2voxel_size(shape=bimg.shape, limits=limits)

    i = np.array(np.nonzero(bimg)).T
    x = grid.grid_i2x(i=i, limits=limits, shape=bimg.shape, mode="c")
    for j, xx in enumerate(x):
        vis[f"{h}/voxel-{j}"].set_object(geometry=mg.Box([voxel_size] * 3), material=material)
        vis[f"{h}/voxel-{j}"].set_transform(mt.translation_matrix(xx))

    return h


def plot_bimg_mesh(vis, h, bimg, limits, level=0, color=None, alpha=1.0):
    h = wrapper_handle(handle=h, default="bimg")

    material = get_material(color=color, alpha=alpha)

    voxel_size = grid.limits2voxel_size(shape=bimg.shape, limits=limits)
    v, f = bimage.bimg2surf(img=bimg, limits=limits + voxel_size / 2, level=level)
    vis[h].set_object(geometry=mg.TriangularMeshGeometry(vertices=v, faces=f), material=material)
    return h


def plot_bimg(vis, h,
              bimg, limits, mode="mesh", **kwargs):
    if bimg is None:
        return

    if mode == "mesh":
        plot_bimg_mesh(vis=vis, h=h, bimg=bimg, limits=limits, **kwargs)

    elif mode == "voxel":
        plot_bimg_voxel(vis=vis, h=h, bimg=bimg, limits=limits, **kwargs)

    else:
        raise ValueError(f"Unknown mode: '{mode}' | ['mesh', 'voxel']")


def get_default_color_alpha(**kwargs):
    kwargs.update(dict(color=kwargs.pop("color", "gray")))
    kwargs.update(dict(alpha=kwargs.pop("alpha", 1.0)))
    return kwargs


# TODO uniform way set color/ alpha / linewidth
# TODO uniform way to update properties
#   over the unique name it is pretty simple, but I don't know yet how to check if a handle already exists.
#   does not hurt i guess, i just can overwrite it

def plot_arrow(vis, h, x, v, length=1.0, **kwargs):
    x, v = np2.squeeze_all(x, v)
    if np.ndim(x) == 2 or np.ndim(v) == 2:
        n = np2.max_size(x, v) // 3
        kwargs = get_default_color_alpha(**kwargs)
        h, length, color, alpha = np2.scalar2array(h, length, kwargs["color"], kwargs["alpha"], shape=n)
        x, v = np2.scalar2array(x, v, shape=(n, 3))
        return [plot_arrow(vis=vis, h=hh, x=xx, v=vv, lenght=ll, color=cc, alpha=aa)
                for (hh, xx, vv, ll, cc, aa) in zip(h, x, v, length, color, alpha)]

    h = wrapper_handle(handle=h, default="arrow")
    h_cone = f"{h}-cone"
    h_cylinder = f"{h}-cylinder"

    scale_length2width = 0.05
    scale_length2cone_width = 0.1
    scale_length2cone_length = 0.3

    length_cone = length * scale_length2cone_length
    length_cylinder = length - length_cone

    radius_cylinder = length * scale_length2width
    radius_cone = length * scale_length2cone_width

    cylinder = mg.Cylinder(height=length_cylinder, radius=radius_cylinder)
    cone = mg.Cylinder(height=length_cone, radiusBottom=radius_cone, radiusTop=0)

    material = get_material(**kwargs)
    vis[h_cylinder].set_object(geometry=cylinder, material=material)
    vis[h_cone].set_object(geometry=cone, material=material)

    vy = v
    vx = geometry.get_orthonormal(vy)
    vz = np.ones(3)
    dcm = geometry.make_rhs(xyz=np.vstack([vx, vy, vz])).T

    f_cylinder = spatial.trans_dcm2frame(trans=x, dcm=dcm) @ spatial.trans2frame(y=length_cylinder / 2)
    f_cone = spatial.trans_dcm2frame(trans=x, dcm=dcm) @ spatial.trans2frame(y=length_cylinder + length_cone / 2)

    vis[h_cone].set_transform(f_cone)
    vis[h_cylinder].set_transform(f_cylinder)

    return h


def transform(vis, h, f):
    if f is None:
        return

    if isinstance(h, str) and f.ndim == 2:
        vis[h].set_transform(f)

    elif isinstance(h, (list, tuple, np.ndarray)) and f.ndim == 2:
        for hh in zip(h):
            vis[hh].set_transform(f)

    elif f.ndim == 3:
        assert len(h) == len(f)
        for hh, ff in zip(h, f):
            vis[hh].set_transform(ff)

    else:
        raise ValueError


def __load_mesh(mesh):
    ext = os.path.splitext(mesh)[-1][1:]
    print(ext, mesh)
    return MeshGeometry_DICT[ext].from_file(mesh)


def plot_meshes(vis, h, meshes, f=None, color=None, alpha=1.):
    material = get_material(color=color, alpha=alpha)

    for hh, mm in zip(h, meshes):
        vis[hh].set_object(__load_mesh(mm), material)

    transform(vis=vis, h=h, f=f)


def robot_meshes2handles(robot, exclude_idx=None):
    n = len(robot.meshes.files)
    if exclude_idx is None:
        include_idx = np.arange(n)
    else:
        include_idx = set(np.arange(n)).difference(exclude_idx)
        include_idx = np.array(include_idx, dtype=int)

    h = [f"{robot.id}/{os.path.split(os.path.splitext(m)[0])[-1]}-{i}"
         for i, m in enumerate(robot.meshes.files)
         if i in include_idx]
    return h, include_idx


def init_plot_robot(vis, robot,
                    exclude_idx=None,
                    **kwargs):
    h, idx = robot_meshes2handles(robot=robot, exclude_idx=exclude_idx)
    return plot_meshes(vis=vis, h=h, meshes=robot.meshes.files[idx], **kwargs)


def plot_robot_configuration(vis, robot, q,
                             f0=None,
                             exclude_idx=None):
    h, idx = robot_meshes2handles(robot=robot, exclude_idx=exclude_idx)

    f = robot.get_frames(q)
    f = f[robot.meshes.f_idx[idx]]
    f = spatial.apply_f_or_none(f=f, f_or_none=f0)
    f = f @ robot.meshes.f
    transform(vis=vis, h=h, f=f)


def plot_lines(vis, h, x, lines, **kwargs):
    pass
    # TODO


def plot_faces(vis, h, x, faces, **kwargs):
    pass
    # TODO


def plot_cube(vis, h, limits, mode="faces", **kwargs):
    if limits is None:
        return
    v, e, f = geometry.cube(limits=limits)

    if mode == "faces":
        return plot_faces(vis=vis, h=h, x=v, faces=f, **kwargs)
    elif mode == "lines":
        return plot_lines(vis=vis, h=h, x=v, lines=e, **kwargs)
    else:
        raise ValueError


def plot_spheres(vis, h, x, r, **kwargs):
    pass


def plot_coordinate_frames(vis, h, f, scale=1.0, **kwargs):
    xyz_str = "xyz"

    if np.ndim(f) == 2:
        h = wrapper_handle(handle=h, default="frame")

        color = kwargs.pop("color", ("red", "green", "blue"))
        color = np2.scalar2array(color, shape=3)

        for i in range(3):
            plot_arrow(vis=vis, h=f"{h}-{xyz_str[i]}", x=f[:3, -1], v=f[:3, i], length=scale, color=color[i], **kwargs)

        return h

    elif np.ndim(f) == 3:

        n = len(f)
        color = kwargs.pop("color", None)
        h, color = np2.scalar2array(h, color, shape=n)

        return [plot_coordinate_frames(vis=vis, h=h, f=ff, scale=scale, color=cc, **kwargs)
                for hh, ff, cc in zip(h, f, color)]


def try_plot_bimg():
    from wzk.perlin import perlin_noise_3d
    bimg = perlin_noise_3d(shape=(256, 256, 256), res=32) < 0.3
    limits = np.zeros((3, 2))
    limits[:, 1] = 3
    # limits += 0.5

    vis = Visualizer()
    plot_bimg(vis=vis, h=None, bimg=bimg, limits=limits, color="white")


def try_arrow():
    vis = Visualizer()

    vis["triad"].set_object(mg.triad())
    vis["triad1"].set_object(mg.triad())

    f = np.eye(4)
    plot_arrow(vis=vis, h=None, x=f[:3, 3], v=f[:3, 0], alpha=0.5)
    f = spatial.sample_frames()
    plot_arrow(vis=vis, h=None, x=f[:3, 3], v=f[:3, 0], alpha=0.5)

    vis["triad"].set_transform(f)


def try_coordinate_frames(mode="A"):
    vis = Visualizer()

    if mode == "A":
        plot_coordinate_frames(vis=vis, h=None, f=spatial.sample_frames(), color="red", scale=0.1)
        plot_coordinate_frames(vis=vis, h=None, f=spatial.sample_frames(), color="green", scale=0.2)
        plot_coordinate_frames(vis=vis, h=None, f=spatial.sample_frames(), color="blue", scale=0.3)

    elif mode == "B":
        plot_coordinate_frames(vis=vis, h=None, f=spatial.sample_frames(shape=5), color="blue", scale=0.3)

    elif mode == "C":
        plot_coordinate_frames(vis=vis, h=None, f=spatial.sample_frames(shape=5), color="blue", scale=0.3)


if __name__ == "__main__":
    # try_arrow()
    # try_plot_bimg()
    try_coordinate_frames(mode="B")

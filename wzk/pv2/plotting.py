import os
from sys import platform
import numpy as np
import pyvista as pv
from itertools import combinations
from scipy.spatial import ConvexHull
from matplotlib import colors

from wzk.np2 import scalar2array, array2array
from wzk.bimage import bimg2surf
from wzk.spatial import invert
from wzk.geometry import cube

from typing import Union

pv.set_plot_theme('document')

Plotter = pv.Plotter
PolyData = pv.PolyData

headless = False
if platform == 'linux':
    try:
        display = os.environ['DISPLAY']
    except KeyError:
        headless = True


def camera_motion(cp=None, pos=None, focal=None, viewup=None,
                  mode='circle_xy', radius=None, n=100):
    pass
    if viewup is None:
        if cp is None:
            viewup = np.array([0., 0., 1.])
        else:
            viewup = cp[2]
    if focal is None:
        if cp is None:
            focal = np.array([0., 0., 0.])
        else:
            focal = cp[1]
    if pos is None:
        if cp is None:
            pos = np.array([1., 1., 1.])
        else:
            pos = cp[0] - focal

    cp = np.array([pos + focal, focal, viewup])

    if mode == 'circle_xy':
        if radius is None:
            r_xy = np.sqrt(pos[0]**2 + pos[1]**2)
        else:
            r_xy = radius

        for i in range(n):
            angle = i/(n-1) * (2*np.pi)
            cp[0, 0] = focal[0] + r_xy * np.cos(angle)
            cp[0, 1] = focal[1] + r_xy * np.sin(angle)
            yield cp

    else:
        raise NotImplementedError


def plotter_wrapper(pl: Union[pv.Plotter, dict],
                    window_size: tuple = (2048, 1536), camera_position=None,
                    lighting: str = 'three lights', off_screen: bool = False,
                    gif=False):

    if isinstance(pl, dict):
        camera_position = pl.pop('camera_position', None)
        window_size = pl.pop('window_size', window_size)
        lighting = pl.pop('lighting', lighting)
        off_screen = pl.pop('off_screen', off_screen)
        gif = pl.pop('gif', gif)
        off_screen = headless and off_screen

        if off_screen:
            print(platform)
            if platform == 'linux':
                pv.start_xvfb()  # start_xvfb only supported on linux

    if not isinstance(pl, pv.Plotter):
        pl = pv.Plotter(window_size=window_size, off_screen=off_screen, lighting=lighting)

    if camera_position is not None:
        pl.camera_position = camera_position

    if gif:
        pass
        assert isinstance(gif, str)
        if not gif.endswith('.gif'):
            gif = f"{gif}.gif"
        pl.open_gif(gif)  # noqa

    return pl


class DummyPlotter:
    def add_mesh(self, *args):
        pass


def faces2pyvista(x):
    n, d = x.shape
    x2 = np.empty((n, d+1), dtype=int)
    x2[:, 0] = d
    x2[:, 1:] = x
    return x2.ravel()


def pyvista2faces(f: np.ndarray):
    d = int(f[0])
    f = np.reshape(f, (-1, d+1)).copy()
    assert np.all(f[:, 0] == d)
    return f[:, 1:]


def plot_convex_hull(x=None, hull=None,
                     pl=None, h=None,
                     **kwargs):

    if hull is None:
        hull = ConvexHull(x.copy())
    faces = faces2pyvista(hull.simplices)

    if h is None:
        h0 = PolyData(hull.points, faces=faces)
        h1 = pl.add_mesh(h0, **kwargs)
        h = (h0, h1)

    else:
        h[0].points = hull.points.copy()
        h[0].faces = faces.copy()

    return h


def plot_connections(x, pairs=None,
                     pl=None, h=None,
                     **kwargs):

    if x.ndim == 2:
        if pairs is None:
            pairs2 = np.array(list(combinations(np.arange(len(x)), 2)))
        else:
            pairs2 = pairs
    elif x.ndim == 3:
        n, n2, n3 = x.shape
        assert n2 == 2
        assert n3 == 3
        pairs2 = np.arange(n*2).reshape(n, 2)
        x = x.reshape(-1, 3)

    else:
        raise ValueError

    lines = faces2pyvista(pairs2)

    if h is None:
        h0 = PolyData(x, lines=lines)
        h1 = pl.add_mesh(h0, **kwargs)
        h = (h0, h1)
    else:
        h[0].points = x.copy()
        if pairs is not None:
            h[0].lines = lines

    return h


def plot_cube(limits, pl=None, **kwargs):
    # r = [[ll, ll + side_length] for ll in lower_left]
    if limits is None:
        return
    v, e, f = cube(limits=limits)
    plot_connections(x=v, pairs=e, pl=pl, **kwargs)


def plot_collision(pl, xa, xb, ab, **kwargs):
    plot_convex_hull(pl=pl, x=xa, opacity=0.2)
    plot_convex_hull(pl=pl, x=xb, opacity=0.2)
    plot_connections(x=ab, pl=pl, **kwargs)


def set_color(h, color):
    pl = h[1].GetProperty()
    pl.SetColor(colors.to_rgb(color))


def plot_bimg(img, limits,
              mode='mesh',
              pl=None, h=None,
              **kwargs):

    if img is None:
        return

    if mode == 'voxel':
        if h is None:
            x, y, z = np.meshgrid(*(np.linspace(limits[i, 0], limits[i, 1], img.shape[i] + 1) for i in range(3)),
                                  indexing='xy')
            h0 = pv.StructuredGrid(x, y, z)
            hidden_cells = ~img.transpose(2, 0, 1).ravel().astype(bool)
            print(hidden_cells.sum())
            h0.hide_cells(hidden_cells)
            h1 = pl.add_mesh(h0, show_scalar_bar=False, **kwargs)
            h = (h0, h1)
        else:
            h[0].hide_cells(~img.ravel())

    elif mode == 'mesh':
        verts, faces = bimg2surf(img=img, limits=limits)
        faces = faces2pyvista(faces)

        if h is None:
            h0 = PolyData(verts, faces=faces)
            h1 = pl.add_mesh(h0, **kwargs)
            h = (h0, h1)

        else:
            h[0].points = verts
            h[0].faces = faces

    else:
        raise ValueError(f"Unknown mode {mode}; either 'mesh' or 'voxel'")

    return h


def plot_spheres(x, r,
                 pl=None, h=None,
                 **kwargs):
    r = scalar2array(r, shape=len(x), safe=True)
    h0 = [pv.Sphere(center=xi, radius=ri) for xi, ri in zip(x, r)]
    if h is None:
        h1 = [pl.add_mesh(h0i, **kwargs) for h0i in h0]
        h = (h0, h1)
    else:
        for h0i, h0i_new in zip(h[0], h0):
            h0i.overwrite(h0i_new)

    return h


def plot_frames(f,
                scale=1., shift=np.zeros(3),
                pl=None, h=None,
                color=None, opacity=None, **kwargs):

    if np.ndim(f) == 3:
        n = len(f)
        h = scalar2array(h, shape=n, safe=False)
        color = array2array(color, shape=(n, 3))
        opacity = array2array(opacity, shape=(n, 3))
        h = [plot_frames(f=fi, pl=pl, h=hi, color=ci,  opacity=oi, scale=scale, shift=shift,
                         **kwargs) for fi, hi, ci, oi in zip(f, h, color, opacity)]
        return h

    else:
        assert f.shape == (4, 4), f"{f.shape}"
        if color is None or np.all(color == np.array([None, None, None])):
            color = np.eye(3)
            color = tuple(color[i] for i in range(3))

        if opacity is None:
            opacity = np.ones(3)

        color, opacity = scalar2array(color, opacity, shape=3, safe=False)
        h0 = [pv.Arrow(start=f[:3, -1]+shift[i]*f[:3, i], direction=f[:3, i], scale=scale) for i in range(3)]
        if h is None:
            h1 = [pl.add_mesh(h0i, color=color[i], opacity=opacity[i], **kwargs) for i, h0i in enumerate(h0)]
            h = (h0, h1)
        else:
            for i, h0i in enumerate(h[0]):
                h0i.overwrite(h0[i])

        return h


class TransformableMesh:
    def __init__(self, mesh, f0=np.eye(4)):
        super().__init__()
        self.f_oa = f0
        self.mesh = mesh

    def transform(self, f):
        # old points of the mesh p1 = T_oa * p
        # new points of the mesh p2 = T_ob * p
        # p2 = * T_ob * T_oa' * (T_oa * p)
        # p2 = * T_ob * T_oa' * p1

        f_ab = f @ invert(self.f_oa)
        self.f_oa = f.copy()
        if not np.allclose(f_ab, np.eye(4)):
            self.mesh.transform(f_ab.copy(), inplace=True)


def plot_mesh(m, f,
              pl, h, **kwargs):

    if h is None:
        h0 = PolyData(m)
        h0 = TransformableMesh(h0)
        h0.transform(f)
        h1 = pl.add_mesh(h0.mesh, **kwargs)
        h = (h0, h1)

    else:
        (h0, h1) = h
        h0.transform(f)

    return h


def load_meshes2numpy(files):
    meshes = [PolyData(f) for f in files]
    n = [0] + [len(m.points) for m in meshes]
    n_cs = np.cumsum(n)
    faces = [pyvista2faces(f=m.faces)+n_cs[i] for i, m in enumerate(meshes)]
    points = np.concatenate([m.points for m in meshes])
    return points, faces

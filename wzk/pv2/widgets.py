import numpy as np

import pyvista as pv

from wzk import geometry, spatial
from wzk.grid import create_grid


class RHSWidget:
    def __init__(self, pl, origin=None, xyz=None, scale=1.0, color='k',
                 callback=None):

        if origin is None:
            origin = np.zeros(3)
        if xyz is None:
            xyz = np.eye(3)

        self.pl = pl
        self.custom_callback = callback
        self.scale = scale
        self.order = [0, 1]
        plane_widget_options = dict(origin=origin, bounds=self.o2b(origin),
                                    normal_rotation=True,
                                    outline_translation=False, origin_translation=False,
                                    test_callback=False, implicit=True, factor=1)
        self.wx = self.pl.add_plane_widget(self.update_x, normal='x', color='r', **plane_widget_options)
        self.wy = self.pl.add_plane_widget(self.update_y, normal='y', color='g', **plane_widget_options)
        self.wz = self.pl.add_plane_widget(self.update_z, normal='z', color='b', **plane_widget_options)
        self.wo = self.pl.add_sphere_widget(self.update_o, center=origin, radius=scale * 0.1, color=color,
                                            test_callback=False)
        self.set_xyz(xyz=xyz)

    def get_origin(self):
        return np.array(self.wo.GetCenter())

    def get_xyz(self) -> np.ndarray:
        return np.array((self.wx.GetNormal(), self.wy.GetNormal(), self.wz.GetNormal()))

    def get_frame(self):
        return spatial.trans_dcm2frame(trans=self.get_origin(), dcm=self.get_xyz().T)

    def set_origin(self, o):
        self.wx.SetOrigin(o)
        self.wy.SetOrigin(o)
        self.wz.SetOrigin(o)

    def set_xyz(self, xyz):
        self.wx.SetNormal(xyz[:, 0])
        self.wy.SetNormal(xyz[:, 1])
        self.wz.SetNormal(xyz[:, 2])

    def set_frame(self, f):
        self.set_origin(f[:-1, -1])
        self.set_xyz(f[:-1, :-1])

    def set_bounds(self, b):
        self.wx.PlaceWidget(b)
        self.wy.PlaceWidget(b)
        self.wz.PlaceWidget(b)

    def o2b(self, o):
        b = np.repeat(o, 2)
        b[0] -= self.scale
        b[1] += self.scale
        return b

    def update_x(self, normal, origin):  # noqa
        self.update_xyz(i=0)

    def update_y(self, normal, origin):  # noqa
        self.update_xyz(i=1)

    def update_z(self, normal, origin):  # noqa
        self.update_xyz(i=2)

    def update_order(self, i):
        if self.order[0] != i:
            self.order[1:] = self.order[:-1]
            self.order[0] = i
        return self.order

    def update_xyz(self, i):
        xyz = geometry.make_rhs(self.get_xyz(), order=tuple(self.update_order(i=i))).T  # not sure if this is the best way
        self.set_xyz(xyz)

        self.callback()
        self.pl.render()

    def update_o(self, o):
        o = np.array(o)
        xyz = np.array(self.get_xyz()).T
        self.set_bounds(self.o2b(o))
        self.set_origin(o)
        self.set_xyz(xyz)

        self.callback()
        self.pl.render()

    def callback(self):
        if self.custom_callback is not None:
            self.custom_callback(self.get_frame())


def add_rhs_widget(pl, origin, xyz=None,
                   scale=1.0, color='k', callback=None):
    return RHSWidget(pl=pl, origin=origin, xyz=xyz, scale=scale, color=color, callback=callback)


def add_multiple_slider_widgets(pl, ranges, names, grid, idx,
                                callback=None, x0=None,
                                style='modern', title_height=0.02):

    # style_dict = pv.rcParams['slider_style'][style]
    # width = max(style_dict['tube_width'], style_dict['slider_width'])  # what about cap_width ?

    if x0 is None:
        x0 = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) / 2
    x0 = np.squeeze(x0)

    def cb_wrapper(ii):
        def cb(value):
            return callback(value, ii)
        return cb

    grid_xy, grid_s = grid
    h = []
    for k, r in enumerate(ranges):
        i, j = idx[k]
        pointa = np.array((grid_xy[0][i], grid_xy[1][j]))
        pointb = pointa + np.array((grid_s[0], 0))
        h.append(pl.add_slider_widget(callback=cb_wrapper(k),
                                      rng=ranges[k],
                                      value=x0[k],
                                      title=names[k],
                                      pointa=pointa, pointb=pointb,
                                      title_height=title_height,
                                      style=style))

    return h


def add_key_slider_widget(pl, slider, callback, step=1.):
    r = slider.GetRepresentation()
    mi = r.GetMinimumValue()
    ma = r.GetMaximumValue()

    def __on_key(s):
        v2 = r.GetValue() + s
        v2 = np.clip(v2, a_min=mi, a_max=ma)
        r.SetValue(v2)
        callback(v2)
        pl.render()

    def on_left():
        __on_key(s=-step)

    def on_right():
        __on_key(s=+step)

    def on_down():
        __on_key(s=-step*100)

    def on_up():
        __on_key(s=+step*100)

    pl.add_key_event(key='Left', callback=on_left)
    pl.add_key_event(key='Right', callback=on_right)
    pl.add_key_event(key='Down', callback=on_down)
    pl.add_key_event(key='Up', callback=on_up)


class MultipleSpheresWidget:

    def __init__(self, pl, n=None,
                 callback=None, **kwargs):
        self.x = np.zeros((n, 3))
        self.r = np.ones(n) * 0.1

        self.callback = callback if callback is not None else lambda x, r: None

        limits = np.zeros((n, 2))
        limits[:, 0] = 0.01
        limits[:, 1] = 0.20

        grid = create_grid(ll=(0.05, 0.05), ur=(0.95, 0.22), n=(n, 1), pad=(-0.015, 0.05))
        names = [f'{i}' for i in range(n)]
        idx = np.zeros((n, 2), dtype=int)
        idx[:n, 0] = np.arange(n)

        self.h_spheres = [pl.add_sphere_widget(callback=self.update_spheres, center=x, radius=r,
                                               test_callback=False, **kwargs)
                          for x, r in zip(self.x, self.r)]

        self.h_slider = add_multiple_slider_widgets(pl=pl, ranges=limits, grid=grid, idx=idx, names=names,
                                                    callback=self.update_slider, x0=self.r)

    def update_slider(self, value, idx):
        self.r[idx] = value
        self.h_spheres[idx].SetRadius(value)

        self.update()

    def update_spheres(self, *args):
        print(*args)
        self.x = np.array([h.GetCenter() for h in self.h_spheres])

        self.update()

    def update(self):
        print('x', repr(self.x))
        print('r', repr(self.r))
        self.callback(self.x, self.r)


def add_ruler_widget(pl):

    h_center = pv.PolyData(np.random.random((3, 3)))
    h_center["My Labels"] = np.array(['a', 'center', 'b'])
    pl.add_mesh(h_center, color='r', point_size=10)
    # h_labels = pl.add_point_labels(h_center, "My Labels", point_size=20, font_size=36)

    def update(pointa, pointb):
        pointa, pointb = np.atleast_1d(pointa, pointb)
        h_center.points = np.stack([pointa,
                                    pointa + (pointb-pointa)/2,
                                    pointb])

        d = np.linalg.norm(pointa - pointb)
        print(d)
        h_center["My Labels"] = np.array([f"({pointa[0]}, {pointa[1]}, {pointa[2]})",
                                          f"{d:.3f}",
                                          f"({pointb[0]}, {pointb[1]}, {pointb[2]})"])
        pl.render()

    h_ruler = pl.add_line_widget(callback=update, use_vertices=True)

    return h_ruler


if __name__ == '__main__':
    pass
    # import pyvista as pv
    # pl = pv.Plotter()
    # MultipleSpheresWidget(pl=pl, n=3)
    # add_ruler_widget(pl=pl)
    # pl.show()
    #
    # import numpy as np

    # import pyvista as pv
    # # from wzk.pv
    #
    # pl = pv.Plotter()
    # pl.add_axes()
    # h_center = pv.PolyData(np.random.random((3, 3)))
    # h_center["My Labels"] = np.array(['a', 'center', 'b'])
    # pl.add_mesh(h_center, color='r', point_size=10)
    # h_labels = pl.add_point_labels(h_center, "My Labels", point_size=20, font_size=36)
    #
    #
    #
    #
    # h = pl.add_line_widget(callback=simulate, use_vertices=True)
    # pl.show()

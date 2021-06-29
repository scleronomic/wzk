import numpy as np
from wzk.geometry import make_rhs


class RHSWidget:
    def __init__(self, p, origin, size=1.0, color='k',
                 callback=None):

        self.p = p
        self.custom_callback = callback
        self.size = size
        self.order = [0, 1]
        plane_widget_options = dict(origin=origin, bounds=self.o2b(origin),
                                    normal_rotation=True,
                                    outline_translation=False, origin_translation=False,
                                    test_callback=False, implicit=True, factor=1)
        self.wx = p.add_plane_widget(self.update_x, normal='x', color='r', **plane_widget_options)
        self.wy = p.add_plane_widget(self.update_y, normal='y', color='g', **plane_widget_options)
        self.wz = p.add_plane_widget(self.update_z, normal='z', color='b', **plane_widget_options)

        self.wo = p.add_sphere_widget(self.update_o, center=origin, radius=size * 0.1, color=color,
                                      test_callback=False)

    def get_xyz(self):
        return self.wx.GetNormal(), self.wy.GetNormal(), self.wz.GetNormal()

    def set_xyz(self, xyz):
        self.wx.SetNormal(xyz[0])
        self.wy.SetNormal(xyz[1])
        self.wz.SetNormal(xyz[2])

    def set_origin(self, o):
        self.wx.SetOrigin(o)
        self.wy.SetOrigin(o)
        self.wz.SetOrigin(o)

    def get_origin(self):
        return np.array(self.wo.GetCenter())

    def set_bounds(self, b):
        self.wx.PlaceWidget(b)
        self.wy.PlaceWidget(b)
        self.wz.PlaceWidget(b)

    def o2b(self, o):
        b = np.repeat(o, 2)
        b[0] -= self.size
        b[1] += self.size
        return b

    # noinspection PyUnusedLocal
    def update_x(self, normal, origin):
        self.update_xyz(i=0)

    # noinspection PyUnusedLocal
    def update_y(self, normal, origin):
        self.update_xyz(i=1)

    # noinspection PyUnusedLocal
    def update_z(self, normal, origin):
        self.update_xyz(i=2)

    def update_order(self, i):
        if self.order[0] != i:
            self.order[1:] = self.order[:-1]
            self.order[0] = i
        return self.order

    def update_xyz(self, i):
        xyz = make_rhs(self.get_xyz(), order=self.update_order(i=i))
        self.set_xyz(xyz)
        o = self.get_origin()

        self.callback(o, xyz)
        self.p.render()

    def update_o(self, o):
        o = np.array(o)
        xyz = self.get_xyz()
        self.set_bounds(self.o2b(o))
        self.set_origin(o)
        self.set_xyz(xyz)

        self.callback(o, xyz)
        self.p.render()

    def callback(self, o, xyz):
        if self.custom_callback is not None:
            self.custom_callback((o, xyz))


def add_rhs_widget(p, origin, size=1.0, color='k', callback=None):
    return RHSWidget(p=p, origin=origin, size=size, color=color, callback=callback)


def add_multiple_slider_widgets(p, ranges=None, names=None, grid=None, idx=None,
                                callback=None, x0=None,
                                style='modern', title_height=0.02):

    # style_dict = pv.rcParams['slider_style'][style]
    # width = max(style_dict['tube_width'], style_dict['slider_width'])  # what about cap_width ?

    if x0 is None:
        x0 = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) / 2

    def cb_wrapper(ii):
        def cb(value):
            return callback(value, ii)
        return cb

    grid_xy, grid_s = grid
    for k, r in enumerate(ranges):
        i, j = idx[k]
        pointa = np.array((grid_xy[0][i], grid_xy[1][j]))
        pointb = pointa + np.array((grid_s[0], 0))
        p.add_slider_widget(callback=cb_wrapper(k),
                            rng=ranges[k],
                            value=x0[k],
                            title=names[k],
                            pointa=pointa, pointb=pointb,
                            title_height=title_height,
                            style=style)


def add_key_slider_widget(p, slider, callback, step=1.):
    r = slider.GetRepresentation()
    mi = r.GetMinimumValue()
    ma = r.GetMaximumValue()

    def on_left():
        v2 = max(mi, r.GetValue() - step)
        r.SetValue(v2)
        callback(v2)
        p.render()

    def on_right():
        v2 = min(ma, r.GetValue() + step)
        r.SetValue(v2)
        callback(v2)
        p.render()

    p.add_key_event(key='Left', callback=on_left)
    p.add_key_event(key='Right', callback=on_right)

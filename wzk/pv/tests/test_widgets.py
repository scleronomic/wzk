from unittest import TestCase
import pyvista as pv

from wzk.pv.widgets import *
from wzk.pv.plotting import plot_frames


class Test(TestCase):
    pass


def test_add_rhs_widget():
    origin0 = np.zeros(3)
    origin1 = np.ones(3) * (-0.2)
    scale = 0.2
    p = pv.Plotter()

    h = plot_frames(f=np.eye(4), scale=scale, p=p)

    def update(oxyz):
        o, xyz = oxyz
        f = np.eye(4)
        f[:3, :3] = xyz
        f[:3, -1] = o
        plot_frames(f=f, h=h, scale=scale)

    rhs = RHSWidget(p=p, origin=origin0, scale=scale, callback=update)
    RHSWidget(p=p, origin=origin1, scale=.3)

    p.show()


def test_key_slider_widget():
    p = pv.Plotter()
    add_key_slider_widget(p=p, slider=None, callback=lambda x: print('A'), step=2.)
    p.show()


if __name__ == '__main__':
    test_add_rhs_widget()

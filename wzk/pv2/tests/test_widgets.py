from unittest import TestCase

from wzk.pv2 import *


class Test(TestCase):
    pass


def test_add_rhs_widget():
    origin0 = np.zeros(3)
    origin1 = np.ones(3) * (-0.2)
    scale = 0.2
    pl = Plotter()

    h = plot_frames(pl=pl, f=np.eye(4), scale=scale)

    def update(oxyz):
        o, xyz = oxyz
        f = np.eye(4)
        f[:3, :3] = xyz
        f[:3, -1] = o
        plot_frames(f=f, h=h, scale=scale)

    RHSWidget(pl=pl, origin=origin0, scale=scale, callback=update)
    RHSWidget(pl=pl, origin=origin1, scale=.3)

    pl.show()


def test_key_slider_widget():
    pl = pv.Plotter()
    add_key_slider_widget(pl=pl, slider=None, callback=lambda x: print('A'), step=2.)
    pl.show()


if __name__ == '__main__':
    test_add_rhs_widget()

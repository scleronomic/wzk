from unittest import TestCase

import numpy as np
from wzk import pv2


class Test(TestCase):
    pass


def test_add_rhs_widget():
    origin0 = np.zeros(3)
    origin1 = np.ones(3) * (-0.2)
    scale = 0.2
    pl = pv2.Plotter()

    h = pv2.plot_coordinate_frames(pl=pl, f=np.eye(4), scale=scale)

    def update(f):
        pv2.plot_coordinate_frames(f=f, h=h, scale=scale)

    pv2.RHSWidget(pl=pl, origin=origin0, scale=scale, callback=update)
    pv2.RHSWidget(pl=pl, origin=origin1, scale=.3)

    pl.show()


def test_key_slider_widget():
    pl = pv2.Plotter()
    pv2.add_key_slider_widget(pl=pl, slider=None, callback=lambda x: print("A"), step=2.)
    pl.show()


if __name__ == "__main__":
    test_add_rhs_widget()

from unittest import TestCase
import pyvista as pv

from wzk.pv.widgets import *


class Test(TestCase):
    pass


def test_add_rhs_widget():
    origin0 = np.zeros(3)
    origin1 = np.ones(3)

    p = pv.Plotter()
    RHSWidget(p=p, origin=origin0, size=0.2)
    RHSWidget(p=p, origin=origin1, size=1.0)
    p.show()


def test_key_slider_widget():
    p = pv.Plotter()
    add_key_slider_widget(p=p, slider=None, callback=lambda x: print('A'), step=2.)
    p.show()

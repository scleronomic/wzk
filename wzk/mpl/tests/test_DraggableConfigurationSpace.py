from unittest import TestCase

from wzk.mpl.DraggableConfigurationSpace import *
verbose = 1


class TestDraggableCircle(TestCase):

    def test_DraggableConfigSpace(self):
        x = np.random.random((20, 3))
        limits = np.zeros((x.shape[-1], 2))
        limits[:, 1] = 1
        dcs = DraggableConfigSpace(x=x, limits=limits, circle_ratio=1 / 4, color='k')

        print(dcs.get_x().shape)

        dcs.set_x(x=np.zeros((20, 3)))
        dcs.set_x(x=np.ones((20, 3)))

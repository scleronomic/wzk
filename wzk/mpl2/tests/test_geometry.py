import unittest

from wzk.mpl2.geometry import *

from wzk.mpl2.figure import new_fig
from wzk.mpl2.axes import set_ax_limits
from wzk.mpl2 import DraggableCircle

from wzk.spatial.transform import sample_matrix
from wzk.spatial.transform_2d import trans_theta2frame


class Test(unittest.TestCase):

    def test_mpl_all(self):

        xy0 = np.array([0, 1])
        xy1 = np.array([1, 3])
        xy2 = np.array([1, 2])
        r0 = 1
        r1 = 2
        r2 = 4

        fig, ax = new_fig(aspect=1)
        ax.plot(*np.array([xy0, xy1, xy2]).T, 'xk')

        # Check switching of theta0 <-> theta1
        draw_arc(xy=xy2, radius=r2, ax=ax, theta0=3, theta1=-1, color='r')
        draw_arc(xy=xy2, radius=r2, ax=ax, theta0=-1, theta1=3, color='b')

        # Basic arc + intersection
        draw_arc(xy=xy0, radius=r0, ax=ax, alpha=0.7, color='k')
        draw_arc(xy=xy1, radius=r1, ax=ax, alpha=0.7, color='k')
        (int0, int1), _ = fill_circle_intersection(xy0=xy0, r0=r0, xy1=xy1, r1=r1, color='y')

        draw_rays(xy=xy1, radius0=r1*0.1, radius1=r1*0.9, theta0=int1[1], theta1=int1[2],
                  n=100, color='gray')

        self.assertTrue(True)

    def test_mpl_fill_circle_intersection(self):

        fig, ax = new_fig(aspect=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        r0 = 0.2
        r1 = 0.3
        xy0 = np.array([-r0/2, 0])
        xy1 = np.array([+r1/2, 0])

        circle0 = DraggableCircle(ax=ax, xy=tuple(xy0), radius=r0, alpha=0.5,
                                  facecolor='blue', edgecolor='k', zorder=30)
        circle1 = DraggableCircle(ax=ax, xy=tuple(xy1), radius=r1, alpha=0.5,
                                  facecolor='red', edgecolor='k', zorder=30)

        def update(*args):  # noqa
            fill_circle_intersection(xy0=circle0.get_xy_drag(), r0=r0,
                                     xy1=circle1.get_xy_drag(), r1=r1, color='y')

        circle0.add_callback(update)
        circle1.add_callback(update)

        self.assertTrue(True)

    def test_mpl_eye_pov(self):

        new_fig(aspect=1)
        for x in np.arange(5):
            for y in np.arange(5):
                eye_pov(xy=(x, y), angle=np.random.uniform(0, 2 * np.pi),
                        radius=0.4, arc=0.7, n_rays=3, ax=None, color='k', lw=2)

        self.assertTrue(True)

    def test_plot_coordinate_frame(self):

        # 2D
        fig, ax = new_fig(aspect=1, title='2D Coordinate Frames')

        h1 = plot_coordinate_frame(ax=ax, x=[1, 1], dcm=trans_theta2frame(theta=1)[:-1, :-1], color='bb')
        h2 = plot_coordinate_frame(ax=ax, dcm=trans_theta2frame(theta=2)[:-1, :-1], color='ry')  # noqa

        plot_coordinate_frame(h=h1, dcm=np.eye(3), x=np.ones(3) * 0.1)

        # 3D
        fig, ax = new_fig(aspect=1, title='3D Coordinate Frames')

        set_ax_limits(ax=ax, limits=np.array([[-1, 1],
                                              [-1, 1],
                                              [-1, 1]]), n_dim=3)

        dcm = sample_matrix()
        h1 = plot_coordinate_frame(ax=ax, dcm=dcm)
        h2 = plot_coordinate_frame(ax=ax, dcm=dcm)  # noqa

        plot_coordinate_frame(h=h1, dcm=np.eye(3), x=np.ones(3) * 0.1)

        self.assertTrue(True)


if __name__ == '__main__':
    test = Test()
    test.test_mpl_all()
    test.test_mpl_fill_circle_intersection()
    test.test_mpl_eye_pov()

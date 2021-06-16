from unittest import TestCase

from wzk.mpl.plotting import *


class Test(TestCase):

    def test_plot_projections_2d(self):
        n_dof = 4
        n = 100
        x = np.random.normal(size=(n, n_dof))
        limits = np.array([[-4, +4]] * n_dof)
        plot_projections_2d(x=x, dim_labels='xyzuvw', limits=limits, aspect=1)

        n_dof = 3
        n = 100
        x = np.random.normal(size=(n, n_dof))
        limits = np.array([[-4, +4]] * n_dof)
        fig, ax = new_fig(n_rows=n_dof, n_cols=1, aspect=1)
        plot_projections_2d(ax=ax, x=x, limits=limits, aspect=1)

        self.assertTrue(True)

    def test_imshow(self):
        arr = np.arange(45).reshape(5, 9)
        mask = arr % 2 == 0
        limits = np.array([[0, 5],
                           [0, 9]])

        arr2 = arr.copy()
        arr2[mask] = 0
        print(arr2)

        fig, ax = new_fig(title='upper, ij')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='upper', axis_order='ij')

        fig, ax = new_fig(title='upper, ji')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='upper', axis_order='ji')

        fig, ax = new_fig(title='lower, ij')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='lower', axis_order='ij')

        fig, ax = new_fig(title='lower, ji')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='lower', axis_order='ji')

        fig, ax = new_fig(title='lower, ji')
        h = imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='lower', axis_order='ji')
        imshow_update(h=h, img=arr, mask=arr % 2 == 1, cmap='Reds', axis_order='ji')

        fig, ax = new_fig(aspect=1)
        arr = np.arange(42).reshape(6, 7)
        imshow(ax=ax, img=arr, limits=None, cmap='Blues', mask=arr % 2 == 0, vmin=0, vmax=100)

        self.assertTrue(True)

    def test_grid_lines(self):
        fig, ax = new_fig()

        limits = np.array([[0, 4],
                           [0, 5]])
        set_ax_limits(ax=ax, limits=limits, n_dim=2)
        grid_lines(ax=ax, start=0.5, step=(0.2, 0.5), limits=limits, color='b', ls=':')

        self.assertTrue(True)

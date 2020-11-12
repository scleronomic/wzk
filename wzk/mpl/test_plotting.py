from unittest import TestCase

from wzk.mpl.figure import *
from wzk.mpl.plotting import *


class Test(TestCase):

    def test_plot_projections_2d():
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
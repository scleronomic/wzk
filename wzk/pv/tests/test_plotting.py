import numpy as np
from wzk.pv.plotting import pv


def test_plot_points():
    plotter = pv.Plotter()
    plotter.add_points(np.random.random((100, 3)), color='black')
    plotter.show()


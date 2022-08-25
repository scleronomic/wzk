import numpy as np
from wzk import pv2


def test_plot_points():
    pl = pv2.Plotter()
    pl.add_points(np.random.random((100, 3)), color='black')
    pl.show()


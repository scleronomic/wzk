from unittest import TestCase


from wzk.trajectory import *
from wzk.testing import compare_arrays


class Test(TestCase):
    def test_get_substeps(self):

        # A
        x = np.array([0, 2, 5]).reshape(-1, 1)
        # A1
        n = 2
        x_ss = get_substeps(x=x, n=n)
        x_true = np.array([0, 1, 2, 3.5, 5]).reshape(-1, 1)
        self.assertTrue(compare_arrays(x_ss, x_true))
        # A2
        n = 4
        x_true = np.array([0, 0.5, 1, 1.5, 2, 2.75, 3.5, 4.25, 5]).reshape(-1, 1)
        x_ss = get_substeps(x=x, n=n)
        self.assertTrue(compare_arrays(x_ss, x_true))

        # B
        q = np.array([[0, 1],
                      [1, 1],
                      [2, 2]])
        # B1
        n = 2
        x_ss = get_substeps(x=x, n=n)
        x_true = np.array([[0, 1],
                           [0.5, 1],
                           [1, 1],
                           [1.5, 1.5],
                           [2, 2]])
        self.assertTrue(compare_arrays(x_ss, x_true))
        # B2
        x_ss = get_substeps(x=x, n=0)
        self.assertTrue(compare_arrays(x_ss, q))
        # B3
        x_ss = get_substeps(x=x, n=1)
        self.assertTrue(compare_arrays(x_ss, q))

    def test_inner2full(self):

        def __assert(full, shape):
            self.assertTrue(np.all(full[..., 0, :] == 1))
            self.assertTrue(np.all(full[..., -1, :] == 2))
            self.assertTrue(full.shape == shape)

        inner = np.full((10, 5, 3), 0)
        start = np.full((1, 1, 3), 1)
        end = np.full((1, 1, 3), 2)
        __assert(full=inner2full(inner=inner, start=start, end=end),
                 shape=(10, 7, 3))

        inner = np.full((10, 5, 3), 0)
        start = np.full((1, 3), 1)
        end = np.full((1, 3), 2)
        __assert(full=inner2full(inner=inner, start=start, end=end),
                 shape=(10, 7, 3))

        inner = np.full((10, 5, 3), 0)
        start = np.full(3, 1)
        end = np.full(3, 2)
        __assert(full=inner2full(inner=inner, start=start, end=end),
                 shape=(10, 7, 3))

    def test_get_substeps_adjusted(self):
        x = np.array([0, 2, 10, 19])[:, np.newaxis]
        x2 = get_substeps_adjusted(x=x, n=20)
        true = np.arange(20)
        self.assertTrue(compare_arrays(x2, true))

        x = np.deg2rad(np.array([[-150], [+150], [90], [-20]]))
        x2 = get_substeps_adjusted(x=x, n=12, is_periodic=[True])
        x2 = np.rad2deg(x2)
        true = np.array([-150, -170, 170.,
                         150, 130, 110,
                         90, 68, 46, 24, 2,
                         -20])[:, np.newaxis]
        self.assertTrue(compare_arrays(x2, true))

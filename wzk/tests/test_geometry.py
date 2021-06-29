import unittest

from wzk.testing import compare_arrays
from wzk.geometry import *
from itertools import combinations


class Test(unittest.TestCase):

    def test_rotation_between_vectors(self):
        a, b = np.random.random((2, 3))
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        r_ab = rotation_between_vectors(a=a, b=b)
        r_ba = rotation_between_vectors(a=b, b=a)

        b1 = (r_ab @ a[:, np.newaxis])[:, 0]
        a1 = (r_ba @ b[:, np.newaxis])[:, 0]

        self.assertTrue(compare_arrays(a, a1))
        self.assertTrue(compare_arrays(b, b1))
        self.assertTrue(compare_arrays(r_ab, r_ba.T))

    def test_get_orthonormal(self):
        a = np.random.random(3)
        b = get_orthonormal(a)
        self.assertTrue(np.allclose(np.dot(a, b), 0))

    def speed_mink(self):
        from wzk import tic, toc
        n = 12
        m = 50
        lines = np.random.random((m, n, 2, 3))
        pairs = np.array(list(combinations(np.arange(n), 2)))

        tic()
        n = 40
        spheres = np.random.random((m, n, 3))
        pairs2 = np.array(list(combinations(np.arange(n), 2)))

        tic()
        for i in range(100):
            res = line_line_pairs(lines=lines, pairs=pairs)
        toc('mink')

        tic()
        for i in range(100):
            res = line_line_pairs(lines=lines, pairs=pairs)
        toc('d1234')

        self.assertTrue(True)

import unittest
from itertools import combinations
import numpy as np
from wzk import geometry, testing


class Test(unittest.TestCase):

    def test_rotation_between_vectors(self):
        a, b = np.random.random((2, 3))
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        r_ab = geometry.rotation_between_vectors(a=a, b=b)
        r_ba = geometry.rotation_between_vectors(a=b, b=a)

        b1 = (r_ab @ a[:, np.newaxis])[:, 0]
        a1 = (r_ba @ b[:, np.newaxis])[:, 0]

        self.assertTrue(testing.compare_arrays(a, a1))
        self.assertTrue(testing.compare_arrays(b, b1))
        self.assertTrue(testing.compare_arrays(r_ab, r_ba.T))

    def test_get_orthonormal(self):
        a = np.random.random(3)
        b = geometry.get_orthonormal(a)
        self.assertTrue(np.allclose(np.dot(a, b), 0))

    def speed_mink(self):
        from wzk import tic, toc
        n = 12
        m = 50
        lines = np.random.random((m, n, 2, 3))
        pairs = np.array(list(combinations(np.arange(n), 2)))

        tic()
        n = 40
        _ = np.random.random((m, n, 3))
        _ = np.array(list(combinations(np.arange(n), 2)))

        tic()
        for i in range(100):
            _ = geometry.line_line_pairs(lines=lines, pairs=pairs)
        toc("mink")

        tic()
        for i in range(100):
            _ = geometry.line_line_pairs(lines=lines, pairs=pairs)
        toc("d1234")

        self.assertTrue(True)

from unittest import TestCase
from wzk.training import *


class Test(TestCase):
    def test_train_test_split(self):

        def __assert(res, _s, _n):
            for x_train, x_test in zip(res):
                self.assertTrue(len(x_test) == _s)
                self.assertTrue(len(x_test) + len(x_train) == _n)

        n = 10
        split = 4
        a = np.ones((n, 1))
        b = np.ones((n, 2)) * 2
        c = np.ones((n, 3)) * 3
        train_test_tuple = train_test_split(a, b, c, split=split)
        __assert(train_test_tuple, _s=split, _n=n)

        n = 100
        split = 0.2
        a = np.ones((n, 10, 1))
        b = np.ones((n, 20, 2)) * 2
        c = np.ones((n, 30, 3)) * 3
        train_test_tuple = train_test_split(a, b, c, split=split)
        __assert(train_test_tuple, _s=split*n, _n=n)

        n = 1000
        split = -1
        a = np.ones((n, 10, 2))
        b = np.ones((n, 20, 2)) * 2
        c = np.ones((n, 30, 2)) * 3
        train_test_tuple = train_test_split(a, b, c, split=split)
        __assert(train_test_tuple, _s=n, _n=n)

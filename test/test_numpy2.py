from unittest import TestCase
from wzk.nump import *


class Test(TestCase):

    def test_DummyArray(self):
        arr = np.random.random((4, 4))
        idx = (1, 2, 3, 4)
        d = DummyArray(arr=arr, shape=(4, 4, 2, 2))
        self.assertTrue(np.allclose(arr, d[idx]))

        d = DummyArray(1, shape=(2, 2))
        self.assertTrue(np.allclose(1, d[1, :]))

    def test_initialize_array(self):

        shape = [4, (4,), (1, 2, 3, 4)]
        dtype = [float, int, bool]
        order = ['c', 'f']

        for s in shape:
            for d in dtype:
                for o in order:
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='zeros'),
                                    np.zeros(shape=s, dtype=d, order='o'))
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='ones'),
                                    np.ones(shape=s, dtype=d, order='o'))
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='empty'),
                                    np.empty(shape=s, dtype=d, order='o'))
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='random'),
                                    np.random.random(s).astype(dtype=d, order=o))

    def test_(self):

        self.assertEqual(np_isinstance(('this', 'that'), tuple))
        self.assertFalse(np_isinstance(4.4, int))
        self.assertTrue(np_isinstance(4.4, float))

        # else
        np_isinstance(np.ones(4, dtype=int), int)  # True
        np_isinstance(np.ones(4, dtype=int), float)  # False
        np_isinstance(np.full((4, 4), 'bert'), str)  # True

    def test_DummyArray(self):
        arr = np.random.random((4, 4))
        idx = (1, 2, 3, 4)
        d = DummyArray(arr=arr, shape=(4, 4, 2, 2))
        self.assertTrue(np.allclose(arr, d[idx]))

        d = DummyArray(1, shape=(2, 2))
        self.assertTrue(np.allclose(1, d[1, :]))

    def test_initialize_array(self):

        shape = [4, (4,), (1, 2, 3, 4)]
        dtype = [float, int, bool]
        order = ['c', 'f']

        for s in shape:
            for d in dtype:
                for o in order:
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='zeros'),
                                    np.zeros(shape=s, dtype=d, order='o'))
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='ones'),
                                    np.ones(shape=s, dtype=d, order='o'))
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='empty'),
                                    np.empty(shape=s, dtype=d, order='o'))
                    self.assertTrue(initialize_array(shape=s, dtype=d, order=o, mode='random'),
                                    np.random.random(s).astype(dtype=d, order=o))

    def test_(self):


        self.assertEqual(np_isinstance(('this', 'that'), tuple))
        self.assertFalse(np_isinstance(4.4, int))
        self.assertTrue(np_isinstance(4.4, float))

        # else
        np_isinstance(np.ones(4, dtype=int), int)  # True
        np_isinstance(np.ones(4, dtype=int), float)  # False
        np_isinstance(np.full((4, 4), 'bert'), str)  # True

    def test_insert(self):
        a = np.ones((4, 5, 3))
        val = 2

        insert(a=a, idx=(1, 2), axis=(0, 2), val=val)

        self.assertTrue(np.allclose(a[1, :, 2], val))

    def test_product2(self):

        sol = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]])

        it = product((range(5), range(3)))
        res = np.array([i for i in it]).T

        self.assertTrue(np.allclose(sol, res))

    def test_argmax(self):
        n = 100
        axis = (0, 2)
        size = (3, 4, 5, 6)

        a = np.random.randint(n, size=size)
        i = argmax(a, axis=axis)

        e = extract(a=a, axis=axis, idx=i, mode='orange')
        amax = np.max(a, axis=axis)
        self.assertTrue(np.allclose(amax, e))

    def test_argmin(self):
        n = 1000
        axis = (1, 3, 5)
        size = (3, 4, 5, 6, 7, 8, 9)

        a = np.random.randint(n, size=size)
        i = argmin(a, axis=axis)

        e = extract(a=a, axis=axis, idx=i, mode='orange')
        amin = np.min(a, axis=axis)
        self.assertTrue(np.allclose(amin, e))

    def test_safe2vectors(self):
        self.assertTrue(np.array_equal([np.array([1])], safe_scalar2array(1, shape=1, squeeze=False)))
        self.assertTrue(np.array_equal(np.array([1]), safe_scalar2array(1, shape=1, squeeze=True)))

        self.assertTrue(np.array_equal([np.array(['a', 'a', 'a'], dtype='<U1'),
                                        np.array(['b', 'b', 'b'], dtype='<U1'),
                                        np.array(['c', 'c', 'c'], dtype='<U1')], safe_scalar2array('a', 'b', 'c', shape=3)))

        self.assertTrue(np.array_equal([np.array([1, 1, 1]),
                                        np.array([None, None, None], dtype=object),
                                        np.array(['a', 'a', 'a'], dtype='<U1')], safe_scalar2array(1, None, 'a', shape=3)))

    def test_find_values(self):
        arr = np.array([3, 5, 5, 6, 7, 8, 8, 8, 10, 11, 1])
        values = [3, 5, 8]
        res = find_values(arr=arr, values=values)
        sol = np.array([True, True, True, False, False, True, True, True, False, False, False])

        self.assertTrue(np.array_equal(res, sol))




# if __name__ == '__main__':
#     t = Test()
#     t.test_insert()

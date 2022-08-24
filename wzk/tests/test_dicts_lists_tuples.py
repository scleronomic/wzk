import os

from unittest import TestCase
from wzk.dicts_lists_tuples import *
from wzk.files import mkdirs, rmdirs


directory = f"{os.path.split(__file__)[0]}/tmp"


class Test(TestCase):

    def test_totuple(self):
        self.assertTrue(totuple('aaa') == ('a', 'a', 'a'))
        self.assertTrue(totuple([1, 2, [3, 4]]) == (1, 2, (3, 4)))

    def test_tolist(self):
        self.assertTrue(tolist('aaa') == ['a', 'a', 'a'])
        self.assertTrue(tolist((1, 2, (3, 4))) == [1, 2, [3, 4]])

    def test_tuple_extract(self):
        self.assertTrue(tuple_extract(t=(1,), default=(0, 0), mode='default') == (1, 0))
        self.assertTrue(tuple_extract(t=(1,), default=(0, 0), mode='repeat') == (1, 1))
        self.assertTrue(tuple_extract(t=(1, (3, 4)), default=(0, 0, (0, 0)), mode='default') == (1, (3, 4), (0, 0)))

    def test_safe_squeeze(self):
        self.assertTrue(safe_squeeze(s=(1.1,)) == 1.1)
        self.assertRaises(AssertionError, safe_squeeze, (1, (3, 4)))

    def remove_nones(self):
        self.assertTrue(remove_nones([1, None, 2, None, None, 3, None, None]) == [1, 2, 3])
        self.assertTrue(remove_nones([1, None, 2, None, None, 'a', None, [None]]) == [1, 2, 'a', [None]])

    def test_atleast_list(self):
        self.assertTrue(atleast_list((1, 2, 3), convert=True) == [1, 2, 3])
        self.assertTrue(atleast_list((1, 2, 3), convert=False) == [(1, 2, 3)])
        self.assertTrue(atleast_list((1, 2, (3, 4)), convert=True) == [1, 2, [3, 4]])
        self.assertTrue(atleast_list(np.array((1, 2, 3)), convert=True) == [1, 2, 3])

    def test_atleast_tuple(self):
        self.assertTrue(atleast_tuple([1, 2, 3], convert=True) == (1, 2, 3))
        self.assertTrue(atleast_tuple([1, 2, 3], convert=False) == ([1, 2, 3],))
        self.assertTrue(atleast_tuple([1, 2, [3, 4]], convert=True) == (1, 2, (3, 4)))
        self.assertTrue(atleast_tuple(np.array((1, 2, 3)), convert=True) == (1, 2, 3))

    def test_dict2json(self):
        mkdirs(directory)

        dummy_file = f"{directory}/dummy_dict.json"
        dict_1 = {'a': 1,
                  'b': {'aa': 2,
                        'bb': 3,
                        'cc': {'aaa': 4,
                               'bbb': 5},
                        'dd': 6},
                  'c': 7}

        write_dict2json(file=dummy_file, d=dict_1, indent=4)
        dict_2 = read_json2dict(dummy_file)
        self.assertTrue(dict_1 == dict_2)

        rmdirs(directory)

    def test_flatten(self):
        list_nested = [(4, (1, (2, (1, [13, 13]))))]
        list_flat = [[4, (1, (2, (1, [13, 13])))],
                     [4, 1, (2, (1, [13, 13]))],
                     [4, 1, 2, (1, [13, 13])],
                     [4, 1, 2, 1, [13, 13]],
                     [4, 1, 2, 1, 13, 13]]
        for i in range(5):
            self.assertTrue(flatten(list_nested, max_depth=i+1) == list_flat[i])

    def test_element_at_depth(self):
        test_list = [1, 2, 3, [[4], [[5]], [6, 7]], [[8, 9], [10, 11, [12, [13, 13]]]], 14, [15], [16, 17]]

        res = [((3,), [[4], [[5]], [6, 7]]), ((4,), [[8, 9], [10, 11, [12, [13, 13]]]]), ((6,), [15]), ((7,), [16, 17])]
        a = element_at_depth(test_list, d=1, with_index=True)

        self.assertTrue(res == a)

    def test_repeat_dict(self):
        d = {'a': 'red',
             'b': (5, 4, 3),
             'c': True}

        res = {0: {'a': 'red', 'b': 5, 'c': True},
               1: {'a': 'red', 'b': 4, 'c': True},
               2: {'a': 'red', 'b': 3, 'c': True}}
        dr = repeat_dict(d=d, n=3)
        self.assertTrue(res == dr)

    def test_weave_lists(self):
        x = [[1, 2, 3], ['a', 'b', 'c'], [11, 12, 13, 14]]
        res = [1, 'a', 11, 2, 'b', 12, 3, 'c', 13]

        a = weave_lists(*x)
        self.assertTrue(a == res)

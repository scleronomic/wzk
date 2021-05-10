from unittest import TestCase
from wzk.dicts_lists_tuples import *
from wzk import safe_remove


class Test(TestCase):

    def test_dict2json(self):
        dummy_file = 'dummy_dict.json'
        dict_1 = {'a': 1,
                  'b': {'aa': 2,
                        'bb': 3,
                        'cc': {'aaa': 4,
                               'bbb': 5},
                        'dd': 6},
                  'c': 7}

        write_dict2json(file=dummy_file, d=dict_1, indent=4)
        dict_2 = read_json2dict(dummy_file)
        safe_remove(dummy_file)

        self.assertTrue(dict_1 == dict_2)

    def test_flatten(self):
        list_nested = [(4, (1, (2, (1, [13, 13]))))]
        list_flattend = [[4, (1, (2, (1, [13, 13])))],
                         [4, 1, (2, (1, [13, 13]))],
                         [4, 1, 2, (1, [13, 13])],
                         [4, 1, 2, 1, [13, 13]],
                         [4, 1, 2, 1, 13, 13]]
        for i in range(5):
            self.assertTrue(flatten(list_flattend[i] == list_nested, max_depth=i+1))

    def test_element_at_depth(self):
        test_list = [1, 2, 3, [[4], [[5]], [6, 7]], [[8, 9], [10, 11, [12, [13, 13]]]], 14, [15], [16, 17]]

        res = [((3,), [[4], [[5]], [6, 7]]), ((4,), [[8, 9], [10, 11, [12, [13, 13]]]]), ((6,), [15]), ((7,), [16, 17])]
        a = element_at_depth(test_list, depth=1, with_index=True)

        self.assertTrue(res == a)

    def test_repeat_dict(self):
        d = {'a': 'red',
             'b': (5, 4, 3),
             'c': True}

        res = {0: {'a': 'red', 'b': 5, 'c': True}, 1: {'a': 'red', 'b': 4, 'c': True}, 2: {'a': 'red', 'b': 3, 'c': True}}
        dr = repeat_dict(d=d, n=3)
        self.assertTrue(res == dr)

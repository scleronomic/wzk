from unittest import TestCase
import numpy as np


class Test(TestCase):
    def test_print_stats(self):
        a = np.random.random(100) * 1
        b = np.random.random(100) * 2
        c = np.random.random(100) * 3
        print_stats(a, b, c, names=('A', 'BB', 'CCC'))

        # self.fail()

    def test_print_table():
        print_table(rows=['A', 'BB', 'CCC'], columns=['Some', 'Random', 'Words', 'Foo'],
                    data=np.arange(12).reshape(3, 4), min_cell_size=10, cell_format='.4f')


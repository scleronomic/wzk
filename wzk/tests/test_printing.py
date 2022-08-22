from unittest import TestCase

from wzk.printing import *


class Test(TestCase):
    def test_print_stats(self):
        a = np.random.random(100) * 1
        b = np.random.random(100) * 2
        c = np.random.random(100) * 3
        print_stats(a, b, c, names=('A', 'BB', 'CCC'))

        # self.fail()

    def test_print_table(self):
        print_table(rows=['A', 'BB', 'CCC'], columns=['Some', 'Random', 'Words', 'Foo'],
                    data=np.arange(12).reshape(3, 4), min_cell_size=10, cell_format='.4f')

    def test_print2(self):
        print2("aaa", 1, 2, verbose=(1, 0))
        print2(dict(b=1, bb=2), 11, 22, verbose=(1, 1))
        print2("ccc", [3, "cc", 333], 33, verbose=(1, 2), sep='---')
        print2("nice", "a", "staircase", verbose=(1, 1), sep='    ')
        print2("back", "to", "level", "zero", verbose=(1, 0), sep='::')

    def test_print_array_3d(self):
        array_3d = np.arange(4 * 5 * 6).reshape((4, 5, 6))
        print_array_3d(array_3d)

    def test_color(self):
        print("normal")
        print(color_text(s="normal2", color='k', background='w'))
        print(color_text(s="red", color='red', background='w'))
        print(color_text(s="red", color='red', background='k'))
        print(color_text(s="normal2", color='k', background='w'))
        print("normal")

        for c in ['w', 'r', 'g', 'y', 'b', 'm', 'c', 'l', 'k', 'w']:
            print(color_text(s=c, color=c, background='w'))

        for c in ['w', 'r', 'g', 'y', 'b', 'm', 'c', 'l', 'k', 'w']:
            print(color_text(s=c, color=c, background=c))

        for c in ['w', 'r', 'g', 'y', 'b', 'm', 'c', 'l', 'k', 'w']:
            print(color_text(s=c, color=c, background='k'))

    def test_save_string_concatenate(self):
        s = 'test - full'
        prefix = 'hello'
        suffix = 'bye'
        delimiter = ' | '
        assert pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                              delimiter=delimiter) == f"{prefix}{delimiter}{s}{delimiter}{suffix}"

        s = ''
        prefix = 'hello'
        suffix = 'bye'
        delimiter = ' / '
        assert pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                              delimiter=delimiter) == f"{prefix}{delimiter}{suffix}"

        s = 'test - only prefix'
        prefix = 'hello'
        suffix = ''
        delimiter = ' : '
        assert pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                              delimiter=delimiter) == f"{prefix}{delimiter}{s}"

        s = 'test - only suffix'
        prefix = ''
        suffix = 'bye'
        delimiter = ' : '
        assert pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                              delimiter=delimiter) == f"{s}{delimiter}{suffix}"

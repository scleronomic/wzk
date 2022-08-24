from unittest import TestCase

from wzk.mpl.figure import *

from wzk.files import rmdir
_dir = os.path.abspath(os.path.dirname(__file__)) + '/'
_file = 'temp__test_mpl_latex'


class Test(TestCase):
    def test_save_fig(self):
        file = f"{_dir}{_file}"
        fig, ax = new_fig()
        save_fig(file=file, fig=fig, formats=('png',), view=False)
        rmdir(file)
        self.assertTrue(True)

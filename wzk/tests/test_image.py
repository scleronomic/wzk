from unittest import TestCase

from wzk.image import *


class Test(TestCase):

    def test_block_collage(self):
        img = np.ones((4, 3, 2, 2))
        block_collage(img_arr=img, inner_border=(1, 1), outer_border=2, fill_boarder=2)
        self.assertTrue(False)

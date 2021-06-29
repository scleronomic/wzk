from unittest import TestCase

from wzk.image import *


class Test(TestCase):

    def test_img2compressed(self):
        size = (64, 35)
        n_dim = 2
        img = np.random.random(size)
        img_cmp = img2compressed(img=img, n_dim=n_dim, )
        img2 = compressed2img(img_cmp=img_cmp, n_voxels=size, n_dim=n_dim, dtype=float)

        self.assertTrue(np.allclose(img, img2))



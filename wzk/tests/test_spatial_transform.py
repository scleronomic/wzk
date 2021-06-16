from wzk.spatial.transform import *
from unittest import TestCase
# from wzk.testing import compare_arrays


class Test(TestCase):
    def test_a(self):
        from wzk.spatial.difference import frame_difference
        a = sample_frames(x_low=np.zeros(3), x_high=np.ones(3))
        b = sample_frames(x_low=np.zeros(3), x_high=np.ones(3))

        a[:3, :3] = sample_matrix_noise(shape=1, scale=1e-8)
        b[:3, :3] = sample_matrix_noise(shape=1, scale=1e-8)

        c = invert(a) @ b
        c_rc = frame2rotvec(c)
        print('rad c ', np.linalg.norm(c_rc))

        ab_trans, ab_rot = frame_difference(a, b)
        print('rad ab', ab_rot)
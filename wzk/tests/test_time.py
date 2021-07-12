from unittest import TestCase

from wzk.time import *


class Test(TestCase):
    def test_tictoc(self):

        tic()  # 1
        sleep(0.1)
        tic()  # 2
        sleep(0.2)
        toc()  # 2
        sleep(0.2)
        tic()  # 3
        sleep(0.2)
        toc()  # 3
        sleep(0.1)
        t = toc()  # 1
        self.assertTrue(t > 0.8)
        self.assertTrue(t < 0.9)

    def test_get_timestamp(self):
        s = get_timestamp(year=True, month=True, day=True, hour=True, minute=True, second=True, millisecond=True,
                          date_separator='.', time_separator=':', date_time_separator=' ')
        print(s)
        s = get_timestamp(year=True, month=False, day=False, hour=False, minute=False, second=False, millisecond=False,
                          date_separator='.', time_separator=':', date_time_separator=' ')
        print(s)
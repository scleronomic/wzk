from unittest import TestCase

from wzk.strings import *


class Test(TestCase):

    def test_split_insert_joint(self):
        s0 = 'abcdx1234xikjx789xxtest'
        s1_true = 'YabcdxY1234xYikjxY789xYxYtest'
        s1 = split_insert_join(s=s0, split='x', insert_pre='Y')
        self.assertTrue(s1 == s1_true)

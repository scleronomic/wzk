from unittest import TestCase
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt

from wzk.mpl2.move_figure import *
from wzk.mpl2.backend import plt


class Test(TestCase):

    def test_move_fig(self):
        fig, ax = plt.subplots()
        move_fig(fig=fig, position='top right')

        fig, ax = plt.subplots()
        move_fig(fig=fig, position='top left')

        fig, ax = plt.subplots()
        move_fig(fig=fig, position='bottom right')

        fig, ax = plt.subplots()
        move_fig(fig=fig, position='bottom left')
        plt.pause(0.1)
        plt.close('all')

        for i in range(1, 10):
            fig, ax = plt.subplots()
            move_fig(fig=fig, position=(3, 3, i))

        plt.pause(0.1)
        plt.close('all')

        self.assertTrue(True)

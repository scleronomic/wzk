from unittest import TestCase

from wzk.mpl.figure import new_fig, plt
from wzk.mpl.Patches2 import *
from wzk.mpl import size_units2points


class Test(TestCase):

    def test_transform(self):
        height = 0.1
        width = 1
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim([-width * 1.2, width * 1.2])
        ax.set_ylim([-width * 1.2, width * 1.2])

        deg = 180
        rec = plt.Rectangle((0, 0), width=width, height=height, alpha=0.8,
                            transform=get_aff_trafo(xy1=(0.5, 0.5), theta=90, por=(0.5, 0.05), ax=None))
        ax.add_patch(rec)

        do_aff_trafo(patch=rec, xy=(0, 0), por=(0, 0.1), theta=90)
        plt.pause(1)
        do_aff_trafo(patch=rec, xy=(0, 1), por=(0.5, 0.1), theta=45)
        plt.pause(1)
        do_aff_trafo(patch=rec, xy=(0, 0), por=(0.5, 0.05), theta=135)
        self.assertTrue(True)

    def test_RelativeArrowPatch(self):

        fig, ax = new_fig(aspect=1, scale=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        arrow1 = RelativeFancyArrow(0.5, 0.5, 0.0, 0.5, width=1, head_length=0.5, head_width=2)
        arrow2 = RelativeFancyArrow(0.5, 0.5, 0.5, 0.0, width=1, head_length=0.5, head_width=2)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)

        ax.plot(0.5, 0.5, marker='o', markersize=size_units2points(size=0.5, ax=ax), alpha=0.5, zorder=100)
        self.assertTrue(True)

    def test_AbsoluteFancyBboxPatch(self):
        from wzk.mpl import new_fig, patches

        fig, ax = new_fig(aspect=1)
        ax.add_patch(FancyBbox(xy=(0.1, 0.1), boxstyle='Round4',
                               height=0.5, width=0.5, pad=0.1, corner_size=0))



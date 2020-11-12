from unittest import TestCase

from wzk.mpl.figure import *


class Test(TestCase):

    def __test_transform_tick_labels_x(self):
        fig, ax = new_fig()
        transform_tick_labels(ax=ax, xt=0, yt=-0.1, axis='x')
        transform_tick_labels(ax=ax, xt=0.4, yt=0, rotation=45, axis='y')
        fig, ax = new_fig()
        transform_tick_labels(ax=ax, xt=0, yt=0, rotation=90, axis='both', ha='center', va='center')

        self.assertTrue(True)

    def __test_grid_lines(self):
        fig, ax = new_fig()

        limits = np.array([[0, 4],
                           [0, 5]])
        set_ax_limits(ax=ax, limits=limits, n_dim=2)
        grid_lines(ax=ax, start=0.5, step=(0.2, 0.5), limits=limits, color='b', ls=':')

        self.assertTrue(True)

    def test_annotate_arrow(self):
        fig, ax = new_fig()
        from_xy = np.vstack([np.ones(5), np.arange(5)]).T / 5
        to_xy = np.vstack([np.arange(5)+0.1, np.arange(1, 6)]).T / 5
        annotate_arrow(ax=ax, xy0=from_xy, xy1=to_xy, offset=0, color='r')
        annotate_arrow(ax=ax, xy0=from_xy, xy1=to_xy, offset=0.04, color='k')
        annotate_arrow(ax=ax, xy0=from_xy, xy1=to_xy, offset=0.08, color='b')
        annotate_arrow(ax=ax, xy0=(0.2, 0.2), xy1=(0.4, 0.4), offset=0.08, color='k')
        
        # close_all()
        self.assertTrue(True)

    def test_imshow(self):
        arr = np.arange(45).reshape(5, 9)
        mask = arr % 2 == 0
        limits = np.array([[0, 5],
                           [0, 9]])
    
        arr2 = arr.copy()
        arr2[mask] = 0
        print(arr2)
    
        fig, ax = new_fig(title='upper, ij')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='upper', axis_order='ij')
        
        fig, ax = new_fig(title='upper, ji')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='upper', axis_order='ji')
        
        fig, ax = new_fig(title='lower, ij')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='lower', axis_order='ij')
        
        fig, ax = new_fig(title='lower, ji')
        imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='lower', axis_order='ji')
    
        fig, ax = new_fig(title='lower, ji')
        h = imshow(ax=ax, img=arr, limits=limits, cmap='Blues', mask=mask, origin='lower', axis_order='ji')
        imshow_update(h=h, img=arr, mask=arr % 2 == 1, cmap='Reds', axis_order='ji')
        
        fig, ax = new_fig(aspect=1)
        arr = np.arange(42).reshape(6, 7)
        imshow(ax=ax, img=arr, limits=None, cmap='Blues', mask=arr % 2 == 0, vmin=0, vmax=100)
        
        self.assertTrue(True)

    def test_make_legend_arrow_wrapper(self):

        fig, ax = new_fig(aspect=1)
        arrow_a = ax.arrow(0, 0, np.cos(0.3), np.sin(0.3), head_width=0.1, color='r', label='A')
        arrow_b = ax.arrow(0, 0, np.cos(0.5), np.sin(0.5), head_width=0.05, color='b', label='B')
        arrow_c = ax.arrow(0, 0, np.cos(0.9), np.sin(0.9), head_width=0.05, color='matrix', label='C')
        ax.legend([arrow_a, arrow_b, arrow_c], ['My label - A', 'Dummy - B', 'Another One - C'],
                  handlelength=1, borderpad=1.2, labelspacing=1.2,
                  handler_map={patches.FancyArrow: make_legend_arrow_wrapper(theta=0.9,
                                                                             label2theta_dict={'A': 0.3,
                                                                                               'B': 0.5})}
                  )

        self.assertTrue(True)

    def test_change_tick_appearance(self):
        fig, ax = new_fig(aspect=1)

        n = 10
        ax.plot(range(n))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        change_tick_appearance(ax=ax, position='left', i=5, size=10, color='r')
        change_tick_appearance(ax=ax, position='bottom', i=3, size=20, color='b')
        change_tick_appearance(ax=ax, position='bottom', i=7, size=None, color='y')

        self.assertTrue(True)

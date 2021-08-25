import os
import shutil
import numpy as np

from wzk.mpl.backend import plt
from wzk.mpl.move_figure import move_fig

from wzk.dicts_lists_tuples import atleast_tuple
from wzk.files import start_open, safe_create_dir, copy2clipboard
from wzk.math2 import get_mean_divisor_pair, golden_ratio
from wzk.printing import print_progress
from wzk.strings import uuid4


_width_1c_ieee = [3 + 1 / 2, (3 + 1 / 2) / golden_ratio]
_width_2c_ieee = [7 + 1 / 16, (7 + 1 / 12) / golden_ratio]


def figsize_wrapper(width, height=None, height_ratio=1/golden_ratio):
    # https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/eic-guide.pdf
    if isinstance(width, str):
        if width.lower() == 'ieee1c':
            width = _width_1c_ieee[0]
        elif width.lower() == 'ieee2c':
            width = _width_2c_ieee[0]
        else:
            raise ValueError

    elif isinstance(width, (float, int)):
        pass
    else:
        raise ValueError

    height = width * height_ratio if height is None else height

    return width, height


def new_fig(*, width=_width_2c_ieee[0], height=None, h_ratio=1 / golden_ratio,
            n_dim=2,
            n_rows=1, n_cols=1,
            share_x='none', share_y='none',  # : bool or {'none', 'all', 'row', 'col'},
            aspect='auto',
            title=None,
            position=None, monitor=-1,
            **kwargs):

    fig = plt.figure(figsize=figsize_wrapper(width=width, height=height, height_ratio=h_ratio), **kwargs)

    if n_dim == 2:
        ax = fig.subplots(nrows=n_rows, ncols=n_cols, sharex=share_x, sharey=share_y)

        if isinstance(ax, np.ndarray):
            for i in np.ndindex(*np.shape(ax)):
                ax[i].set_aspect(aspect)  # Not implemented for 3D
        else:
            ax.set_aspect(aspect)

    else:
        import mpl_toolkits.mplot3d.art3d as art3d  # noqa
        ax = plt.axes(projection='3d')

    if title is not None:
        fig.suptitle(title)

    move_fig(fig=fig, position=position, monitor=monitor)
    return fig, ax


def save_fig(filename=None, fig=None, formats=('png',),
             dpi=300, bbox='tight', pad=0.1,
             save=True, replace=True, view=False, copy2cb=False,
             verbose=1, **kwargs):
    """
    Adaption of the matplotlib 'savefig' function with some added convenience.
    bbox = tight / standard (standard does not crop but saves the whole figure)
    pad: padding applied to the thigh bounding box in inches
    """

    if not save:
        return

    if fig is None:
        fig = plt.gcf()

    if filename is None:
        filename = get_fig_suptitle(fig=fig)

    dir_name = os.path.dirname(filename)
    if dir_name != '':
        safe_create_dir(directory=dir_name)

    formats = atleast_tuple(formats, convert=False)
    for f in formats:
        file = f'{filename}.{f}'

        if replace or not os.path.isfile(path=file):
            fig.savefig(file, format=f, bbox_inches=bbox, pad_inches=pad,  dpi=dpi, **kwargs)
            if verbose >= 1:
                print(f'{file} saved')
        else:
            print(f'{file} already exists')

    if view:
        start_open(file=f"{filename}.{formats[0]}")

    if copy2cb:
        copy2clipboard(file=f"{filename}.{formats[0]}")


def save_ani(filename, fig, ani, n, dpi=300, bbox=None, fps=30):
    dir_temp = os.path.split(filename)[0]
    if dir_temp == '':
        dir_temp = os.getcwd()
        filename = dir_temp + '/' + filename
    dir_temp += '/' + uuid4()

    if isinstance(n, int):
        n = np.arange(n)

    for nn in n:
        print_progress(i=nn, n=n[-1]+1)
        ani(nn)
        save_fig(filename="{}/frame{:0>6}".format(dir_temp, nn), fig=fig, formats='png', dpi=dpi, bbox=bbox,
                 verbose=0)

    os.system(f'ffmpeg -r {fps} -i {dir_temp}/'
              f'frame%06d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" {filename}.mp4')
    shutil.rmtree(dir_temp)


def save_all(directory=None, close=False, **kwargs):
    if directory is None:
        directory = input('Directory to which the figures should be saved:')

    fig_nums = plt.get_fignums()
    for n in fig_nums:
        fig = plt.figure(num=n)
        title = get_fig_suptitle(fig=fig)
        title = "" if title is None else title
        save_fig(filename=f"{directory}N{n}_{title}", fig=fig, **kwargs)

    if close:
        close_all()


# noinspection PyProtectedMember
def get_fig_suptitle(fig):
    try:
        return fig._suptitle._text
    except AttributeError:
        return ''


def close_all():
    plt.close('all')


def subplot_grid(n, squeeze=False, **kwargs):
    n_rows, n_cols = get_mean_divisor_pair(n)

    if n >= 7 and n_rows == 1:
        n_rows, n_cols = get_mean_divisor_pair(n+1)

    _, ax = new_fig(n_rows=n_rows, n_cols=n_cols, **kwargs)

    if not squeeze:
        ax = np.atleast_2d(ax)
    return ax


def test_pdf2latex():
    fig, ax = new_fig(scale=0.5)
    ax.plot(np.random.random(20))
    ax.set_xlabel('Magnetization')
    save_fig(filename='/Users/jote/Documents/Vorlagen/LaTeX Vorlagen/IEEE-open-journal-template/aaa', fig=fig,
             formats='pdf')

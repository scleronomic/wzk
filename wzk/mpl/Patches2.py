import numpy as np
from matplotlib import patches, transforms, pyplot, path

from wzk.geometry import projection_point_line


class RelativeFancyArrow(patches.FancyArrow):
    def __init__(self, x, y, dx, dy, width=0.2, length_includes_head=True, head_width=0.3, head_length=0.4,
                 shape='full', overhang=0, head_starts_at_zero=False, **kwargs):
        length = np.hypot(dx, dy)
        super().__init__(x, y, dx, dy, width=width*length, length_includes_head=length_includes_head*length,
                         head_width=head_width*length, head_length=head_length*length,
                         shape=shape, overhang=overhang, head_starts_at_zero=head_starts_at_zero, **kwargs)


class FancyArrowX2(patches.FancyArrow):
    def __init__(self, xy0, xy1, offset0=0.0, offset1=0.0,
                 width=0.2, head_width=0.3, head_length=0.4, overhang=0,
                 shape='full',  length_includes_head=True, head_starts_at_zero=False, **kwargs):

        xy0, xy1 = np.array(xy0), np.array(xy1)
        dxy = xy1 - xy0
        dxy /= np.linalg.norm(dxy, keepdims=True)

        xy0 += offset0 * dxy

        dxy = xy1 - xy0
        dxy -= dxy / np.linalg.norm(dxy) * offset1

        super().__init__(*xy0, *dxy, width=width, length_includes_head=length_includes_head,
                         head_width=head_width, head_length=head_length,
                         shape=shape, overhang=overhang, head_starts_at_zero=head_starts_at_zero, **kwargs)


class FancyBbox(patches.FancyBboxPatch):
    def __init__(self, xy, width, height, boxstyle='Round', pad=0.3, corner_size=None, **kwargs):
        if boxstyle in ['Roundtooth', 'Sawtooth']:
            bs = patches.BoxStyle(boxstyle, pad=pad, tooth_size=corner_size)
        elif boxstyle in ['Round', 'Round4']:
            bs = patches.BoxStyle(boxstyle, pad=pad, rounding_size=corner_size)
        else:
            bs = patches.BoxStyle(boxstyle, pad=pad)

        super().__init__(xy=(xy[0]+pad, xy[1]+pad), width=width - 2*pad, height=height - 2*pad, boxstyle=bs, **kwargs)


class RoundedPolygon(patches.PathPatch):
    def __init__(self, xy, pad, **kwargs):
        p = path.Path(*self.round(xy=xy, pad=pad))
        super().__init__(path=p, **kwargs)

    def round(self, xy, pad):
        n = len(xy)

        for i in range(0, n):

            x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])

            d01, d12 = x1 - x0, x2 - x1
            d01, d12 = d01 / np.linalg.norm(d01), d12 / np.linalg.norm(d12)

            x00 = x0 + pad * d01
            x01 = x1 - pad * d01
            x10 = x1 + pad * d12
            x11 = x2 - pad * d12

            if i == 0:
                verts = [x00, x01, x1, x10]
            else:
                verts += [x01, x1, x10]
        codes = [path.Path.MOVETO] + n*[path.Path.LINETO, path.Path.CURVE3, path.Path.CURVE3]

        return np.atleast_1d(verts, codes)


def CurlyBrace(x0, x1, x2, curliness=1/np.e, return_verts=False, **patch_kw):
    """
    Create a matplotlib patch corresponding to a curly brace (i.e. this thing: {")
    Parameters
    ----------
    x : float
     x position of left edge of patch
    y : float
     y position of bottom edge of patch
    width : float
     horizontal span of patch
    height : float
     vertical span of patch
    curliness : float
     positive value indicating extent of curliness; default (1/e) tends to look nice
    pointing : str
     direction in which the curly brace points (currently supports 'left' and 'right')
    **patch_kw : any keyword args accepted by matplotlib's Patch
    Returns
    -------
    matplotlib PathPatch corresponding to curly brace

    Notes
    -----
    It is useful to supply the `transform` parameter to specify the coordinate system for the Patch.
    To add to Axes `ax`:
    cb = CurlyBrace(x, y)
    ax.add_artist(cb)
    This has been written as a function that returns a Patch because I saw no use in making it a class, though one could extend matplotlib's Patch as an alternate implementation.

    Thanks to:
    https://graphicdesign.stackexchange.com/questions/86334/inkscape-easy-way-to-create-curly-brace-bracket
    http://www.inkscapeforum.com/viewtopic.php?t=11228
    https://css-tricks.com/svg-path-syntax-illustrated-guide/
    https://matplotlib.org/users/path_tutorial.html
    Ben Deverett, 2018.
    Examples
    --------
    >>>from curly_brace_patch import CurlyBrace
    >>>import matplotlib.pyplot as pl
    >>>fig,ax = pl.subplots()
    >>>brace = CurlyBrace(x=.4, y=.2, width=.2, height=.6, pointing='right', transform=ax.transAxes, color='magenta')
    >>>ax.add_artist(brace)
    # https://github.com/bensondaled/curly_brace/blob/master/curly_brace_patch.py
    """
    x0, x1, x2,  = np.atleast_1d(x0, x1, x2)

    x0_p = projection_point_line(x0=x0, x1=x1, x2=x2)
    x10_p = x1 - x0_p
    x20_p = x2 - x0_p

    if np.linalg.norm(x10_p) < np.linalg.norm(x20_p):
        curliness_v = x10_p * curliness
    else:
        curliness_v = x20_p * curliness

    verts = np.stack([x1,                   #
                      x0 + x10_p,
                      x0_p + curliness_v,
                      x0,                   #
                      x0_p - curliness_v,
                      x0 + x20_p,
                      x2])                  #

    codes = [patches.Path.MOVETO] + 6 * [patches.Path.CURVE4]
    path = patches.Path(verts, codes)

    patch_kw['edgecolor'] = patch_kw.pop('color', 'black')

    pp = patches.PathPatch(path, facecolor='none', **patch_kw)
    if return_verts:
        return pp, verts
    else:
        return pp


def test_curly_brace():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.axis('off')

    for i, (w, h, c) in enumerate(zip(
            np.linspace(.1, .18, 4),
            np.linspace(.95, .5, 4),
            np.linspace(.1, .5, 4))):
        x = i * .1
        lw = 3 * i + 1
        col = plt.cm.plasma(i / 8)
        brace = CurlyBrace(x0=(x+0.05, h/2), x1=(x, 0.1) , x2=(x, 0.1+h), lw=lw,
                           curliness=c, color=col)
        ax.add_artist(brace)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_aspect(1)
    for i in range(10):
        x = np.random.random((3, 2))
        x = np.sort(x, axis=0)
        brace = CurlyBrace(x[1], x[0], x[2])
        ax.add_artist(brace)

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    brace, verts = CurlyBrace(x1=(1, 1), x2=(4, 5), x0=(2.5, 1.5), color='b', return_verts =True)
    ax.add_artist(brace)
    ax.plot(*verts.T, ls='', marker='o', alpha=0.5)
    ax.plot(*verts[[0, 1, 5, 6, 0]].T)

    text = ['a', 'ac0', 'ac1', 'c', 'cb0', 'cb1', 'b']
    for i, v in enumerate(verts):
        ax.annotate(xy=v, s=str(i) + '\n' + text[i], ha='center', va='center', zorder=100)


# Transformations
def do_aff_trafo(patch, theta, xy=None, por=(0, 0)):
    if xy is not None:
        patch.set_xy(xy=xy)
    patch.set_transform(get_aff_trafo(theta=theta, por=por, patch=patch))


def get_aff_trafo(xy0=None, xy1=None, theta=0, por=(0, 0), ax=None, patch=None):
    """

    :param xy0: current position of the object, if not provided patch.get_xy() is used
    :param xy1: desired position of the object
    :param theta: rotation in degrees
    :param por: point of rotation relative to the objects coordinates
    :param ax:
    :param patch:
    :return:
    """

    if xy0 is None:
        if patch is None:
            xy0 = (0, 0)
        else:
            xy0 = patch.get_xy()

    if xy1 is None:
        xy1 = xy0

    if ax is None:
        if patch is None:
            ax = pyplot.gca()
        else:
            ax = patch.axes

    return (transforms.Affine2D().translate(-xy0[0]-por[0], -xy0[1]-por[1])
                                 .rotate_deg_around(0, 0, theta)
                                 .translate(xy1[0], xy1[1]) + ax.transData)



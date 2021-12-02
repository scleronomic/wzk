import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

from wzk.mpl.bool_image_boundaries import get_combined_edges

from wzk.numpy2 import limits2cell_size, grid_i2x, grid_x2i  # noqa


def plot_img_patch(img, limits,
                   ax=None, **kwargs):
    """
    Plot an image as a Collection of square Rectangle Patches.
    Draw all True / nonzero pixels.
    """

    voxel_size = limits2cell_size(shape=img.shape, limits=limits)

    ij = np.array(np.nonzero(img)).T
    xy = grid_i2x(i=ij, limits=limits, shape=img.shape, mode='b')

    pc = PatchCollection([Rectangle((x, y), width=voxel_size, height=voxel_size,
                                    fill=False, snap=True) for (x, y) in xy], **kwargs)
    ax.add_collection(pc)
    return pc


def plot_img_outlines(img, limits,
                      ax, **kwargs):
    """
    Plot the image by drawing the outlines of the areas where the values are True.
    """

    combined_edges = get_combined_edges(img)
    combined_edges = [grid_i2x(i=ce, shape=img.shape, limits=limits, mode='b')
                      for ce in combined_edges]

    lc = LineCollection(combined_edges, **kwargs)
    ax.add_collection(lc)
    return lc


def __img_none_limits(limits=None, img=None):
    if limits is None and img is not None:
        limits = np.zeros((img.ndim, 2))
        limits[:, 1] = img.shape

    return limits


def plot_img_patch_w_outlines(ax, img, limits,
                              color=None, edgecolor='k', hatchcolor='k', facecolor='None',
                              hatch='xx',
                              lw=2, alpha_outline=1, alpha_patch=1.,
                              **kwargs):
    if img is None:
        return None

    if color is not None:
        facecolor = color
        edgecolor = color

    limits = __img_none_limits(limits=limits, img=img)

    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])

    plot_img_outlines(img=img, limits=limits, ax=ax, color=edgecolor, ls='-', lw=lw, alpha=alpha_outline,
                      **kwargs)
    plot_img_patch(img=img, limits=limits, ax=ax, lw=0, hatch=hatch, facecolor=facecolor, edgecolor=hatchcolor,
                   alpha=alpha_patch, **kwargs)

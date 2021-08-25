import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

from wzk.mpl.bool_image_boundaries import get_combined_edges, get_combined_faces
# noinspection PyUnresolvedReferences
from wzk.numpy2 import limits2cell_size, grid_i2x, grid_x2i


def plot_img_patch(img, limits,
                   ax=None, **kwargs):
    """
    Plot an image as a Collection of square Rectangle Patches.
    Draw all True / nonzero pixels.
    """

    cell_size = limits2cell_size(shape=img.shape, limits=limits)

    ij = np.array(np.nonzero(img)).T
    xy = grid_i2x(i=ij, cell_size=cell_size, lower_left=limits[:, 0], mode='b')

    pc = PatchCollection([Rectangle((x, y), width=cell_size, height=cell_size,
                                    fill=False, snap=True) for (x, y) in xy], **kwargs)
    ax.add_collection(pc)
    return pc


def plot_img_outlines(img, limits,
                      ax, **kwargs):
    """
    Plot the image by drawing the outlines of the areas where the values are True.
    """

    combined_edges = get_combined_edges(img)

    cell_size = limits2cell_size(shape=img.shape, limits=limits)
    combined_edges = [grid_i2x(i=ce, cell_size=cell_size, lower_left=limits[:, 0], mode='b')
                      for ce in combined_edges]

    lc = LineCollection(combined_edges, **kwargs)
    ax.add_collection(lc)
    return lc


def __img_none_limits(limits=None, img=None):
    if limits is None and img is not None:
        limits = np.zeros((img.ndim, 2))
        limits[:, 1] = img.shape

    return limits


def plot_img_patch_w_outlines(img, limits,
                              ax,
                              color=None, edgecolor='k', hatchcolor='k', facecolor='None',
                              hatch='xx',
                              lw=2, alpha_outline=1, alpha_patch=1):
    if img is None:
        return None

    if color is not None:
        facecolor = color
        edgecolor = color

    limits = __img_none_limits(limits=limits, img=img)

    if img.ndim == 2:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

        plot_img_outlines(img=img, limits=limits, ax=ax, color=edgecolor, ls='-', lw=lw, alpha=alpha_outline)
        plot_img_patch(img=img, limits=limits, ax=ax, lw=0, hatch=hatch, facecolor=facecolor, edgecolor=hatchcolor,
                       alpha=alpha_patch)

    else:  # n_dim == 3
        cell_size = limits2cell_size(shape=img.shape, limits=limits)
        if isinstance(img, tuple):
            rect_pos, rect_size = img
            face_vtx = rectangles2face_vertices(rect_pos=rect_pos, rect_size=rect_size)
        else:
            face_vtx = get_combined_faces(img=img)

        face_vtx = grid_i2x(i=face_vtx, cell_size=cell_size, lower_left=limits[:, 0], mode='b')
        # plot_poly_collection_3d(face_vtx=face_vtx, ax=ax, facecolor=facecolor, edgecolor=edgecolor, alpha=0.4)

        from wzk import new_fig
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig, ax = new_fig(n_dim=3)

        poly = Poly3DCollection(face_vtx, facecolor=facecolor, edgecolor=edgecolor, alpha=0.4)
        ax.add_collection3d(poly)
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-1, 1)
        ax.scatter(*face_vtx.reshape(-1, 3).T)

# def plot_poly_collection_3d(face_vtx=face_vtx, ax=ax, facecolor=facecolor, edgecolor=edgecolor, alpha=0.4):

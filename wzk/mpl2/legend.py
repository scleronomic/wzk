import numpy as np

from matplotlib import patches, collections, transforms
from matplotlib.legend_handler import HandlerPatch, HandlerPathCollection

from wzk.np2 import flatten_without_last


class HandlerMultiPathCollection(HandlerPathCollection):
    """
    Handler for PathCollections, which are used by scatter
    """
    def __init__(self, offsets_x=None, **kwargs):
        super().__init__(**kwargs)
        self.offsets_x = offsets_x

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):

        if offsets is not None:
            center = np.mean([(o[0]) for o in offsets])
            offsets = [(center, o[1]) for o in offsets]

        p = type(orig_handle)(orig_handle.get_paths(), sizes=sizes,
                              offsets=offsets,
                              offset_transform=offset_transform,
                              )
        return p

    @staticmethod
    def get_pc(ax,
               paths, sizes,
               facecolors, edgecolors=None):
        PC = collections.PathCollection(paths, sizes, transOffset=ax.transData,
                                        facecolors=facecolors, edgecolors=edgecolors)
        PC.set_transform(transforms.IdentityTransform())
        return PC

    def get_pc2(self, ax, h_list):
        paths = [h.get_paths()[0] for h in h_list]
        sizes = [h.get_sizes()[0] for h in h_list]
        facecolors = [h.get_facecolors()[0] for h in h_list]
        edgecolors = [h.get_edgecolors()[0] for h in h_list]
        # alphas = [h.get_edgecolors()[0] for h in h_list]
        return self.get_pc(ax=ax, paths=paths, sizes=sizes,
                           facecolors=facecolors, edgecolors=edgecolors)


# Annotations
def annotate_arrow(ax, xy0, xy1, offset=0.,
                   xycoords="axes fraction", color="k", arrowstyle="->", zorder=None,
                   squeeze=True):

    xy0 = flatten_without_last(xy0)
    xy1 = flatten_without_last(xy1)

    ab = xy1 - xy0
    ab /= np.linalg.norm(ab, axis=-1, keepdims=True)
    xy0 = xy0 + offset * ab
    xy1 = xy1 - offset * ab
    arrowprops = dict(arrowstyle=arrowstyle,
                      color=color)
    handles = []
    for a, b in zip(xy0, xy1):
        handles.append(ax.annotate("", xytext=a, xy=b, xycoords=xycoords, arrowprops=arrowprops,
                       verticalalignment="center", horizontalalignment="center", zorder=zorder))

    if squeeze and len(handles) == 1:
        handles = handles[0]
    return handles


# Legends
def make_legend_arrow_lr(legend, orig_handle, xdescent, ydescent, width, height, fontsize):  # noqa
    return patches.FancyArrow(x=0, y=height/2, dx=width, dy=0, length_includes_head=True, head_width=0.75*height)


def make_legend_arrow_rl(legend, orig_handle, xdescent, ydescent, width, height, fontsize):  # noqa
    return patches.FancyArrow(x=width, y=height/2, dx=-width, dy=0, length_includes_head=True, head_width=0.75*height)


def make_legend_arrow_wrapper(theta, label2theta_dict=None):
    if isinstance(theta, str):
        if theta == "lr":
            return HandlerPatch(patch_func=make_legend_arrow_lr)
        elif theta == "rl":
            return HandlerPatch(patch_func=make_legend_arrow_rl)
        else:
            raise ValueError

    def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):  # noqa
        if label2theta_dict is not None and orig_handle.get_label() in label2theta_dict:
            theta2 = label2theta_dict[orig_handle.get_label()]
        else:
            theta2 = theta

        dx = np.cos(theta2) * height
        dy = np.sin(theta2) * height
        return patches.FancyArrow(x=width/2-dx, y=height/2-dy, dx=2*dx, dy=2*dy, length_includes_head=True,
                                  head_width=height/2)
    return HandlerPatch(patch_func=make_legend_arrow)


def align_legend(fig, l_str, w=80):
    r = fig.canvas.get_renderer()

    l_al = []
    for i, s in enumerate(l_str[0]):
        t = fig.text(0.1, 0.1, s, fontsize=8)
        bb = t.get_window_extent(renderer=r)
        l_al.append(s + r"\hspace{" + str(w - bb.fig_width_inch) + "pt} | " + l_str[1][i])
        t.remove()

    fig.axes[0].legend(l_al)


def remove_duplicate_labels(ax):
    """https://stackoverflow.com/a/26339101/7570817"""
    handles, labels = ax.get_legend_handles_labels()
    new_labels, new_handles = [], []
    for handle, label in zip(handles, labels):
        if label not in new_labels:
            new_labels.append(label)
            new_handles.append(handle)
    ax.legend(new_handles, new_labels)


def rectangle_legend(*, ax, xy=(-100, -100), **kwargs):
    p = patches.Rectangle(xy=xy, width=0.1, height=0.1, **kwargs)
    ax.add_patch(p)


def circle_legend(*, ax, xy=(-100, -100), **kwargs):
    p = patches.Circle(xy=xy, radius=0.1, **kwargs)
    ax.add_patch(p)

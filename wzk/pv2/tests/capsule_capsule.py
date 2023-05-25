import numpy as np

from itertools import combinations
from wzk import pv2, geometry


def vis_lines(n=2):

    lines = np.random.random((n, 2, 3))

    pairs = np.array(list(combinations(np.arange(n), 2)))
    pairs2 = np.arange(2 * len(pairs)).reshape(len(pairs), 2)
    lines2 = np.array(geometry.line_line_pairs(lines=lines, pairs=pairs)).swapaxes(0, -2)

    pl = pv2.Plotter()
    h_lines = pv2.plot_lines(x=lines, lines=pairs, pl=pl, opacity=1, color="blue")
    h_lines_penetration = pv2.plot_lines(x=lines2, lines=pairs2, pl=pl, opacity=1, color="red")

    def on_drag(point, i):
        lines.reshape(n * 2, 3)[i] = point
        lines2[:] = np.array(geometry.line_line_pairs(lines=lines, pairs=pairs)).swapaxes(0, -2)

        pv2.plot_lines(x=lines, lines=pairs, pl=pl, opacity=1, color="blue", h=h_lines)
        pv2.plot_lines(x=lines2, lines=pairs2, pl=pl, opacity=1, color="red", h=h_lines_penetration)

    pl.add_sphere_widget(on_drag, center=lines.reshape(n * 2, 3), radius=0.01, color="yellow")
    pl.show()


def vis_capsules(n=2):
    lines = np.random.random((n, 2, 3))
    radii = np.random.random(n)
    pairs = np.array(list(combinations(np.arange(n), 2)))
    pairs2 = np.arange(2 * len(pairs)).reshape(len(pairs), 2)
    print("a", pairs)
    print("b", pairs2)
    lines2 = np.array(geometry.capsule_capsule_pairs(lines=lines, pairs=pairs, radii=radii)[:2]).swapaxes(0, -2)

    hulls = [geometry.get_points_on_multisphere(x=xx, r=rr, n=500)[1] for xx, rr in zip(lines, radii)]

    pl = pv2.Plotter()
    h_lines_capsules = pv2.plot_lines(x=lines, lines=pairs, pl=pl, opacity=1, color="blue")
    h_lines_penetration = pv2.plot_lines(x=lines2, lines=pairs2, pl=pl, opacity=1, color="red")
    h_hulls = [pv2.plot_convex_hull(x=None, hull=h, pl=pl, opacity=0.5) for h in hulls]

    def on_drag(point, i):
        lines.reshape(n * 2, 3)[i] = point

        hulls[:] = [geometry.get_points_on_multisphere(x=xx, r=rr, n=100)[1] for xx, rr in zip(lines, radii)]
        lines2[:] = np.array(geometry.capsule_capsule_pairs(lines=lines, pairs=pairs, radii=radii)[:2]).swapaxes(0, -2)

        pv2.plot_lines(x=lines, lines=pairs, pl=pl, opacity=1, color="blue", h=h_lines_capsules)
        pv2.plot_lines(x=lines2, lines=pairs2, pl=pl, opacity=1, color="red", h=h_lines_penetration)
        [pv2.plot_convex_hull(x=None, hull=h, pl=pl, opacity=0.5, h=hh) for hh, h in zip(h_hulls, hulls)]

    pl.add_sphere_widget(on_drag, center=lines.reshape(n * 2, 3), radius=0.1, color="yellow")
    pl.show()


def vis_capsule_pairs():
    n = 2
    pairs = [[0, 1]]
    lines = np.random.random((n, 2, 3))
    radii = np.random.random(n)

    xa, xb, d0 = geometry.capsule_capsule(line_a=lines[0], line_b=lines[1],
                                          radius_a=radii[0], radius_b=radii[1])

    print("d", d0)
    hulls = [geometry.get_points_on_multisphere(x=xx, r=rr, n=100)[1] for xx, rr in zip(lines, radii)]

    print(np.array([[xa, xb], [xa, xb]]))
    pl = pv2.Plotter()
    h_lines_capsules = pv2.plot_lines(x=lines, lines=pairs, pl=pl, opacity=1, color="blue")
    h_lines_penetration = pv2.plot_lines(x=np.array([[xa, xb], [xa, xb]])[:, :, 0, :], lines=pairs, pl=pl, opacity=1, color="red")
    h_hulls = [pv2.plot_convex_hull(x=None, hull=h, pl=pl, opacity=0.5) for h in hulls]

    def on_drag(point, i):
        lines.reshape(n * 2, 3)[i] = point

        hulls[:] = [geometry.get_points_on_multisphere(x=xx, r=rr, n=100)[1] for xx, rr in zip(lines, radii)]
        print("lines")
        print(lines[0])
        print(lines[1])
        print("---")
        xa[:], xb[:], d = geometry.capsule_capsule(line_a=lines[0], line_b=lines[1],
                                                   radius_a=radii[0], radius_b=radii[1])

        print(d)
        pv2.plot_lines(pl=pl, h=h_lines_capsules, x=lines, lines=pairs, opacity=1, color="blue")
        pv2.plot_lines(pl=pl, h=h_lines_penetration, x=np.array([[xa, xb], [xa, xb]])[:, :, 0, :], lines=pairs,
                       opacity=1, color="red")
        [pv2.plot_convex_hull(x=None, hull=h, pl=pl, opacity=0.5, h=hh) for hh, h in zip(h_hulls, hulls)]

    pl.add_sphere_widget(on_drag, center=lines.reshape(n * 2, 3), radius=0.01, color="yellow")
    pl.show()


if __name__ == "__main__":
    # vis_capsules(2)
    vis_capsule_pairs()

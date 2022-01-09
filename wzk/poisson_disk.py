import numpy as np
from wzk.numpy2 import grid_x2i
from wzk.mpl import new_fig, plt, imshow, grid_lines


def ccw(a, b, c):
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) > (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


# Return true if line segments AB and CD intersect
def intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


d = 2
k = 100
r = 0.1
n = int(np.ceil(np.sqrt(d)/r))

np.random.random()

grid = np.full((n,)*d, -1)
limits = np.array([[-1.0, +1.0],
                   [-1.0, +1.0]])

# for xx in x:
#     ax.plot(*xx, color='black', marker='o', markersize=1)


def aaa(t):
    fig, ax = new_fig(aspect=1)
    ax.set_axis_off()
    ax.set_xlim(-1, +1)
    ax.set_ylim(-1, +1)
    grid_lines(ax=ax, start=limits[:, 0], step=r/np.sqrt(d), limits=limits)

    x0 = np.zeros(d)
    x = x0[np.newaxis, :].copy()
    x[:, 1] = -1
    # x = np.array([[-0.6, -0.5],
    #               [-0.4, -0.5],
    #               [-0.2, -0.5],
    #               [+0.0, -0.5],
    #               [+0.2, -0.5],
    #               [+0.4, -0.5],
    #               [+0.6, -0.5]])
    i = grid_x2i(x, limits=limits, shape=grid.shape)
    grid[i[:, 0], i[:, 1]] = np.arange(len(x))
    h = None
    active = [True] * len(x)
    while np.any(active):
        i = np.random.randint(0, np.sum(active))
        i = np.nonzero(active)[0][i]

        active[i] = False
        print(np.size(active), np.sum(active))
        for _ in range(k):
            _r = np.random.uniform(low=r, high=2*r, size=d)
            if i == 0:
                # phi = np.random.uniform(low=np.pi/3+0.8, high=2/3*np.pi-0.8)
                phi = np.random.normal(loc=np.pi/2, scale=t)
            else:
                dd = x[i] - x0
                # phi = np.random.normal(loc=np.arctan2(dd[1], dd[0]), scale=t)
                phi = np.random.normal(loc=np.pi/2, scale=t)

            x1 = x[i] + np.array([np.cos(phi), np.sin(phi)]) * _r
            if np.linalg.norm(x1-x0) > 1:
                continue

            i1 = grid_x2i(x1, limits=limits, shape=grid.shape)
            # if all(np.linalg.norm(x1 - x, axis=-1) >= r):
            if grid[i1[0], i1[1]] == -1:
                grid[i1[0], i1[1]] = len(x) + 1
                x = np.concatenate([x, x1[np.newaxis, :]])
                active += [True]
                # ax.plot(*x1, color='black', marker='o', markersize=1)
                ax.plot((x[i, 0], x1[0]), (x[i, 1], x1[1]), color='black', lw=0.5)
                h = imshow(ax=ax, h=h, img=grid, limits=limits, cmap='black', alpha=0.1, mask=grid != -1)
                plt.pause(0.001)
                active[i] = True



aaa(np.pi/10)

# for tt in [np.pi/15, np.pi/20, np.pi/25]:
#     aaa(tt)
#
#
# for i in range(5):
#     aaa(np.pi/25)
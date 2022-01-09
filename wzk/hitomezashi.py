import numpy as np


def to_bool(s, mode=None):
    if mode is None:
        try:
            int(s)
            mode = 'odds'
        except ValueError:
            mode = 'vowels'

    if mode == 'odds':
        b = __odds2bools(s)

    elif mode == 'vowels':
        b = __vowels2bools(s)
    else:
        raise ValueError

    return b


def __vowels2bools(s):
    vowels = 'aeiouäöü'
    return np.array([ss.lower() in vowels for ss in s])


def __odds2bools(s):
    return np.array([(int(ss) % 2) == 1 for ss in s])


def get_initial_rows(a, b, mode='vowels'):
    return to_bool(a), to_bool(b)


def make_grid(a, b):
    a = np.hstack([[False], a])
    ia, ib = np.meshgrid(range(len(b)), range(len(a)), indexing='ij')
    aaa = (ia + a[np.newaxis, :]) % 2 == 1
    bbb = (ib + b[:, np.newaxis]) % 2 == 1

    grid = np.concatenate([aaa[:1, :], bbb], axis=0)
    grid[0] = (np.cumsum(grid[0]) % 2) == 1
    grid = (np.cumsum(grid, axis=0) % 2) == 1
    grid = grid.T
    grid = grid[:, ::-1]

    return grid


def plot_grid(grid):
    from wzk.mpl import new_fig, turn_ticklabels_off, turn_ticks_off, plot_img_patch_w_outlines

    fig, ax = new_fig(aspect=1)
    limits = np.array([[0, grid.shape[0]],
                       [0, grid.shape[1]]])

    limits2 = np.array([[0, grid.shape[1]],
                        [0, grid.shape[0]]])

    # imshow(ax=ax, img=grid, limits=limits2, cmap='blue', mask=~grid, alpha=0.5)
    plot_img_patch_w_outlines(ax=ax, img=grid, limits=limits, alpha_patch=0.5, hatch='//////', lw=0.5)
    ax.set_xlim(-1, limits[0, 1]+1)
    ax.set_ylim(-1, limits[1, 1]+1)
    turn_ticklabels_off(ax=ax, axes='xy')
    turn_ticks_off(ax=ax)


def plot_stitches(ax, limits):
    pass


def main(a, b):
    a = a.strip()
    b = b.strip()
    ia, ib = get_initial_rows(a=a, b=b)
    grid = make_grid(a=ia, b=ib)

    plot_grid(~grid)


if __name__ == '__main__':
    main(a='LisaSophiaTenhumberg', b='JohannesValentinHamm')
    # main(a='Julia', b='Hamm')
    # main(a='Johannes', b='Tenhumberg')
    # main(a='Theresa', b='Tenhumberg')

    # main(b='FabianWaldenmaier',
    #      a='CharlotteKonkel')

    # main(b='Ein Teil des Teils der Anfangs alles war',
    #      a='Ein Teil der Finsternis, die sich das Licht gebar')

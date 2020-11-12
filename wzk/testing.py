import numpy as np
from wzk import new_fig


def all_close(arr_a, arr_b, axis=None, names=('A', 'B'), title='',
              verbose=1):
    assert arr_a.shape == arr_b.shape
    are_equal = np.allclose(arr_a, arr_b)

    if verbose >= 1:
        print(f"{title}: , {are_equal}")
    if verbose >= 2 and not are_equal:
        _, axes = new_fig(title=f"{title} {names[0]} - {names[1]}",
                          n_rows=3, n_cols=1, share_x=True)
        axes[0].set_ylabel(names[0])
        axes[1].set_ylabel(names[1])
        axes[2].set_ylabel('Difference')
        if axis is None:
            axes[0].plot(arr_a.ravel())
            axes[1].plot(arr_b.ravel())
            axes[2].plot((arr_b-arr_a).ravel())

        else:
            for a, b in zip(np.split(arr_a, arr_a.shape[axis], axis),
                            np.split(arr_b,  arr_b.shape[axis], axis)):
                axes[0].plot(a.ravel())
                axes[1].plot(b.ravel())
                axes[2].plot((b-a).ravel())

    return are_equal

import numpy as np
from wzk.splines import NURBS
from wzk.mpl2 import new_fig, plt, DraggableCircleList


def test_basis_function():

    n_points = 10
    p = np.random.random((n_points, 3))
    nurbs = NURBS(p=p, degree=0)

    u = np.linspace(0, 1, 100)
    for degree in range(5):
        nurbs.degree = degree
        nurbs.set_knotvector(k=np.linspace(0, 1, n_points+degree+1))
        fig, ax = new_fig()
        ax.set_aspect(1)
        for i in range(10):
            n_in = nurbs.n_in(u=u, i=i, n=degree)
            ax.plot(u, n_in+i*0.001,  label=f"{degree} - {i}")

        ax.legend()


def test_random():
    n = 3
    p = np.random.random((n, 2))
    u = np.linspace(0, 1, 20)

    nurbs = NURBS(p=p, degree=3)
    x = nurbs.evaluate(u)

    fig, ax = new_fig()
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(*x.T, color="black", marker="o", markersize=3)
    ax.plot(*p.T, color="blue", marker="o")


def test_random_jac():
    n = 5
    p = np.random.random((n, 2))
    u = np.linspace(0, 1, 20)

    nurbs = NURBS(p=p, degree=3)
    x = nurbs.evaluate(u)

    def length_jac(xx):
        steps = xx[..., 1:, :] - xx[..., :-1, :]
        return steps[..., :-1, :] - steps[..., +1:, :]  # do_dx

    fig, ax = new_fig()
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hx = ax.plot(*x.T, color="black", marker="o", markersize=3)[0]
    hp = ax.plot(*p.T, color="blue", marker="o")[0]

    dx_dp = nurbs.evaluate_jac(u)[..., 1:-1, :]  # does not depend on p
    for i in range(100):
        do_dx = length_jac(x)
        do_dp = (do_dx[:, np.newaxis, :] * dx_dp[:, :, np.newaxis]).sum(axis=0)
        nurbs.p[1:-1] -= do_dp[1:-1]
        x = nurbs.evaluate(u)

        hx.set_data(*x.T)
        hp.set_data(*nurbs.p.T)
        plt.pause(0.1)
        # input('press key for next gradient step')


def test_unit_circle():
    k = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=float)
    p = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]])

    sq22 = np.sqrt(2)/2
    w = np.array([1, sq22, 1, sq22, 1, sq22, 1, sq22, 1])
    nurbs = NURBS(p=p, k=k, w=w, degree=2)
    u = np.linspace(0, 1, 90)
    x = nurbs.evaluate(u=u)
    j = nurbs.evaluate_jac(u=u)
    print("x", x)
    print("j", j)
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.plot(*x.T, color="black", marker="o")
    ax.plot(*p.T, color="blue", marker="o", markersize=3)


def test_gui():
    n = 3
    p = np.random.random((n, 2))
    u = np.linspace(0., 1, 20)

    fig, ax = new_fig(aspect=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    dcl = DraggableCircleList(ax=ax, xy=p, radius=0.02, color="r")
    nurbs = NURBS(p=p, degree=3)
    x = nurbs.evaluate(u)
    h = ax.plot(*x.T, color="k", marker="o", lw=3)[0]

    def update(*args):  # noqa
        nurbs.p = dcl.get_xy()
        xx = nurbs.evaluate(u)
        h.set_xdata(xx[:, 0])
        h.set_ydata(xx[:, 1])

    dcl.set_callback_drag(update)


if __name__ == "__main__":
    pass
    # pass
    # test_random_jac()
    # test_basis_function()
    # test_unit_circle()
    # test_random()

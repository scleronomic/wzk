import numpy as np


def random_ball_search(fun, n_outer, n_inner, x0, verbose=0):
    x0 = x0.ravel()
    n_x = len(x0)

    x_outer = np.zeros((n_outer+1, n_x))
    x_outer[0] = x0
    o_outer = np.zeros(n_outer+1)
    o_outer[0] = fun(x0)

    radius = 0.1
    for i in range(n_outer):

        delta = (np.random.random((n_inner, n_x)) - 0.5) * radius

        x_inner = np.zeros((n_inner, n_x))
        o_inner = np.zeros(n_inner)

        for j in range(n_inner):
            x_inner[j] = x_outer[i]+delta[j]
            o_inner[j] = fun(x_inner[j])

        j = np.argmin(o_inner)
        if o_inner[j] < o_outer[i]:
            o_outer[i+1] = o_inner[j]
            x_outer[i+1] = x_inner[j]
            radius = np.linalg.norm(delta[j]) * 1.5
        else:
            o_outer[i+1] = o_outer[i]
            x_outer[i+1] = x_outer[i]
            radius = radius * 0.75
        if verbose > 0:
            print(f"iteration: {i} | objective: {o_outer[i+1]:.5}")

    return x_outer[-1], o_outer[-1]


def random_line_search(fun, n_outer, n_inner, x0):
    x0 = x0.ravel()
    n_x = len(x0)

    x_outer = np.zeros((n_outer+1, n_x))
    x_outer[0] = x0
    o_outer = np.zeros(n_outer+1)
    o_outer[0] = fun(x0)

    for i in range(n_outer):

        v = np.random.random(n_x)
        v = v / np.linalg.norm(v)

        x_inner = np.zeros((n_inner, n_x))
        o_inner = np.zeros(n_inner)

        ss = np.logspace(-6, 1, n_inner//2)
        ss = np.hstack([-ss, +ss])
        for j in range(n_inner):
            x_inner[j] = x_outer[i]+ss[j]*v
            o_inner[j] = fun(x_inner[j])

        j = np.argmin(o_inner)
        x_outer[i+1] = x_inner[j]
        o_outer[i+1] = o_inner[j]

        print(i, o_outer[i+1])
        # print(np.abs(np.linspace(-0.1, +0.1, n_ls)[np.argmin(o)]))
        # fig, ax = mpl2.new_fig()
        # ax.plot(o)



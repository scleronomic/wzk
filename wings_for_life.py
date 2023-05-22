import numpy as np

from wzk import mpl2


v_car0 = 7
a_car = 0.5
a_cat_dt = 1


def v_auto(tt):
    v = v_car0 + a_car * (tt-2)
    v = np.atleast_1d(v)
    v[tt == 0] = 0
    v[tt == 1] = 0

    return v


def s_auto(tt):
    if isinstance(tt, (int, np.int64)):
        return np.sum([v_auto(tt) for tt in range(0, tt+1)])
    else:
        return [s_auto(ttt) for ttt in tt]


t_max = 10
t = np.arange(t_max+1)


fig, ax = mpl2.new_fig()
ax.plot(t, v_auto(t), color="red", marker="o")
ax.plot(t, s_auto(t), color="blue", marker="o")
ax.plot(t, s_auto(t)/t, color="magenta", marker="o")
ax.grid()
ax.set_xlim(0, t_max)
ax.set_ylim(0, 50)

# s_auto(t) = 7*t + sum_(0.5 + 1 + 1.5)
print(v_auto(t))
print(s_auto(t))

# 0: 0
# 1: 0.5
# 2: 1.5
# 3: 3
# 4: 5
# 5: 7.5
# sum

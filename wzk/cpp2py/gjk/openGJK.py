import numpy as np
import importlib
openGJK = importlib.import_module("wzk.cpp2py.gjk.wzkopenGJK")

_cpp_dtype = "f8"  # "f4" -> float, "f8" -> double
_cpp_order = "c"


def _np2cpp(a):
    return a.astype(dtype=_cpp_dtype, order=_cpp_order)


def gjk(p1, p2, s=None):
    if s is None:
        s = np.zeros((4, 3), dtype=_cpp_dtype, order=_cpp_order)
    else:
        s = _np2cpp(s)

    p1 = _np2cpp(p1)
    p2 = _np2cpp(p2)

    c1 = np.zeros(3, dtype=_cpp_dtype, order=_cpp_order)
    c2 = np.zeros(3, dtype=_cpp_dtype, order=_cpp_order)
    d = np.zeros(1, dtype=_cpp_dtype, order=_cpp_order)

    openGJK.compute_minimum_dist(p1, p1.shape[0], p2, p2.shape[0], s, c1, c2, d)

    return d[0], c1, c2


def test_gjk():
    p1 = np.array([[+0.0, +5.5, +0.0],
                   [+2.3, +1.0, -2.0],
                   [+8.1, +4.0, +2.4],
                   [+4.3, +5.0, +2.2],
                   [+2.5, +1.0, +2.3],
                   [+7.1, +1.0, +2.4],
                   [+1.0, +1.5, +0.3],
                   [+3.3, +0.5, +0.3],
                   [+6.0, +1.4, +0.2]])

    p2 = np.array([[-0.0, -5.5, +0.0],
                   [-2.3, -1.0, +2.0],
                   [-8.1, -4.0, -2.4],
                   [-4.3, -5.0, -2.2],
                   [-2.5, -1.0, -2.3],
                   [-7.1, -1.0, -2.4],
                   [-1.0, -1.5, -0.3],
                   [-3.3, -0.5, -0.3],
                   [-6.0, -1.4, -0.2]])
    d, c1, c2 = gjk(p1, p2)

    assert d == 3.6536497222945004
    assert np.isclose(c1, np.array([+1., +1.5, +0.3]))
    assert np.isclose(c2, np.array([-1., -1.5, -0.3]))

    # TODO does not compute negative distances
    #   check gjkepa

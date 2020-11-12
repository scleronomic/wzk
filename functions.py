

def get_argument_names(fun):
    return fun.__code__.co_varnames[:fun.__code__.co_argcount]


def get_number_of_arguments(fun):
    return fun.__code__.co_argcount


def common_argument_wrapper(fun, **kwargs_common):
    def fun2(**kwargs):
        kwargs.update(kwargs_common)
        return fun(**kwargs)
    return fun2


def test_common_argument_wrapper():

    def fun(a, b, c, d, e, f, g):
        return a + b + c + d + e + f + g

    aa, bb, cc, dd, ee, ff, gg = 1, 2, 3, 4, 5, 6, 7

    fun_c = common_argument_wrapper(fun=fun, a=aa, b=bb, c=cc, d=dd)

    assert fun(a=aa, b=bb, c=cc, d=dd, e=ee, f=ff, g=gg) == fun_c(e=ee, f=ff, g=gg)

import numpy as np

# noinspection PyUnresolvedReferences
from pyOpt.pySLSQP.pySLSQP import SLSQP
# noinspection PyUnresolvedReferences
from pyOpt.pyCOBYLA.pyCOBYLA import COBYLA
# from pyOpt.pyNLPQLP.pyNLPQLP import NLPQLP

# noinspection PyUnresolvedReferences
from pyOpt import Optimization
from scipy.optimize import least_squares  # TODO combine PyOpt and scipy to one


def print_result(res, verbose):
    if verbose > 1:
        print("------ Result ------")
        print(res)


def fun_wrapper(fun):

    def fun2(_x):
        f = fun(_x)
        g = []
        fail = 0
        return f, g, fail

    return fun2


def minimize_cobyla(fun, x0, options, verbose=0):
    cobyla = COBYLA(pll_type=options["pll_type"])
    cobyla.setOption("IPRINT", 3)
    cobyla.setOption("MAXFUN", 10000)

    opt_prob = Optimization("", fun_wrapper(fun))
    for i, x0_i in enumerate(x0):
        opt_prob.addVar(f"x{i+1}", "c", lower=x0_i-1, upper=x0_i+1, value=x0_i)
    opt_prob.addObj("f")
    cobyla(opt_prob)

    res = opt_prob.solution(0)
    print_result(res=res, verbose=verbose)

    vs = res.getVarSet()
    x = np.array([vs[key].value for key in range(len(x0))])
    return x


def minimize_slsqp(fun, x0, options, verbose=0):
    slsqp = SLSQP(pll_type=options["pll_type"])
    slsqp.setOption("IPRINT", 0 if verbose > 5 else -1)
    slsqp.setOption("MAXIT", options["maxiter"])
    try:
        slsqp.setOption("ACC", options["ftol"])
    except KeyError:
        pass

    opt_prob = Optimization("", fun_wrapper(fun))
    for i, x0_i in enumerate(x0):
        opt_prob.addVar(f"x{i+1}", "c", lower=x0_i-1, upper=x0_i+1, value=x0_i)
        # opt_prob.addVar(f"x{i+1}", 'c', lower=-10, upper=+10, value=x0_i)
    opt_prob.addObj("f")
    slsqp(opt_prob,
          sens_type=options["sens_type"],
          sens_step=options["sens_step"])

    res = opt_prob.solution(0)
    print_result(res=res, verbose=verbose)

    vs = res.getVarSet()
    x = np.array([vs[key].value for key in range(len(x0))])
    return x


def minimize(method, fun, x0, options, verbose):
    if method == "PyOpt-SLSQP":
        x = minimize_slsqp(fun=fun, x0=x0, options=options, verbose=verbose - 1)
    elif method == "PyOpt-COBYLA":
        x = minimize_cobyla(fun=fun, x0=x0, options=options, verbose=verbose - 1)
    elif method == "SciPy-LS":
        x = least_squares(fun=fun, x0=x0, method="lm").x
    else:
        raise ValueError(f"Unknown optimizer: {method}")

    return x

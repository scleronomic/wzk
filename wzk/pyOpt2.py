import numpy as np

from pyOpt.pySLSQP.pySLSQP import SLSQP
from pyOpt.pyCOBYLA.pyCOBYLA import COBYLA
# from pyOpt.pyNLPQLP.pyNLPQLP import NLPQLP
from pyOpt import Optimization

from scipy import optimize
# from scipy.optimize import least_squares, slsqp


# bayes_opt and pyswarms seem so infinitely slow compared to SLSQP, not sure when they are better suited

def print_result(res, verbose):
    if verbose > 1:
        print("------ Result ------")
        print(res)


def fw_objective_pyopt(fun):

    def fun2(_x):
        f = fun(_x)
        g = []
        fail = 0
        return f, g, fail

    return fun2


def create_opt_problem(fun, x0, lower=-0.1, upper=+0.1):
    opt_prob = Optimization("", fw_objective_pyopt(fun))
    for i, x0_i in enumerate(x0):
        opt_prob.addVar(f"x{i+1}", "c", lower=x0_i+lower, upper=x0_i+upper, value=x0_i)
    opt_prob.addObj("f")
    return opt_prob


def minimize_cobyla(fun, x0, options, verbose=0):
    cobyla = COBYLA(pll_type=options["pll_type"])
    cobyla.setOption("IPRINT", 3)
    cobyla.setOption("MAXFUN", 10000)

    opt_prob = create_opt_problem(fun=fun, x0=x0)

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

    opt_prob = create_opt_problem(fun=fun, x0=x0)

    slsqp(opt_prob,
          sens_type=options["sens_type"],
          sens_step=options["sens_step"])

    res = opt_prob.solution(0)
    print_result(res=res, verbose=verbose)

    vs = res.getVarSet()
    x = np.array([vs[key].value for key in range(len(x0))])
    return x


# --- Bayes Opt ---
def ih_get_keys(n):
    return [f"x{i:0>3}" for i in range(n)]


def fw_objective_bayes(fun):

    def fun2(**kwargs):
        x = np.array([kwargs[k] for k in ih_get_keys(len(kwargs))])
        return -fun(x)

    return fun2


def minimize_bayes_opt(fun, x0, options, verbose=0):
    from bayes_opt import BayesianOptimization
    low = -0.1
    high = +0.1
    n = len(x0)
    bounds = {k: (low, high) for i, k in enumerate(ih_get_keys(n))}

    optimizer = BayesianOptimization(
        f=fw_objective_bayes(fun),
        pbounds=bounds,
    )

    optimizer.maximize(
        init_points=2000,
        n_iter=20,
    )

    x = optimizer.max["params"]
    x = np.array([x[k] for k in ih_get_keys(n)])
    return x


# --- Swarms ---
def fw_objective_swarms(fun):

    def fun2(x):
        return np.array([fun(xx) for xx in x])

    return fun2


def minimize_swarms(fun, x0, options, verbose=0):
    import pyswarms

    n = len(x0)
    low = -0.1
    high = +0.1
    bounds = ( np.full(n, low), np.full(n, high))
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = pyswarms.single.GlobalBestPSO(n_particles=500, dimensions=n, options=options, bounds=bounds)
    o, x = optimizer.optimize(fw_objective_swarms(fun), iters=1000)

    return x


def minimize(method, fun, x0, options, verbose):
    if "PyOpt" in method:
        if method == "PyOpt-SLSQP":
            x = minimize_slsqp(fun=fun, x0=x0, options=options, verbose=verbose - 1)
        elif method == "PyOpt-COBYLA":
            x = minimize_cobyla(fun=fun, x0=x0, options=options, verbose=verbose - 1)
        else:
            raise ValueError(f"Unknown optimizer: {method}")

    elif "SciPy" in method:
        if method == "SciPy-LS":
            x = optimize.least_squares(fun=fun, x0=x0, method="lm").x

        else:
            method = method.split("-")[1]
            method = method.lower()
            x = optimize.minimize(fun=fun, x0=x0, method=method, tol=1e-13,
                                  options=dict(disp=True, maxiter=1000)).x

    elif "Bayes" in method:
        x = minimize_bayes_opt(fun=fun, x0=x0, options=options)

    elif "Swarms" in method:
        x = minimize_swarms(fun=fun, x0=x0, options=options)

    else:
        raise ValueError(f"Unknown optimizer: {method}")

    return x

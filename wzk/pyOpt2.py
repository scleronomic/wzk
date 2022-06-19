import numpy as np

from pyOpt.pySLSQP.pySLSQP import SLSQP
from pyOpt import Optimization


def minimize_slsqp(fun, x0, options, verbose=0):
    slsqp = SLSQP()
    slsqp.setOption('IPRINT', -1)
    slsqp.setOption('MAXIT', options['maxiter'])
    try:
        slsqp.setOption('ACC', options['ftol'])
    except KeyError:
        pass

    def fun2(_x):
        f = fun(_x)
        g = []
        fail = 0
        return f, g, fail

    opt_prob = Optimization('', fun2)
    for i, x0_i in enumerate(x0):
        opt_prob.addVar(f"x{i+1}", 'c', lower=-10, upper=10, value=x0_i)
    opt_prob.addObj('f')
    slsqp(opt_prob, sens_type='FD')
    res = opt_prob.solution(0)
    if verbose > 1:
        print('------ Result ------')
        print(res)

    vs = res.getVarSet()
    x = np.array([vs[key].value for key in range(len(x0))])
    return x

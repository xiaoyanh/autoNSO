import numpy as np
import cvxpy as cp

tol = 1e-6
m_params = {}
# m_params = {
#             'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_INFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_MU_RED': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_NEAR_REL': 10,
#             'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
#             # 'MSK_IPAR_OPTIMIZER': 'CONIC',
#             'MSK_IPAR_INTPNT_MAX_ITERATIONS': int(1e4),
#             # mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
#             }

g_params = {'BarConvTol': 1e-10,
            'BarQCPConvTol': 1e-10,
            'FeasibilityTol': 1e-9,
            'OptimalityTol': 1e-9,}

# Combinatorially find leaving index, MIP-Version 1
def get_lam_MIP(dfS, new_df=None,rank=None, solver='MOSEK'):

    if new_df is not None:
        if rank is None:
            rank = dfS.shape[0]
        dfS2 = np.stack((dfS,new_df))
    else:
        dfS2 = dfS.copy()

    if rank >= dfS2.shape[0]:
        rank = None

    k = dfS2.shape[0]
    xdim = dfS2.shape[1]

    p_tmp = cp.Variable(xdim)
    lam   = cp.Variable(k)

    constraints = [cp.sum(lam) == 1.0]
    constraints += [lam >= 0]

    if (rank is not None) or (new_df is not None):

        non_zero = cp.Variable(k, integer=True)
        constraints += [non_zero <= 1]
        constraints += [lam <= non_zero]
        constraints += [cp.sum(non_zero) == rank]

    # Find lambda (warm start with previous iteration)
    prob = cp.Problem(cp.Minimize(cp.quad_form(p_tmp, np.eye(xdim))),
                      constraints + [lam @ dfS2 == p_tmp])

    if solver=='MOSEK':
        prob.solve(warm_start=True, solver=cp.MOSEK, mosek_params=m_params)
    elif solver=='GUROBI':
        prob.solve(solver=cp.GUROBI, verbose=True)

    if (rank is not None) or (new_df is not None):
        keep_inds = np.where(non_zero.value > 0)
        return np.sqrt(prob.value), lam.value.copy(), keep_inds[0]
    else:
        return np.sqrt(prob.value), lam.value.copy()

# Combinatorially find leaving index, MIP-Version 2
def get_active(dfS, rank=None, solver='MOSEK'):

    dfS2 = dfS.copy()
    k, x_dim = dfS2.shape

    if rank >= k:
        return np.arange(dfS.shape[0])

    non_zero = cp.Variable(k, integer=True)
    M        = cp.Variable((k,k), diag=True)

    constraints = [non_zero <= 1]
    constraints += [0 <= non_zero]
    constraints += [cp.sum(non_zero) == rank]
    constraints += [M == cp.diag(non_zero)]

    # Find rows
    prob = cp.Problem(cp.Maximize(cp.trace(dfS2.T @ M @ dfS2)),constraints)

    if solver == 'MOSEK':
        prob.solve(warm_start=True, solver=cp.MOSEK, mosek_params=m_params)
    elif solver == 'GUROBI':
        prob.solve(solver=cp.GUROBI, verbose=True)

    return np.where(non_zero.value > 0)[0]
import quadprog
import numpy as np
import cvxpy as cp
import matlab.engine
import multiprocessing
from IPython import embed
from scipy.optimize import linprog
from joblib import Parallel, delayed

tol = 1e-10

# m_params = {'MSK_DPAR_INTPNT_QO_TOL_DFEAS': tol,
#             'MSK_DPAR_INTPNT_QO_TOL_INFEAS': tol,
#             'MSK_DPAR_INTPNT_QO_TOL_MU_RED': tol,
#             'MSK_DPAR_INTPNT_QO_TOL_NEAR_REL': 10,
#             'MSK_DPAR_INTPNT_QO_TOL_PFEAS': tol,
#             'MSK_DPAR_INTPNT_QO_TOL_REL_GAP': tol,
#             mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
#             # 'MSK_IPAR_INTPNT_MAX_ITERATIONS': int(1e3),
#             }

m_params = {}
# m_params = {
#             'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_INFEAS': 1e-12*(tol*1e8),
#             'MSK_DPAR_INTPNT_CO_TOL_MU_RED': tol,
#             # 'MSK_DPAR_INTPNT_CO_TOL_NEAR_REL': 10,
#             'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
#             # 'MSK_IPAR_OPTIMIZER': 'CONIC',
#             'MSK_IPAR_INTPNT_MAX_ITERATIONS': int(1e4),
#             mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
#             }
g_params = {'BarConvTol': 1e-10,
            'BarQCPConvTol': 1e-10,
            'FeasibilityTol': 1e-9,
            'OptimalityTol': 1e-9,}

osqp_params = {"eps_abs":1e-10,
               "eps_rel":1e-10,
               'eps_prim_inf':1e-7,
               'eps_dual_inf':1e-10,
               'max_iter':int(1e5),
               # 'polish' : 1,
               'adaptive_rho': 1,
               'sigma':1e-8,
               'alpha':1.1}

cvx_params = {'max_iters' : int(1e3),
                # 'abstol'  : 1e-9,
                # 'reltol'  : 1e-8,
                # 'feastol' : 1e-9,
                'kktsolver' : 'robust',
              }

matlab_params = {'tol' : 1e-16,
                 'maxiter' : 1e4,}

# Combinatorially find leaving index
def get_lam(dfS,new_df=None, solver='MOSEK', eng=None):

    if solver is 'MATLAB':
        assert eng is not None
        maxiter = matlab_params['maxiter']
        tol     = matlab_params['tol']

        P = matlab.double(dfS.T.tolist())

        if new_df is not None:
            sub_vec = matlab.double(new_df.tolist())
            lam, delta = eng.WolfeAlg(P, tol, tol, tol, maxiter, sub_vec, nargout=2)
            lam = np.asarray(lam).squeeze()
            delta = np.asarray(delta).squeeze()
        else:
            lam, delta = eng.WolfeAlg(P, tol, tol, tol, maxiter, nargout=2)
        lam = np.asarray(lam).squeeze()
        delta = np.asarray(delta).squeeze()

        return delta, lam
    else:
        k = dfS.shape[0]

        if new_df is not None:
            def conv_size(i):
                dfS_ = dfS.copy()
                dfS_[i,:] = new_df
                return get_lam(dfS_, solver=solver)
            jobs = Parallel(n_jobs=min(multiprocessing.cpu_count(),k))(delayed(conv_size)(i) for i in range(k))

            jobs_delta = np.array([jobs[i][0] for i in range(k)])
            jobs_lambda = np.array([jobs[i][1] for i in range(k)])
            return jobs_delta, jobs_lambda
        else:
            Q = dfS @ dfS.T

            if solver == 'quadprog':
                Q *= 2
                C = np.concatenate((np.ones(k)[np.newaxis],np.eye(k)))
                b = np.zeros(k+1)
                b[0] = 1
                prob = quadprog.solve_qp(Q,np.zeros(k),C.T,b,1)

                return np.sqrt(prob[1]), prob[0]
            else: # cvxpy
                lam = cp.Variable(k)
                constraints = [cp.sum(lam) == 1.0]
                constraints += [lam >= 0.0]

                # Find lambda (warm start with previous iteration)
                prob = cp.Problem(cp.Minimize(cp.quad_form(lam, Q)), constraints)

                try:
                    if solver == 'MOSEK':
                        prob.solve(solver=cp.MOSEK, mosek_params=m_params)
                    elif solver == 'GUROBI':
                        prob.solve(solver=cp.GUROBI,**g_params)
                    elif solver == 'OSQP':
                        prob.solve(solver=cp.OSQP, **osqp_params)
                    elif solver == 'CVXOPT':
                        prob.solve(solver=cp.CVXOPT, **cvx_params)
                except:
                    if solver == 'MOSEK':
                        prob.solve(solver=cp.MOSEK, mosek_params=m_params, verbose=True)
                    elif solver == 'GUROBI':
                        prob.solve(solver=cp.GUROBI,**g_params, verbose=True)
                    elif solver == 'OSQP':
                        prob.solve(solver=cp.OSQP, **osqp_params, verbose=True)
                    elif solver == 'CVXOPT':
                        prob.solve(solver=cp.CVXOPT, **cvx_params, verbose=True)

                # return np.sqrt(prob.value), lam.value.copy()
                return np.linalg.norm(lam.value@dfS), lam.value.copy()

def get_LS(S,fS,dfS,sub_ind=None,new_S=None,new_fS=None,new_df=None,):
    dfS_  = dfS.copy()
    S_    = S.copy()
    fS_   = fS.copy()
    if sub_ind is not None:
        dfS_[sub_ind] = new_df
        S_[sub_ind]   = new_S
        fS_[sub_ind]  = new_fS

    A_ub = np.concatenate((dfS_,-np.ones(len(fS_))[:,np.newaxis]),axis=1)
    b_ub = np.einsum("sj,sj->s",dfS_,S_) - fS_
    c = np.zeros(S_.shape[1]+1)
    c[-1] = 1

    lp = linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=(None,None))
    return lp.fun
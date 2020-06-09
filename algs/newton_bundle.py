import torch
import mosek
import quadprog
import numpy as np
import cvxpy as cp
import multiprocessing

from IPython import embed
from utils.pinv import pinv2
# from scipy.linalg import pinv2
from algs.optAlg import OptAlg
from scipy.sparse import diags
from utils.diameter import get_diam
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

# Bundle Newton Method from Lewis-Wylie 2019
class NewtonBundle(OptAlg):
    def __init__(self, objective, k=4, delta_thres=0, diam_thres=0, proj_hess=False, warm_start=None, start_type='bundle',
                 bundle_prune='lambda', rank_thres=1e-3, pinv_cond=float('-inf'), random_sz=1e-1, adaptive_bundle=False,
                 store_hessian=False, solver='MOSEK', **kwargs):
        objective.oracle_output='hess+'

        super(NewtonBundle, self).__init__(objective, **kwargs)

        # Set up criterion
        self.criterion = self.objective.obj_func

        # Add start with initial point
        self.delta_thres = delta_thres
        self.diam_thres  = diam_thres
        self.proj_hess   = proj_hess
        self.rank_thres  = rank_thres
        self.pinv_cond   = pinv_cond
        self.random_sz   = random_sz
        self.adaptive_bundle = adaptive_bundle
        self.store_hessian = store_hessian

        self.solver = solver
        assert solver in ['MOSEK','GUROBI','OSQP','CVXOPT','quadprog']

        print("Project Hessian: {}".format(self.proj_hess),flush=True)

        # Prepare the bundle
        if warm_start is None:
            self.cur_x = self.x0
            self.S = None
            self.k = k  # bundle size
        else:
            self.cur_x      = warm_start['x']
            self.cur_iter   = warm_start['iter']
            self.x0         = None
            self.x_dim      = len(self.cur_x)

            if start_type == 'bundle':
                self.S      = warm_start['bundle']
                self.k = self.S.shape[0]
            elif start_type == 'random':
                self.k = k
                self.S = self.cur_x + np.random.randn(self.k, self.x_dim) * np.linalg.norm(self.cur_x) * self.random_sz
            else:
                raise Exception('Start type must me bundle or random')

            self.path_x = np.zeros([self.cur_iter, self.x_dim]) * np.nan
            self.path_hess = np.zeros([self.cur_iter, self.x_dim]) * np.nan
            self.path_fx = np.zeros([self.cur_iter]) * np.nan
            self.path_diam = np.zeros([self.cur_iter]) * np.nan
            self.path_delta = np.zeros([self.cur_iter]) * np.nan
            # self.path_vio   = np.zeros([self.cur_iter]) * np.nan

        self.cur_fx = self.criterion(torch.tensor(self.cur_x, dtype=torch.double, requires_grad=False)).data.numpy()

        if self.S is None: # If bundle is none, randomly initialize it (k * n)
            self.S = np.zeros([self.k,self.x_dim])
            self.S[0,:] = self.x0
            if self.k > 1:
                self.S[1:,:] = self.x0 + np.random.randn(self.k-1,self.x_dim)

        # Add higher order info results
        self.fS   = np.zeros(self.k)
        self.dfS  = np.zeros([self.k,self.x_dim])
        self.d2fS = np.zeros([self.k,self.x_dim,self.x_dim])
        for i in range(self.k):
            oracle = self.objective.call_oracle(self.S[i,:])
            self.fS[i]   = oracle['f']
            self.dfS[i,:]  = oracle['df']
            self.d2fS[i,:,:] = oracle['d2f']

        # # Add extra step where we reduce rank of S
        if warm_start and start_type=='bundle' and (bundle_prune is not None):
            assert bundle_prune in ['lambda','svd','log_lambda','log_svd','svd2','duals']

            print('Preprocessing bundle with {}.'.format(bundle_prune), flush=True)

            def active_from_vec(rank,vec):
                if self.proj_hess:
                    rank = min(rank,self.x_dim)
                return np.argsort(vec)[-rank:]

            def geo_gap(vec,exclude_first=True):
                sorted = np.argsort(abs(np.diff(np.log10(np.sort(vec)[::-1]))))
                ind = -2 if exclude_first else -1
                return sorted[ind] + 1

            if bundle_prune == 'svd':
                sig = np.linalg.svd(self.dfS,compute_uv=False)
                rank = int(1 * sum(sig > max(sig)*self.rank_thres))
                active = active_from_vec(rank,warm_start['duals'])
            elif bundle_prune == 'svd2':
                sig = np.linalg.svd(np.concatenate((self.dfS, np.ones(self.k)[:, np.newaxis]), axis=1),compute_uv=False)
                rank = int(1 * sum(sig > max(sig) * self.rank_thres))
                active = active_from_vec(rank,warm_start['duals'])
            elif bundle_prune == 'duals':
                assert k is not None
                rank = k
                active = active_from_vec(rank,warm_start['duals'])
            elif bundle_prune == 'lambda':
                _, tmp_lam = get_lam(self.dfS, solver=self.solver)
                rank = sum(tmp_lam > self.rank_thres * max(tmp_lam))
                active = active_from_vec(rank, tmp_lam)
            elif bundle_prune == 'log_lambda':
                _, tmp_lam = get_lam(self.dfS, solver=self.solver)
                rank   = geo_gap(tmp_lam, exclude_first=True)
                active = active_from_vec(rank, tmp_lam)
            elif bundle_prune == 'log_svd':
                sig = np.linalg.svd(self.dfS,compute_uv=False)
                rank   = geo_gap(sig, exclude_first=True)
                active = active_from_vec(rank,warm_start['duals'])

            self.S     = self.S[active, :]
            self.fS    = self.fS[active]
            self.dfS   = self.dfS[active,:]
            self.d2fS  = self.d2fS[active,:]

        # Set params
        self.cur_delta, self.lam_cur = get_lam(self.dfS, solver=self.solver)
        self.update_k()
        self.update_params()

    def step(self):

        super(NewtonBundle, self).step()

        G  = self.D @ self.dfS # See Lewis-Wylie (2019)
        b_l = self.D@(np.einsum('ij,ij->i',self.dfS,self.S) - self.fS)

        hessian = np.einsum('s,sij->ij', self.lam_cur, self.d2fS)
        if self.store_hessian:
            self.hessian = hessian.copy()

        if self.proj_hess: # Project hessian. See Lewis-Wylie 2019
            Q, R    = np.linalg.qr(G.T, mode='complete')
            V = Q[:,:(self.k-1)]
            U = Q[:,(self.k-1):]

            p = V @ np.linalg.inv(G@V) @ b_l

            UhU = np.stack([U.T @ self.d2fS[i,:,:] @ U for i in range(self.k)])

            A = np.einsum('s,sij->ij',self.lam_cur,UhU)
            b1 = np.einsum('s,sij,jk,ks->i',self.lam_cur,UhU,U.T,(self.S - p).T)
            b2 = np.einsum('s,ij,js->i',self.lam_cur,U.T,self.dfS.T)
            b  = b1 - b2

            xu = pinv2(A, rcond=self.pinv_cond) @ b
            self.cur_x = U@xu + p
        else:
            A = np.zeros([self.x_dim+self.k,self.x_dim+self.k])

            A[0:self.x_dim,0:self.x_dim]=hessian
            A[0:self.x_dim,self.x_dim:(self.x_dim+self.k)] = self.dfS.T
            A[self.x_dim,self.x_dim:(self.x_dim+self.k)]   = 1
            A[(self.x_dim+1):, 0:self.x_dim]               = G

            b =  np.zeros(self.x_dim+self.k)
            b[0:self.x_dim] = np.einsum('s,sij,sj->i',self.lam_cur,self.d2fS,self.S)
            b[self.x_dim]   = 1
            b[self.x_dim+1:] = b_l

            self.cur_x = (pinv2(A, cond=self.pinv_cond) @ b)[0:self.x_dim]

        # self.vio = np.linalg.norm(A @ self.cur_x - b)

        # optimality check
        # self.opt_check(A, b)

        # Get current gradient and hessian
        oracle = self.objective.call_oracle(self.cur_x)
        old_fx = self.cur_fx.copy() if (self.cur_fx is not None) else float('inf')
        self.cur_fx  = oracle['f']
        self.fx_step = (old_fx - self.cur_fx)

        # Combinatorially find leaving index
        conv_size = lambda i : get_lam(self.dfS,sub_ind=i,new_df=oracle['df'],solver=self.solver)
        jobs = Parallel(n_jobs=min(multiprocessing.cpu_count(),self.k))(delayed(conv_size)(i) for i in range(self.k))

        jobs_delta = [jobs[i][0] for i in range(self.k)]
        k_sub = np.argmin(jobs_delta)

        if jobs_delta[k_sub] > self.cur_delta and self.adaptive_bundle:
            self.S    = np.concatenate((self.S, self.cur_x[np.newaxis]))
            self.fS   = np.concatenate((self.fS, self.cur_fx[np.newaxis]))
            self.dfS  = np.concatenate((self.dfS, oracle['df'][np.newaxis]))
            self.d2fS = np.concatenate((self.d2fS, oracle['d2f'][np.newaxis]))
            self.update_k()

            old_delta = self.cur_delta.copy()

            self.cur_delta, self.lam_cur = get_lam(self.dfS,solver=self.solver)

            if self.cur_delta > old_delta:
                raise Exception('delta increased')
        else:
            self.lam_cur = jobs[k_sub][1]

            self.S[k_sub, :] = self.cur_x
            self.fS[k_sub]   = self.cur_fx
            self.dfS[k_sub, :] = oracle['df']
            self.d2fS[k_sub, :, :] = oracle['d2f']

        self.cur_delta = np.linalg.norm(self.lam_cur @ self.dfS)

        # print(self.cur_delta, flush=True)

        # Update current iterate value and update the bundle
        self.update_params()

    def update_params(self):

        self.cur_diam = np.array(get_diam(self.S))
        print('Diam: {}'.format(self.cur_diam),flush=True)
        print('Delta: {}'.format(self.cur_delta), flush=True)

        if self.path_x is not None:
            self.path_x = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
            self.path_diam = np.concatenate((self.path_diam, self.cur_diam[np.newaxis]))
            self.path_delta = np.concatenate((self.path_delta, self.cur_delta[np.newaxis]))
            # self.path_vio = np.concatenate((self.path_vio, self.cur_delta[np.newaxis]))

            if self.store_hessian:
                hess_spec = torch.svd(self.hessian,compute_uv=False)[1].data.numpy()
                self.path_hess = np.concatenate((self.path_hess, hess_spec[np.newaxis]))
        else:
            self.path_x = self.cur_x[np.newaxis]
            self.path_fx = self.cur_fx[np.newaxis]
            self.path_diam = self.cur_diam[np.newaxis]
            self.path_delta = self.cur_delta[np.newaxis]
            # self.path_vio = self.cur_vio[np.newaxis]

            if self.store_hessian:
                hess_spec = torch.svd(self.hessian,compute_uv=False).data.numpy()
                self.path_hess = hess_spec[np.newaxis]

        # Update paths and bundle constraints
        self.cur_iter += 1

    def stop_cond(self):

        return ((self.cur_delta < self.delta_thres)
                and (self.cur_diam < self.diam_thres))

    def opt_check(self, A, b):
            mu  = (np.linalg.pinv(A,rcond=self.pinv_cond) @ b)[self.x_dim:self.x_dim+self.k]
            tmp = np.zeros(self.k)
            tmp2 = np.zeros(self.x_dim)
            for i in range(self.k):
                tmp[i] = self.fS[i] + self.dfS[i,:]@(self.cur_x - self.S[i,:])
                tmp2 += self.lam_cur[i] * self.d2fS[i] @ (self.cur_x - self.S[i,:])
                tmp2 += mu[i] * self.dfS[i]

            assert np.all([np.isclose(tmp[0], val) for val in tmp]) # Check active set
            assert np.isclose(np.linalg.norm(tmp2),0) # Check first order cond
            assert np.isclose(sum(mu),1) # Check duals

    def update_k(self):
        self.k = self.S.shape[0]
        self.D = diags([1, -1], offsets=[0, 1], shape=(self.k - 1, self.k)).toarray()

        print('Bundle Size Set to {}'.format(self.k), flush=True)

        self.name = 'NewtonBundle (bund-sz=' + str(self.k)
        if self.proj_hess:
            self.name += ' U-projected'
        self.name += ')'

# Combinatorially find leaving index
def get_lam(dfS,sub_ind=None,new_df=None, solver='MOSEK'):
    k = dfS.shape[0]

    dfS_ = dfS.copy()
    if sub_ind is not None:
        dfS_[sub_ind] = new_df
    Q = dfS_ @ dfS_.T

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
        return np.linalg.norm(lam.value@dfS_), lam.value.copy()
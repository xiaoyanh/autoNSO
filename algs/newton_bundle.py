import torch
import numpy as np
import cvxpy as cp
import multiprocessing

from IPython import embed
from algs.optAlg import OptAlg
from scipy.sparse import diags
from joblib import Parallel, delayed

tol = 1e-15
m_params = {'MSK_DPAR_INTPNT_QO_TOL_DFEAS': tol,
            'MSK_DPAR_INTPNT_QO_TOL_INFEAS': tol,
            'MSK_DPAR_INTPNT_QO_TOL_MU_RED': tol,
            'MSK_DPAR_INTPNT_QO_TOL_NEAR_REL': 10,
            'MSK_DPAR_INTPNT_QO_TOL_PFEAS': tol,
            'MSK_DPAR_INTPNT_QO_TOL_REL_GAP': tol,
            }

# Subgradient method
class NewtonBundle(OptAlg):
    def __init__(self, objective, k=4, delta_thres=0, diam_thres=0, proj_hess=False, warm_start=None, start_type='bundle', **kwargs):
        objective.oracle_output='hess+'

        super(NewtonBundle, self).__init__(objective, **kwargs)

        # Set up criterion
        self.criterion = self.objective.obj_func

        # Add start with initial point
        self.delta_thres = delta_thres
        self.diam_thres  = diam_thres
        self.proj_hess   = proj_hess

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
                self.S = self.cur_x + np.random.randn(self.k, self.x_dim) * np.linalg.norm(self.cur_x) * 1e-1
            else:
                raise Exception('Start type must me bundle or random')

            self.path_x = np.zeros([self.cur_iter, self.x_dim]) * np.nan
            self.path_fx = np.zeros([self.cur_iter]) * np.nan
            self.path_diam = np.zeros([self.cur_iter]) * np.nan
            self.path_delta = np.zeros([self.cur_iter]) * np.nan


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
        if warm_start and start_type=='bundle':
            # sig = np.linalg.svd(self.dfS,compute_uv=False)
            # rank = min(int(1 * sum(sig > max(sig)*1e-4)),self.dfS.shape[0])

            # active = get_active(self.dfS, rank=rank)
            # active = np.argsort(warm_start['duals'])[-rank:]

            _, tmp_lam = get_lam(self.dfS)
            # tmp_lam *= np.linalg.norm(self.dfS,axis=1)
            # active  = tmp_lam.argsort()[-int(rank):]
            active = np.where(tmp_lam > 1e-3*max(tmp_lam))[0]

            # print('Solving MIP for rank {} subset'.format(rank), flush=True)
            # _, _, active = get_lam_MIP(self.dfS,rank=rank)
            # print('MIP Solved', flush=True)

            self.k     = len(active)
            self.S     = self.S[active, :]
            self.fS    = self.fS[active]
            self.dfS   = self.dfS[active,:]
            self.d2fS  = self.d2fS[active,:]

        if self.proj_hess: # difference matrix for projection
            self.D = diags([1,-1],offsets=[0,1],shape=(self.k-1,self.k)).toarray()

        self.cur_delta, self.lam_cur = get_lam(self.dfS)

        self.name = 'NewtonBundle (bund_sz=' + str(self.k) + ')'

        self.update_params()

    def step(self):

        super(NewtonBundle, self).step()

        G = self.D @ self.dfS # See Lewis-Wylie (2019)

        if self.proj_hess: # Project hessian
            u, d, vh = np.linalg.svd(G)
            null_dim = max((self.x_dim - self.k), 0) + sum(d < 1e-3 * max(d))
            U = vh[-null_dim:, :].T

            P = U @ U.T
            hess = np.stack([P.T @ self.d2fS[i,:,:] @ P for i in range(self.k)])
        else:
            hess = self.d2fS

        A = np.zeros([self.x_dim+self.k,self.x_dim+self.k])
        top_left = np.einsum('s,sij->ij',self.lam_cur,hess)

        A[0:self.x_dim,0:self.x_dim]=top_left
        A[0:self.x_dim,self.x_dim:(self.x_dim+self.k)] = self.dfS.T
        A[self.x_dim,self.x_dim:(self.x_dim+self.k)]   = 1
        A[(self.x_dim+1):, 0:self.x_dim]               = G

        b =  np.zeros(self.x_dim+self.k)
        b[0:self.x_dim] = np.einsum('s,sij,sj->i',self.lam_cur,hess,self.S)
        b[self.x_dim]   = 1
        b[self.x_dim+1:] = self.D@(np.einsum('ij,ij->i',self.dfS,self.S) - self.fS)
        self.cur_x = (np.linalg.pinv(A, rcond=1e-12) @ b)[0:self.x_dim]

        # optimality check
        # self.opt_check(A, b)

        # Get current gradient and hessian
        oracle = self.objective.call_oracle(self.cur_x)
        old_fx = self.cur_fx.copy() if (self.cur_fx is not None) else float('inf')
        self.cur_fx  = oracle['f']
        self.fx_step = (old_fx - self.cur_fx)

        # Combinatorially find leaving index
        conv_size = lambda i : get_lam(self.dfS,sub_ind=i,new_df=oracle['df'])
        jobs = Parallel(n_jobs=min(multiprocessing.cpu_count(),self.k))(delayed(conv_size)(i) for i in range(self.k))
        jobs_delta = [jobs[i][0] for i in range(self.k)]
        k_sub = np.argmin(jobs_delta)
        self.lam_cur = jobs[k_sub][1]

        self.S[k_sub, :] = self.cur_x
        self.fS[k_sub]   = self.cur_fx
        self.dfS[k_sub, :] = oracle['df']
        self.d2fS[k_sub, :, :] = oracle['d2f']

        # Update current iterate value and update the bundle
        self.update_params()

    def update_params(self):

        self.cur_diam = max(np.linalg.norm(self.S, axis=1))

        if self.path_x is not None:
            self.path_x = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
            self.path_diam = np.concatenate((self.path_diam, self.cur_diam[np.newaxis]))
            self.path_delta = np.concatenate((self.path_delta, self.cur_delta[np.newaxis]))
        else:
            self.path_x = self.cur_x[np.newaxis]
            self.path_fx = self.cur_fx[np.newaxis]
            self.path_diam = self.cur_diam[np.newaxis]
            self.path_delta = self.cur_delta[np.newaxis]

        # Update paths and bundle constraints
        self.cur_iter += 1

    def stop_cond(self):

        return ((self.cur_delta < self.delta_thres)
                and (self.cur_diam < self.diam_thres))

    def opt_check(self, A, b):
            mu  = (np.linalg.pinv(A,rcond=1e-12) @ b)[self.x_dim:self.x_dim+self.k]
            tmp = np.zeros(self.k)
            tmp2 = np.zeros(self.x_dim)
            for i in range(self.k):
                tmp[i] = self.fS[i] + self.dfS[i,:]@(self.cur_x - self.S[i,:])
                tmp2 += self.lam_cur[i] * self.d2fS[i] @ (self.cur_x - self.S[i,:])
                tmp2 += mu[i] * self.dfS[i]

            assert np.all([np.isclose(tmp[0], val) for val in tmp]) # Check active set
            assert np.isclose(np.linalg.norm(tmp2),0) # Check first order cond
            assert np.isclose(sum(mu),1) # Check duals

# Combinatorially find leaving index
def get_lam(dfS,sub_ind=None,new_df=None):
    k = dfS.shape[0]
    xdim = dfS.shape[1]
    dfS2 = dfS.copy()

    p_tmp = cp.Variable(xdim)
    lam   = cp.Variable(k)

    if sub_ind is not None:
        dfS2[sub_ind] = new_df

    constraints = [cp.sum(lam) == 1.0]
    constraints += [lam >= 0]

    # Find lambda (warm start with previous iteration)
    prob = cp.Problem(cp.Minimize(cp.quad_form(p_tmp, np.eye(xdim))),
                      constraints + [lam @ dfS2 == p_tmp])

    prob.solve(warm_start=True, solver=cp.MOSEK, mosek_params=m_params)
    # prob.solve(solver=cp.GUROBI)

    return np.sqrt(prob.value), lam.value.copy()

# Combinatorially find leaving index
def get_lam_MIP(dfS, new_df=None,rank=None):

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

    prob.solve(warm_start=True, solver=cp.MOSEK, mosek_params=m_params)
    # prob.solve(solver=cp.GUROBI, verbose=True)

    if (rank is not None) or (new_df is not None):
        keep_inds = np.where(non_zero.value > 0)
        return np.sqrt(prob.value), lam.value.copy(), keep_inds[0]
    else:
        return np.sqrt(prob.value), lam.value.copy()

# Combinatorially find leaving index
def get_active(dfS, rank=None):

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

    prob.solve(warm_start=True, solver=cp.MOSEK, mosek_params=m_params)
    # prob.solve(solver=cp.GUROBI, verbose=True)

    return np.where(non_zero.value > 0)[0]
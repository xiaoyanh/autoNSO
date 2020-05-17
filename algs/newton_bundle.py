import torch
import numpy as np
import cvxpy as cp
import multiprocessing

from IPython import embed
from algs.optAlg import OptAlg
from joblib import Parallel, delayed

# Subgradient method
class NewtonBundle(OptAlg):
    def __init__(self, objective, k=4, delta_thres=0, diam_thres=0, warm_start=None, **kwargs):
        objective.oracle_output='hess+'

        super(NewtonBundle, self).__init__(objective, **kwargs)

        # Set up criterion
        self.criterion = self.objective.obj_func

        # Add start with initial point
        self.delta_thres = delta_thres
        self.diam_thres  = diam_thres

        # Prepare the bundle
        if warm_start is not None:
            self.cur_x  = warm_start['x']
            self.S      = warm_start['bundle']
            self.cur_iter = warm_start['iter']
            self.k = self.S.shape[0]

            self.x0 = None
            self.x_dim = len(self.cur_x)

            self.path_x  = np.zeros([self.cur_iter, self.x_dim]) * np.nan
            self.path_fx = np.zeros([self.cur_iter]) * np.nan
            self.path_diam = np.zeros([self.cur_iter]) * np.nan
            self.path_delta = np.zeros([self.cur_iter]) * np.nan
        else:
            self.cur_x = self.x0
            self.S = None
            self.k = k  # bundle size
        self.cur_fx = self.criterion(torch.tensor(self.cur_x, dtype=torch.double, requires_grad=False)).data.numpy()

        self.name = 'NewtonBundle (k=' + str(self.k) + ')'

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

        self.cur_delta, self.lam_cur = get_lam(self.dfS)
        self.update_params()

    def step(self):

        super(NewtonBundle, self).step()

        A = np.zeros([self.x_dim+1+self.k,self.x_dim+1+self.k])
        top_left = np.einsum('s,sij->ij',self.lam_cur,self.d2fS)

        A[0:self.x_dim,0:self.x_dim]=top_left
        A[0:self.x_dim,self.x_dim:(self.x_dim+self.k)] = self.dfS.T
        A[self.x_dim,self.x_dim:(self.x_dim+self.k)]   = 1
        A[(self.x_dim+1):, 0:self.x_dim]               = self.dfS
        A[(self.x_dim+1):,-1]                          = -1

        b =  np.zeros(self.x_dim+1+self.k)
        b[0:self.x_dim] = np.einsum('s,sij,sj->i',self.lam_cur,self.d2fS,self.S)
        b[self.x_dim]   = 1
        b[self.x_dim+1:] = np.einsum('ij,ij->i',self.dfS,self.S) - self.fS

        self.cur_x = (np.linalg.pinv(A,rcond=1e-12) @ b)[0:self.x_dim]

        # optimality check
        # self.opt_check(A, b)

        # Get current gradient and hessian
        oracle = self.objective.call_oracle(self.cur_x)
        self.cur_fx = oracle['f']

        # k_sub = np.argmax(np.linalg.norm(self.S, axis=1))
        # k_sub = np.argmin(self.lam_cur)
        # k_sub = np.argmin(self.lam_cur*np.linalg.norm(self.S, axis=1))

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

    constraints = [np.ones(k) @ lam == 1]
    constraints += [lam >= 0]

    # Find lambda (warm start with previous iteration)
    prob = cp.Problem(cp.Minimize(cp.quad_form(p_tmp, np.eye(xdim))),
                      constraints + [lam @ dfS2 == p_tmp])
    prob.solve(solver=cp.GUROBI)

    return np.sqrt(prob.value), lam.value
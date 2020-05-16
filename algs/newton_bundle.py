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
        self.cur_lam = None
        self.delta   = float('inf')
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
        else:
            self.cur_x = self.x0
            self.S = None
            self.k = k  # bundle size

        self.name = 'NewtonBundle (k=' + str(self.k) + ')'

        self.cur_fx = self.criterion(torch.tensor(self.cur_x, dtype=torch.double, requires_grad=False)).data.numpy()

        if self.S is None: # If bundle is none, randomly initialize it (k * n)
            self.S = np.zeros([self.k,self.x_dim])
            self.S[0,:] = self.x0
            if self.k > 1:
                self.S[1:,:] = self.x0 + np.random.randn(self.k-1,self.x_dim)

        self.update_params()

        # Add higher order info results
        self.fS   = np.zeros(self.k)
        self.dfS  = np.zeros([self.k,self.x_dim])
        self.d2fS = np.zeros([self.k,self.x_dim,self.x_dim])
        for i in range(self.k):
            oracle = self.objective.call_oracle(self.S[i,:])
            self.fS[i]   = oracle['f']
            self.dfS[i,:]  = oracle['df']
            self.d2fS[i,:,:] = oracle['d2f']

        # Set up CVX
        self.p              = cp.Variable(self.x_dim)
        self.lam_var        = cp.Variable(self.k)
        self.constraints    = [np.ones(self.k)@self.lam_var == 1]
        self.constraints    += [self.lam_var >= 0]

    def step(self):

        super(NewtonBundle, self).step()

        # Find lambda (warm start with previous iteration)
        self.lam_var.value = self.cur_lam
        prob = cp.Problem(cp.Minimize(cp.quad_form(self.p,np.eye(self.x_dim))), self.constraints+[self.lam_var @ self.dfS == self.p])
        prob.solve(warm_start=True, solver=cp.GUROBI)
        self.lam_cur = self.lam_var.value.copy()

        # Solve optimality conditions for x
        self.delta = np.sqrt(prob.value)

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

        # Check optimality conditions
        # if self.cur_iter == 26:
        #     self.cur_x = (np.linalg.pinv(A,rcond=1e-12) @ b)[0:self.x_dim]
        #     mu  = (np.linalg.pinv(A,rcond=1e-12) @ b)[self.x_dim:self.x_dim+self.k]
        #     tmp = np.zeros(self.k)
        #     tmp2 = np.zeros(self.x_dim)
        #     for i in range(self.k):
        #         tmp[i] = self.fS[i] + self.dfS[i,:]@(self.cur_x - self.S[i,:])
        #         tmp2 += self.lam_cur[i] * self.d2fS[i] @ (self.cur_x - self.S[i,:])
        #         tmp2 += mu[i] * self.dfS[i]
        #
        #     assert np.all([np.isclose(tmp[0], val) for val in tmp]) # Check active set
        #     assert np.isclose(np.linalg.norm(tmp2),0) # Check first order cond
        #     assert np.isclose(sum(mu),1) # Check duals

        # Get current gradient and hessian
        oracle = self.objective.call_oracle(self.cur_x)
        self.cur_fx = oracle['f']

        # k_sub = np.argmax(np.linalg.norm(self.S, axis=1))
        # k_sub = np.argmin(self.lam_cur)
        # k_sub = np.argmin(self.lam_cur*np.linalg.norm(self.S, axis=1))

        # Combinatorially find leaving index
        def conv_size(i,dfS,xdim,k,new_df):
            dfS2 = dfS.copy()

            p_tmp = cp.Variable(xdim)
            lam   = cp.Variable(k)
            dfS2[i] = new_df

            self.constraints = [np.ones(k) @ lam == 1]
            self.constraints += [lam >= 0]

            # Find lambda (warm start with previous iteration)
            prob = cp.Problem(cp.Minimize(cp.quad_form(p_tmp, np.eye(xdim))),
                              self.constraints + [lam @ dfS2 == p_tmp])
            prob.solve(solver=cp.GUROBI)

            return prob.value
        jobs = Parallel(n_jobs=min(multiprocessing.cpu_count(),self.k))(delayed(conv_size)(i,self.dfS,self.x_dim,self.k,oracle['df']) for i in range(self.k))
        k_sub = np.argmin(jobs)

        self.S[k_sub, :] = self.cur_x
        self.fS[k_sub]   = self.cur_fx
        self.dfS[k_sub, :] = oracle['df']
        self.d2fS[k_sub, :, :] = oracle['d2f']

        # Update current iterate value and update the bundle

        self.update_params()

    def update_params(self):

        if self.path_x is not None:
            self.path_x = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
        else:
            self.path_x = self.cur_x[np.newaxis]
            self.path_fx = self.cur_fx[np.newaxis]

        self.diam = max(np.linalg.norm(self.S, axis=1))

        # Update paths and bundle constraints
        self.cur_iter += 1

    def stop_cond(self):
        return (self.delta < self.delta_thres) and (self.diam < self.diam_thres)
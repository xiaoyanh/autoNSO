import os
import numpy as np
import matlab.engine
import multiprocessing

from IPython import embed
from utils.pinv import pinv2
from algs.optAlg import OptAlg
from scipy.sparse import diags
from utils.diameter import get_diam
from algs.newton_bundle_aux.aug_bund import create_bundle
from algs.newton_bundle_aux.get_leaving import get_leaving
from algs.newton_bundle_aux.get_lambda import get_lam, get_LS

# Bundle Newton Method from Lewis-Wylie 2019
class NewtonBundle(OptAlg):
    def __init__(self, objective, k=4, delta_thres=0, diam_thres=0, proj_hess=False, warm_start=None, start_type='bundle',
                 bundle_prune='lambda', rank_thres=1e-3, pinv_cond=float('-inf'), random_sz=1e-1,
                 store_hessian=False, leaving_met='delta', solver='MOSEK', adaptive_bundle=False, **kwargs):
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
        self.store_hessian = store_hessian
        self.leaving_met = leaving_met
        self.adaptive_bundle = adaptive_bundle
        self.k = k

        self.solver = solver
        assert solver in ['MOSEK','GUROBI','OSQP','CVXOPT','quadprog','MATLAB']
        assert leaving_met in ['delta','ls']

        if self.solver == 'MATLAB':
            print("Starting parallel pool for MATLAB solver", flush=True)

            threads = multiprocessing.cpu_count()/2
            self.eng = matlab.engine.start_matlab()
            self.eng.parpool('local', threads)
            self.eng.addpath(os.getcwd() + '/algs/newton_bundle_aux', nargout=0)
            print('Started!', flush=True)
        else:
            self.eng=None

        print("Project Hessian: {}".format(self.proj_hess),flush=True)

        # Prepare the bundle
        if warm_start is None:
            self.cur_x = self.x0
            self.S = None
        else:
            self.cur_x      = warm_start['x']
            self.cur_iter   = warm_start['iter']
            self.x0         = None
            self.x_dim      = len(self.cur_x)

            if start_type == 'bundle':
                self.S      = warm_start['bundle']
            elif start_type == 'random':
                self.S = self.cur_x + np.random.randn(self.k, self.x_dim) * np.linalg.norm(self.cur_x) * self.random_sz
            else:
                raise Exception('Start type must me bundle or random')

            self.path_x = np.zeros([self.cur_iter+1, self.x_dim]) * np.nan
            self.path_fx = np.zeros([self.cur_iter+1]) * np.nan
            self.path_diam = np.zeros([self.cur_iter+1]) * np.nan
            self.path_delta = np.zeros([self.cur_iter+1]) * np.nan
            # self.path_vio   = np.zeros([self.cur_iter]) * np.nan

            if self.store_hessian:
                self.path_hess = np.zeros([self.cur_iter+1, self.x_dim]) * np.nan

        oracle = self.objective.call_oracle(self.cur_x)
        self.cur_fx = oracle['f']

        if self.store_hessian:
            self.hessian = oracle['d2f']

        if self.S is None: # If bundle is none, randomly initialize it (k * n)
            assert self.k is not None
            self.S = np.zeros([self.k,self.x_dim])
            self.S[0,:] = self.x0
            if self.k > 1:
                self.S[1:,:] = self.x0 + np.random.randn(self.k-1,self.x_dim)
        elif (self.k is not None) and self.S.shape[0] < self.k:
            self.S = np.concatenate((self.S,np.random.randn(self.k - self.S.shape[0], self.x_dim)))
        elif self.k is None:
            self.k = self.S.shape[0]

        # Add higher order info results
        self.fS   = np.zeros(self.k)
        self.dfS  = np.zeros([self.k,self.x_dim])
        self.d2fS = np.zeros([self.k,self.x_dim,self.x_dim])

        create_bundle(self, bundle_prune, self.k, warm_start)

        # Set params
        self.cur_delta, self.lam_cur = get_lam(self.dfS, solver=self.solver, eng=self.eng)
        self.update_k()

        self.update_params()

    def step(self):

        super(NewtonBundle, self).step()

        G  = self.D @ self.dfS # See Lewis-Wylie (2019)
        b_l = self.D@(np.einsum('ij,ij->i',self.dfS,self.S) - self.fS)

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

            A[0:self.x_dim,0:self.x_dim] = np.einsum('s,sij->ij', self.lam_cur, self.d2fS)
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

        if self.store_hessian:
            self.hessian = oracle['d2f']

        self.update_bundle(oracle) # Update the bundle

        # Update current iterate value and update the bundle
        self.update_params()

    def update_params(self):

        self.cur_diam = np.array(get_diam(self.S))
        self.cur_delta = np.linalg.norm(self.lam_cur @ self.dfS)

        print('Diam: {}'.format(self.cur_diam),flush=True)
        print('Delta: {}'.format(self.cur_delta), flush=True)

        if self.path_x is not None:

            # Update paths and bundle constraints
            self.cur_iter += 1

            self.path_x = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
            self.path_diam = np.concatenate((self.path_diam, self.cur_diam[np.newaxis]))
            self.path_delta = np.concatenate((self.path_delta, self.cur_delta[np.newaxis]))
            # self.path_vio = np.concatenate((self.path_vio, self.cur_delta[np.newaxis]))

            if self.store_hessian:
                self.path_hess = np.concatenate((self.path_hess, np.linalg.svd(self.hessian,compute_uv=False)[np.newaxis]))
        else: # First iteration, do not increment
            self.path_x = self.cur_x[np.newaxis]
            self.path_fx = self.cur_fx[np.newaxis]
            self.path_diam = self.cur_diam[np.newaxis]
            self.path_delta = self.cur_delta[np.newaxis]
            # self.path_vio = self.cur_vio[np.newaxis]

            if self.store_hessian:
                self.path_hess = np.linalg.svd(self.hessian,compute_uv=False)[np.newaxis]

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

        self.name = 'NewtonBundle (bund-sz=' + str(self.k)
        if self.proj_hess:
            self.name += ' U-projected'
        self.name += ')'

        print('Bundle Size Set to {}'.format(self.k), flush=True)

    def update_bundle(self, oracle):
        k_sub = get_leaving(self, oracle) # Finding leaving index
        if k_sub:
            self.S[k_sub, :] = self.cur_x
            self.fS[k_sub]   = self.cur_fx
            self.dfS[k_sub, :] = oracle['df']
            self.d2fS[k_sub, :, :] = oracle['d2f']

        if self.leaving_met == 'ls':
            _, self.lam_cur = get_lam(self.dfS, solver=self.solver, eng=self.eng)
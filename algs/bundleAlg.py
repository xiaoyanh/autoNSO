import os
import numpy as np
import matlab.engine
import multiprocessing

from IPython import embed
from algs.optAlg import OptAlg
from utils.diameter import get_diam
from algs.newton_bundle_aux.get_lambda import get_lam
from algs.newton_bundle_aux.get_leaving import get_leaving
from algs.newton_bundle_aux.approx_hessian import hess_approx_cI
from algs.newton_bundle_aux.aug_bund import create_bundle

# A Bundle Method
class BundleAlg(OptAlg):
    def __init__(self, objective, k=4, delta_thres=0, diam_thres=0, warm_start=None, start_type='bundle',
                 bundle_prune='lambda', rank_thres=1e-3, pinv_cond=float('-inf'), random_sz=1e-1,
                 leaving_met='delta', solver='MOSEK', eng = None, mu_sz=None, adaptive_bundle=False, **kwargs):

        super(BundleAlg, self).__init__(objective, **kwargs)

        # Set up criterion
        self.criterion = self.objective.obj_func

        # Add start with initial point
        self.delta_thres = delta_thres
        self.diam_thres  = diam_thres
        self.rank_thres  = rank_thres
        self.pinv_cond   = pinv_cond
        self.random_sz   = random_sz
        self.leaving_met = leaving_met
        self.adaptive_bundle = adaptive_bundle
        self.k = k
        self.mu = mu_sz

        self.solver = solver

        self.start_matlab(eng)

        # Prepare the bundle
        if warm_start is None:
            self.cur_x = self.x0
            self.S = None
            self.start_iter = 0
        else:
            if type(warm_start) == list:
                assert len(warm_start) == 1
                warm_start = warm_start[0]

            self.cur_x      = warm_start['x']
            self.cur_iter   = warm_start['iter']
            self.start_iter = warm_start['iter']
            self.x0         = None
            self.x_dim      = len(self.cur_x)

            if start_type == 'bundle':
                self.S      = warm_start['bundle']
            elif start_type == 'random':
                self.S = self.cur_x + np.random.randn(self.k, self.x_dim) * np.linalg.norm(self.cur_x) * self.random_sz
            else:
                raise Exception('Start type must me bundle or random')

            self.create_paths()

        create_bundle(self, bundle_prune,  warm_start, start_type)
        self.update_k()

        # Set params
        self.cur_delta, self.lam_cur = get_lam(self.dfS, solver=self.solver, eng=self.eng)
        self.lam_cur = self.lam_cur.reshape(-1)

        self.post_step(intermediate=False)

    def step(self):
        super(BundleAlg, self).step()

    def post_step(self, intermediate=True):
        self.oracle = self.objective.call_oracle(self.cur_x)
        self.cur_fx = self.oracle['f']

        # Compare with convex combination
        self.cur_x_conv = self.lam_cur @ self.S
        self.cur_fx_conv = self.objective.obj_func(self.cur_x_conv).item()

        if intermediate:
            old_fx = self.cur_fx.copy() if (self.cur_fx is not None) else float('inf')
            self.fx_step = (old_fx - self.cur_fx)

            self.update_bundle()  # Update the bundle

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

            self.path_x = np.vstack([self.path_x, self.cur_x])
            self.path_fx = np.vstack([self.path_fx, self.cur_fx])
            self.path_fx_conv = np.vstack([self.path_fx_conv, self.cur_fx_conv])
            self.path_diam = np.vstack([self.path_diam, self.cur_diam])
            self.path_delta = np.vstack([self.path_delta, self.cur_delta])
            self.path_conv_diff = np.vstack([self.path_conv_diff, abs(self.cur_fx_conv-self.cur_fx)])

    def create_paths(self):
        self.path_x = np.zeros([self.cur_iter + 1, self.x_dim]) * np.nan
        self.path_fx = (np.zeros([self.cur_iter + 1]) * np.nan).reshape(self.cur_iter + 1, 1)
        self.path_fx_conv = (np.zeros([self.cur_iter + 1]) * np.nan).reshape(self.cur_iter + 1, 1)
        self.path_diam = (np.zeros([self.cur_iter + 1]) * np.nan).reshape(self.cur_iter + 1, 1)
        self.path_delta = (np.zeros([self.cur_iter + 1]) * np.nan).reshape(self.cur_iter + 1, 1)
        self.path_conv_diff = (np.zeros([self.cur_iter + 1]) * np.nan).reshape(self.cur_iter + 1, 1)
        # self.path_vio   = np.zeros([self.cur_iter]) * np.nan

    def stop_cond(self):

        return ((self.cur_delta < self.delta_thres)
                and (self.cur_diam < self.diam_thres))

    def update_k(self):
        pass

    def update_bundle(self):
        if self.leaving_met != 'delta':
            eng_tmp = self.eng
            delattr(self,'eng')

        k_sub = get_leaving(self,self.oracle)

        if self.leaving_met != 'delta':
            self.eng = eng_tmp

        if k_sub is not None:
            self.S[k_sub, :] = self.cur_x
            self.fS[k_sub]   = self.cur_fx
            self.dfS[k_sub, :] = self.oracle['df']

            if self.objective.oracle_output == 'hess+':
                self.d2fS[k_sub, :, :] = hess_approx_cI(self.oracle['d2f'], sig_type=self.hessian_type, mu=self.mu)

        if self.leaving_met == 'ls':
            _, self.lam_cur = get_lam(self.dfS, solver=self.solver, eng=self.eng)

    def start_matlab(self, eng):
        if self.solver == 'MATLAB':
            if eng is None:
                print("Starting parallel pool for MATLAB solver", flush=True)

                threads = multiprocessing.cpu_count()/2
                self.eng = matlab.engine.start_matlab()
                self.eng.parpool('local', threads)
                self.eng.addpath(os.getcwd() + '/algs/newton_bundle_aux', nargout=0)
                print('MATLAB Started!', flush=True)
            else:
                self.eng = eng
        else:
            self.eng=None
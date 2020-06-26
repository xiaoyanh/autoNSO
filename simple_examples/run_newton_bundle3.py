#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import numpy as np
import matlab.engine
import multiprocessing
from IPython import embed
from algs.torch_alg import BFGS
from vis.visualize import OptPlot
from algs.prox_bundle import ProxBundle
from algs.newton_bundle import NewtonBundle
from obj.obj_funcs import stronglyconvex, nonconvex, partlysmooth, halfandhalf, PartlySmooth2D, PartlySmooth3D, Convex3D
from obj.obj_funcs2 import Matrix4D

# Run newton-bundle optimization algorithm
n = 50
m = 25

# obj_type = 'Strongly Convex'
# obj_type = 'Partly Smooth'
# obj_type = 'Matrix 4D'
# obj_type = 'Half-and-Half'
# obj_type = 'Partly Smooth 2D'
# obj_type = 'Partly Smooth 3D'
obj_type = 'Convex 3D'
# Criteria for switching to newton-bundle

def crit_c3(met):
    return (met.cur_fx is not None) and (met.cur_fx < 1)

proj_hess = False
pinv_cond = 1e-16
rank_thres = None
bfgs_lr = 0.1
rescaled  = False
bundle_prune = 'duals'
fixed_shift = 0.0
naive_prune = False
leaving_met = 'delta'
solver = 'MOSEK'

if obj_type == 'Strongly Convex':
    titl = obj_type + ': {}-dimensional, max over {} quartics'.format(n, 10)
    objective = stronglyconvex(n=n,k=10,oracle_output='both'); mu_sz=1e3; beta_sz=1e-5; iters=125
    k = None
    bfgs_lr = 0.01

    def crit_sc(met):
        return (met.cur_fx is not None) and (met.total_serious > 81)

    crit = crit_sc

    naive_prune = True
    if naive_prune:
        k = 10
        solver = 'MATLAB'
        mu_sz = 1e3
        bundle_prune = 'lambda'
        iters = 300

elif obj_type == 'Non-Convex':
    titl = obj_type + r': $R^{}$, sum over {} |quartics|'.format(n, 10)
    objective = nonconvex(n=n,k=10,oracle_output='both'); mu_sz=1e4; beta_sz=1e-5; iters = 200
    rescaled  = True
elif obj_type == 'Partly Smooth': # This is the only case that doesn't work
    titl = obj_type + ': eig-max sum of {}, {}x{} matrices'.format(n, m, m)
    objective, true_val, true_mult = partlysmooth(n=n,m=m,oracle_output='both');
    mu_sz=1; beta_sz=1e-5; iters = 400
    k = int((true_mult * (true_mult+1)/2.0) - 1)
    # fixed_shift = -true_val
    rescaled = True
    bfgs_lr = 0.01
    bundle_prune = 'duals'

    solver = 'MATLAB'

    def crit_ps(met):
        return (met.is_serious) and (met.latest_null > 20)

    naive_prune = True
    if naive_prune:
        solver = 'MATLAB'
        mu_sz = 1e1
        bundle_prune = 'lambda'
    crit = crit_ps

elif obj_type == 'Partly Smooth 3D':
    titl = obj_type + r': $\sqrt{ (x^2  - y)^2 + z^2 }  +  2(x^2 + y^2 + z^2)$'
    objective = PartlySmooth3D; mu_sz=1e1; beta_sz=1e-5; iters = 100
    n = 3


    def crit_ps3(met):
        return (met.cur_fx is not None) and (met.total_serious > 2)
    crit = crit_ps3
    k = 3

    naive_prune = True
    if naive_prune:
        k = None
        solver = 'MATLAB'
        mu_sz = 0.1
        bundle_prune = 'lambda'

elif obj_type == 'Partly Smooth 2D':
    titl = obj_type + r': $\max(3x^2 + y^2 - y , x^2 + y^2 + y)$'
    objective = PartlySmooth2D; mu_sz=1e1; beta_sz=1e-5; iters = 75
    n = 2
    k = 2

    def crit_ps2(met):
        return (met.cur_fx is not None) and (met.total_serious > 1)
    crit = crit_ps2

    naive_prune = True
    if naive_prune:
        k = None
        solver = 'MATLAB'
        mu_sz = 0.1
        bundle_prune = 'lambda'
elif obj_type == 'Convex 3D':
    titl = obj_type + r': $\sqrt{ (x^2  - y)^2 + z^2 }  +  x^2$'
    objective = Convex3D; mu_sz=1e1; beta_sz=1e-5; iters = 75
    n = 3
    k = 3
    bfgs_lr = 0.1


    def crit_c3(met):
        return (met.cur_fx is not None) and (met.total_serious > 1)
    crit = crit_c3

    naive_prune = True
    if naive_prune:
        k = None
        solver = 'MATLAB'
        mu_sz = 1
        bundle_prune = 'lambda'

elif obj_type == 'Matrix 4D':
    titl = obj_type + r': Max-eigval of $3 \times 3$ matrix; 4D input'
    objective = Matrix4D;
    mu_sz = 1;
    beta_sz = 1e-5;
    iters = 100
    n = 4
    k = 3
    proj_hess = False
    fixed_shift = -1.0 + 1e-15
    bfgs_lr = 0.1

    def crit_m4(met):
        return (met.cur_fx is not None) and (met.total_serious > 2)
    crit = crit_m4

    naive_prune = True
    if naive_prune:
        k = None
        solver = 'MATLAB'
        mu_sz = 1
        bundle_prune = 'lambda'
        iters = 75

elif obj_type == 'Half-and-Half':
    n = 4
    k = 2
    titl = obj_type + ': n={}'.format(n)
    objective = halfandhalf(n=n); mu_sz=1; beta_sz=1e-5; iters = 75

    def crit_hh(met):
        return (met.cur_fx is not None) and (met.total_serious > 4)
    crit = crit_hh

    naive_prune = True
    if naive_prune:
        k = None
        solver = 'MATLAB'
        mu_sz = 1
        bundle_prune = 'lambda'

# x0 = np.random.randn(n)
x0 = np.ones(n)
alg_list = []

optAlg2 = ProxBundle(objective, x0=x0, max_iter=iters, mu=mu_sz, null_k=beta_sz,prune=True, switch_crit=crit,
                     active_thres=1e-8, naive_prune=naive_prune)
optAlg2.optimize()
alg_list += [optAlg2]

# Run Newton-Bundle
optAlg1 = BFGS(objective, x0=x0, max_iter=iters, hist=iters, lr=bfgs_lr, linesearch='lewis_overton',
                ls_params={'c1':1e-4, 'c2':0.9, 'max_ls':1e3}, store_hessian=True,
                tolerance_change=1e-14, tolerance_grad=1e-14) #, switch_crit=crit)
optAlg1.optimize()
alg_list += [optAlg1]

# MATLAB initialization
threads = multiprocessing.cpu_count() / 2
eng = matlab.engine.start_matlab()
eng.parpool('local', threads)
eng.addpath(os.getcwd() + '/algs/newton_bundle_aux', nargout=0)

optAlg0 = NewtonBundle(objective, x0=x0, max_iter=iters, k=k, warm_start=optAlg2.saved_bundle, proj_hess=proj_hess,
                       start_type='bundle', bundle_prune=bundle_prune, rank_thres=rank_thres, pinv_cond=pinv_cond,
                       solver=solver, store_hessian=True, leaving_met=leaving_met, eng=eng)
optAlg0.optimize()
alg_list += [optAlg0]


opt_plot = OptPlot(opt_algs=alg_list, resolution=100,
                   plot_lims={'x1_max':1e-3,'x2_max':1e-3,'x3_max':1e-3,'x1_min':-1e-3,'x2_min':-1e-3,'x3_min':-1e-3})

opt_plot.plotValue(title=titl, rescaled=rescaled, val_list=['path_fx','path_delta','path_diam'], fixed_shift=fixed_shift)
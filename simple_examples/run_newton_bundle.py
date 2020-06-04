#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import numpy as np
from IPython import embed
from algs.torch_alg import BFGS
from vis.visualize import OptPlot
from algs.prox_bundle import ProxBundle
from algs.newton_bundle import NewtonBundle
from obj.obj_funcs import stronglyconvex, nonconvex, partlysmooth,halfandhalf, PartlySmooth3D, Convex3D

# Run newton-bundle optimization algorithm
n = 50
k = 10
# obj_type = 'Partly Smooth'
# obj_type = 'Half-and-Half'
# obj_type = 'Partly Smooth 3D'
# obj_type = 'Convex 3D'
obj_type = 'Strongly Convex'
m = 25
# k = 3
# n = 2

# Criteria for switching to newton-bundle
def crit_ps(met):
    return (met.fx_step > 1e-14) and (abs(met.fx_step) < 1e-7)
    # return (met.cur_fx is not None) and (met.cur_fx < 7.967431759861216)

def crit_sc(met):
    return (met.cur_fx is not None) and (met.cur_fx < 1e-2)

def crit_ps3(met):
    return (met.cur_fx is not None) and (met.cur_fx < 1e-4)

def crit_hh(met):
    return (met.cur_fx is not None) and (met.cur_fx < 1e-6)

def crit_c3(met):
    return (met.cur_fx is not None) and (met.cur_fx < 1e-1)

if obj_type == 'Strongly Convex':
    titl = obj_type + ': {}-dimensional, max over {} quartics'.format(n, k)
    objective = stronglyconvex(n=n,k=k,oracle_output='both'); mu_sz=1e3; beta_sz=1e-5; iters=125
    rescaled = False
    bundle_prune = 'lambda'
    k = None
    crit = crit_sc
    rank_thres = 1e-2
    pinv_cond = 1e-3
    bfgs_lr = 0.1
elif obj_type == 'Non-Convex':
    titl = obj_type + r': $R^{}$, sum over {} |quartics|'.format(n, k)
    objective = nonconvex(n=n,k=10,oracle_output='both'); mu_sz=1e4; beta_sz=1e-5; iters = 200
    rescaled  = True
    bfgs_lr = 0.1
elif obj_type == 'Partly Smooth':
    titl = obj_type + ': eig-max sum of {}, {}x{} matrices'.format(n, m, m)
    objective = partlysmooth(n=n,m=m,oracle_output='both'); mu_sz=1e1; beta_sz=1e-5; iters = 350
    rescaled  = True
    bundle_prune = 'duals'
    # bundle_prune = 'lambda'
    # bundle_prune = 'log_svd'
    k = 51
    crit = crit_ps
    rank_thres = 1e-4
    pinv_cond = 1e-3
    bfgs_lr = 0.01
elif obj_type == 'Partly Smooth 3D':
    titl = obj_type + r': $\sqrt{ (x^2  - y)^2 + z^2 }  +  2(x^2 + y^2 + z^2)$'
    objective = PartlySmooth3D; mu_sz=1e1; beta_sz=1e-5; iters = 50
    rescaled  = False
    n = 3
    crit = crit_ps3
    bundle_prune = 'duals'
    k = 3
    rank_thres = 1e-2
    pinv_cond = 1e-12
    bfgs_lr = 0.1
elif obj_type == 'Convex 3D':
    titl = obj_type + r': $\sqrt{ (x^2  - y)^2 + z^2 }  +  x^2$'
    objective = Convex3D; mu_sz=1e1; beta_sz=1e-5; iters = 75
    rescaled  = False
    n = 3
    crit = crit_c3
    bundle_prune = 'duals'
    k = 3
    rank_thres = 1e-1
    pinv_cond = 1e-5
    bfgs_lr = 0.1
elif obj_type == 'Half-and-Half':
    n = 4
    titl = obj_type + ': n={}'.format(n)
    objective = halfandhalf(n=n); mu_sz=1; beta_sz=1e-5; iters = 50
    rescaled  = True
    crit = crit_hh
    bundle_prune = 'svd2'
    k = None
    rank_thres = 1e-1
    pinv_cond  = 1e1
    bfgs_lr = 0.1

# x0 = np.random.randn(n)
x0 = np.ones(n)
alg_list = []

optAlg2 = ProxBundle(objective, x0=x0, max_iter=iters, mu=mu_sz, null_k=beta_sz,prune=True, switch_crit=crit,
                     active_thres=1e-8)
optAlg2.optimize()
alg_list += [optAlg2]

# Run Newton-Bundle
optAlg1 = BFGS(objective, x0=x0, max_iter=iters, hist=iters, lr=bfgs_lr, linesearch='lewis_overton',
                ls_params={'c1':0, 'c2':0.5, 'max_ls':1e3},
                tolerance_change=1e-14, tolerance_grad=1e-14) #, switch_crit=crit)
optAlg1.optimize()
alg_list += [optAlg1]

optAlg0 = NewtonBundle(objective, x0=x0, max_iter=iters, k=k, warm_start=optAlg2.saved_bundle, proj_hess=False,
                       start_type='bundle', bundle_prune=bundle_prune, rank_thres=rank_thres, pinv_cond=pinv_cond,
                       solver='MOSEK', adaptive_bundle=False)
optAlg0.optimize()
alg_list += [optAlg0]

# alg_list = []
# alg_list += [optAlg2]
# alg_list += [optAlg0]

opt_plot = OptPlot(opt_algs=alg_list, resolution=100,
                   plot_lims={'x1_max':1e-3,'x2_max':1e-3,'x3_max':1e-3,'x1_min':-1e-3,'x2_min':-1e-3,'x3_min':-1e-3})

# if n in [2,3]:
#     opt_plot.plotPath3D(domain_sz=n)

opt_plot.plotValue(title=titl, rescaled=rescaled, val_list=['path_fx','path_diam','path_delta'])







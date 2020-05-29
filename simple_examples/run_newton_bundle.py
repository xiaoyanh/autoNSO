#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import importlib
import numpy as np
from IPython import embed
from algs.torch_alg import BFGS
from vis.visualize import OptPlot
from algs.prox_bundle import ProxBundle
from algs.newton_bundle import NewtonBundle
from obj.obj_funcs import stronglyconvex, nonconvex, partlysmooth

# Run newton-bundle optimization algorithm
n = 50
k = 10
obj_type = 'Partly Smooth'
# obj_type = 'Strongly Convex'
m = 25
# k = 3
# n = 2

# Criteria for switching to newton-bundle
def crit_ps(met):
    return (met.fx_step > 1e-14) and (abs(met.fx_step) < 5e-8)

def crit_sc(met):
    return (met.cur_fx is not None) and (met.cur_fx < 1e-2)

if obj_type == 'Strongly Convex':
    titl = obj_type + ': R^{}, max over {} quartics'.format(n, k)
    objective = stronglyconvex(n=n,k=k,oracle_output='both'); mu_sz=1e3; beta_sz=1e-5; iters=125
    rescaled = False
    bundle_prune = 'lambda'
    crit = crit_sc
elif obj_type == 'Non-Convex':
    titl = obj_type + ': R^{}, sum over {} |quartics|'.format(n, k)
    objective = nonconvex(n=n,k=10,oracle_output='both'); mu_sz=1e4; beta_sz=1e-5; iters = 200
    rescaled  = True
elif obj_type == 'Partly Smooth':
    titl = obj_type + ': eig_max sum of {}, {}x{} matrices'.format(n, m, m)
    objective = partlysmooth(n=n,m=m,oracle_output='both'); mu_sz=1e1; beta_sz=1e-5; iters = 300
    rescaled  = True
    # bundle_prune = 'svd'
    # bundle_prune = 'lambda'
    crit = crit_ps

# x0 = np.random.randn(n)
x0 = np.ones(n)
alg_list = []

# optAlg1 = BFGS(objective, x0=x0, max_iter=iters, hist=iters, lr=0.1, linesearch='lewis_overton',
#                 ls_params={'c1':0, 'c2':0.5, 'max_ls':1e3},
#                 tolerance_change=1e-14, tolerance_grad=1e-14) #, switch_crit=crit)
# optAlg1.optimize()
# alg_list += [optAlg1]

optAlg2 = ProxBundle(objective, x0=x0, max_iter=iters, mu=mu_sz, null_k=beta_sz,prune=False, switch_crit=crit,
                     active_thres=1e-4)
optAlg2.optimize()
alg_list += [optAlg2]

embed()

# # Run Newton-Bundle
optAlg0 = NewtonBundle(objective, x0=x0, max_iter=iters, k=None, warm_start=optAlg2.saved_bundle, proj_hess=False,
                       start_type='bundle', bundle_prune=None, rank_thres=1e-6, pinv_cond=1e-10, solver='MOSEK')
optAlg0.optimize()
alg_list += [optAlg0]

alg_list = []
alg_list += [optAlg2]
alg_list += [optAlg0]

opt_plot = OptPlot(opt_algs=alg_list, resolution=100)

if n == 2:
    opt_plot.plotPath3D()

opt_plot.plotValue(title=titl, rescaled=rescaled)


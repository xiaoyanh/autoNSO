#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import importlib
import numpy as np
from IPython import embed
from algs.torch_alg import LBFGS
from vis.visualize import OptPlot
from algs.prox_bundle import ProxBundle
from algs.newton_bundle import NewtonBundle
from obj.obj_funcs import stronglyconvex, nonconvex, partlysmooth


# Run newton-bundle optimization algorithm
n = 50
k = 10
iters = 100

# objective = stronglyconvex(n=n,k=k,oracle_output='hess+'); bund_sz=10; mu_sz=1e3
# objective = nonconvex(n=n,k=k,oracle_output='hess+'); bund_sz=4; mu_sz=1e2
objective = partlysmooth(n=50,m=25,oracle_output='hess+'); bund_sz=13; mu_sz=1e1

x0 = np.random.randn(n)

alg_list = []

# Criteria for switching to newton-bundle
def crit(met):
    return met.cur_iter == 75
    # return (met.cur_fx is not None) and (met.cur_fx < 1e-6)

optAlg2 = ProxBundle(objective, x0=x0, max_iter=iters, mu=mu_sz, null_k=1e-3)# , switch_crit=crit)
optAlg2.optimize()
alg_list += [optAlg2]

optAlg1 = LBFGS(objective, x0=x0, max_iter=iters, hist=100, lr=0.01)
optAlg1.optimize()
alg_list += [optAlg1]

# Run Newton-Bundle
optAlg0 = NewtonBundle(objective, x0=x0, max_iter=iters, k=bund_sz)# , warm_start=optAlg2.saved_bundle)
optAlg0.optimize()
alg_list += [optAlg0]

opt_plot = OptPlot(opt_algs=alg_list)
opt_plot.plotValue()
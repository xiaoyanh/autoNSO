#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

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

# objective = stronglyconvex(n=n,k=10,oracle_output='hess+'); bund_sz=3
# objective = nonconvex(n=n,k=10,oracle_output='hess+'); bund_sz=3
objective = partlysmooth(n=50,m=25,oracle_output='hess+'); bund_sz=10

x0 = np.random.randn(n)

algs = []

# Criteria for switching to newton-bundle
def crit(met):
    return met.cur_iter == 25

optAlg2 = ProxBundle(objective, x0=x0, max_iter=50, mu=2, null_k=0.001, switch_crit=crit)
optAlg2.optimize()
algs += [optAlg2]

embed()

optAlg0 = NewtonBundle(objective, x0=x0, max_iter=50, k=bund_sz)
optAlg0.optimize()
algs += [optAlg0]

# optAlg1 = LBFGS(objective, x0=x0, max_iter=50, hist=2*n, lr=0.01)
# optAlg1.optimize()
# algs += [optAlg1]

opt_plot = OptPlot(opt_algs=algs)
opt_plot.plotValue()
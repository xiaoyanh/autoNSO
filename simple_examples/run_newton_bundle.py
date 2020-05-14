#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import numpy as np
from vis.visualize import OptPlot
from obj.obj_funcs import stronglyconvex
from algs.newton_bundle import NewtonBundle
from algs.optAlg import LBFGS, ProxBundle

# Run newton-bundle optimization algorithm
n = 50
k = 10
StronglyConvex = stronglyconvex(n=n,k=10,oracle_output='hess+')
x0 = np.random.randn(n)

algs = []

optAlg0 = NewtonBundle(StronglyConvex, x0=x0, max_iter=50, k=3)
optAlg0.optimize()
algs += [optAlg0]

optAlg1 = LBFGS(StronglyConvex, x0=x0, max_iter=50, hist=2*n, lr=0.01)
optAlg1.optimize()
algs += [optAlg1]

# optAlg2 = ProxBundle(StronglyConvex, x0=x0, max_iter=50, mu=10, null_k=0.001)
# optAlg2.optimize()
# algs += [optAlg2]


opt_plot = OptPlot(opt_algs=algs)
opt_plot.plotValue()
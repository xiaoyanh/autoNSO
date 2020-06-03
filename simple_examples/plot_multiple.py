#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""

# Uses the prox-bundle method to solve the simple objective starting at (2,3)

#%%
import torch
from torch import tensor, Tensor
from vis.visualize import OptPlot
from obj.objective import Objective
from algs.prox_bundle import ProxBundle
from algs.torch_alg import Subgradient, Nesterov, BFGS

#%%
# f(x,y) = max(|x|,y^2)
def simple2D(x):
    x = tensor(x,dtype=torch.double, requires_grad=False) if type(x) != Tensor else x # Must be torch tensor
    return torch.max(torch.abs(x[0]),0.5 * x[1]**2)

# Create the objective function
Simple2D = Objective(simple2D)

# Run prox-bundle optimization algorithm
optAlg1 = ProxBundle(Simple2D, x0=[10,3], max_iter=50)
optAlg1.optimize()
optAlg2 = Subgradient(Simple2D, x0=[10,3], max_iter=50)
optAlg2.optimize()
optAlg3 = Nesterov(Simple2D, x0=[10,3], max_iter=50)
optAlg3.optimize()
optAlg4 = BFGS(Simple2D, x0=[10,3], max_iter=50)
optAlg4.optimize()

#%% Plot 
#
opt_plot = OptPlot(opt_algs=[optAlg1, optAlg2, optAlg3, optAlg4])
opt_plot.plotPath() # Plot prox-bundle optimization path
opt_plot.plotValue() # Plot prox-bundle loss by iteration
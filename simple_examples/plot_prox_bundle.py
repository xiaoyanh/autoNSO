#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""

# Uses the prox-bundle method to solve the simple objective starting at (2,3)

#%%
import sys
sys.path.append('..')
import torch
from obj.objective import Objective
from algs.prox_bundle import ProxBundle
from vis.visualize import OptPlot

#%%
# f(x,y) = |x| + y^2
def simple2D(x):
    return torch.max(torch.abs(x[0]),0.5 * x[1]**2)

# Create the objective function
Simple2D = Objective(simple2D)

# Run prox-bundle optimization algorithm
optAlg = ProxBundle(Simple2D, x0=[10,3], max_iter=20)
optAlg.optimize()

#%% Plot 

opt_plot = OptPlot(opt_algs=[optAlg])

# Plot prox-bundle optimization path
opt_plot.plotPath()

# Plot prox-bundle loss by iteration
opt_plot.plotValue()
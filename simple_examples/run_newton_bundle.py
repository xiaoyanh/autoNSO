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
from obj.objective import Objective
from algs.newton_bundle import NewtonBundle

# f(x,y) = |x| + y^2
def simple2D(x):
    return abs(x[0]) + x[1]**2

# Create the objective function
Simple2D = Objective(simple2D)

# Run prox-bundle optimization algorithm
optAlg = NewtonBundle(Simple2D, x0=[2,3], max_iter=10)
optAlg.optimize()
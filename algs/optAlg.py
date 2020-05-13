#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""
import numpy as np
import cvxpy as cp
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from IPython import embed

class OptAlg:
    def __init__(self, objective, max_iter = 1000, x0 = None, verbose=True):
        assert x0 is not None

        if type(x0) is not np.ndarray:
            x0 = np.array(x0)
        assert len(x0.shape)==1  # x0 is a vector
        
        assert len(x0.shape) == 1 # assert initial is vector
        
        self.max_iter   = max_iter
        self.objective  = objective
        
        self.x0         = x0
        self.x_dim      = self.x0.shape[0]
        
        self.cur_iter  = 0
        self.cur_x     = None
        self.cur_fx    = None
        self.path_x      = None
        self.path_fx     = None
        
        self.verbose = verbose
        
        self.opt_x     = None
        self.opt_fx    = None
        self.total_iter = None
        
        self.name = None
    
    def optimize(self):

        # Run the optimization algorithm until a stopping condition is hit
        while self.cur_iter <= self.max_iter:
            self.step()            
            if self.stop_cond():
                break

        self.opt_x = self.path_x[-1]
        self.opt_fx = self.path_fx[-1]
        self.total_iter = self.cur_iter
        
        if self.verbose:
            print('Optimal Value: ' + str(self.opt_fx))
            print('Optimal Point: ' + str(self.opt_x))
    
    def step(self):
        if self.verbose:
            print('iter: ' + str(self.cur_iter))
    
    def stop_cond(self):
        return False
    
    def update_params(self):
        pass
    
        
class ProxBundle(OptAlg):
    def __init__(self, objective, max_iter = 10, x0 = None, mu = 1.0, null_k=0.5):
        super(ProxBundle,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.constraints    = []
        self.p              = cp.Variable(self.x_dim) # variable of optimization
        self.v              = cp.Variable() # value of cutting plane model
        self.mu             = mu
        self.null_k         = null_k
        self.name           = 'ProxBundle'
        self.name += ' (mu='+str(self.mu)+',null_k='+str(self.null_k)+')'
        
        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.cur_y = self.x0 # the auxiliary variables will null values
        self.path_y = None
        self.total_null_serious = 0
        self.update_params(None)

    def step(self):
        
        super(ProxBundle,self).step()
        
        prox_objective = self.v + 0.5 * (self.mu/2.0) * cp.power(cp.norm(self.p - self.cur_x,2),2)
        self.p.value = self.cur_y # Warm-starting
        prob = cp.Problem(cp.Minimize(prox_objective),self.constraints)
        
        # Use MOSEK for accuracy
#        m_params = {'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-12}
#        prob.solve(solver=cp.MOSEK,mosek_params=m_params)
        
        # If you don't have mosek just do:
        prob.solve(warm_start=True)
        
        # Update current iterate value and update the bundle
        self.cur_y = self.p.value
        
        # Update paths and bundle constraints
        self.update_params(self.v.value)
            
    def update_params(self, expected):
        
        if self.path_y is not None:
            self.path_y = np.concatenate((self.path_y, self.cur_y[np.newaxis]))
        else: 
            self.path_y = self.cur_y[np.newaxis]
        
        orcl_call = self.objective.call_oracle(self.cur_y)
        cur_fy    = orcl_call['f']
        
        # Whether to take a serious step
        if expected is not None:
            serious = ((self.path_fx[-1] - cur_fy) > self.null_k * (self.path_fx[-1] - expected))
        else:
            serious = True
  
        if serious:
            self.cur_x          = self.cur_y
            self.cur_fx         = orcl_call['f']
            if self.path_x is not None:
                self.path_x         = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
                self.path_fx        = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
            else:
                self.path_x         = self.cur_x[np.newaxis]
                self.path_fx        = self.cur_fx[np.newaxis]
            self.cur_iter       += 1
        
        # Even if it is null step, add a constraint to cutting plane model
        self.constraints    += [(cur_fy.copy() + 
                                 orcl_call['df'].copy()@(self.p - self.cur_y.copy())) <= self.v]
        self.total_null_serious += 1
        
        
# Subgradient method
class TorchAlg(OptAlg):
    def __init__(self, objective, max_iter=10, x0 = None):
        super(TorchAlg,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        # Add one bundle point to initial point
        self.cur_x  = self.x0
        self.update_params()
        
        # Set up criterion and thing to be optimized
        self.criterion = self.objective.obj_func
        self.p         = torch.tensor(self.x0,dtype=torch.float,requires_grad=True)
        
    def step(self):

        super(TorchAlg,self).step()
        
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        value = self.criterion(self.p)
        value.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Update current iterate value and update the bundle
        self.cur_x      = self.p.data.numpy().copy()
        self.update_params()
    
    def update_params(self):
        
        self.cur_fx = self.objective.call_oracle(self.cur_x)['f']
    
        if self.path_x is not None:
            self.path_x         = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx        = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
        else:
            self.path_x         = self.cur_x[np.newaxis]
            self.path_fx        = self.cur_fx[np.newaxis]
        
        # Update paths and bundle constraints
        self.cur_iter    += 1
            
class Subgradient(TorchAlg):
    def __init__(self, objective, max_iter=10, x0 = None, lr = 1, decay=0.9):
        super(Subgradient,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.lr = lr
        self.decay = decay
        self.name = 'Subgradient'
        self.name += ' (lr=' + str(self.lr)+',decay='+str(self.decay)+')'
        
        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=0)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)

class Nesterov(TorchAlg):
    def __init__(self, objective, max_iter=10, x0 = None, lr = 1, decay=0.9, momentum=0.9):
        super(Nesterov,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.lr = lr
        self.decay = decay
        self.momentum = 0.9
        self.name = 'Nesterov'
        self.name += (' (lr=' + str(self.lr)+',decay='+str(self.decay)
                        +',mom='+str(self.momentum)+')')
        
        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=self.momentum)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)
        
class LBFGS(TorchAlg):
    def __init__(self, objective, max_iter=10, x0 = None, lr = 1, decay=0.9, hist=100):
        super(LBFGS,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.lr = lr
        self.decay = decay
        self.hist  = hist
        self.name = 'LBFGS'
        self.name += (' (lr=' + str(self.lr)+',decay='+str(self.decay)
                        +',hist='+str(self.hist)+')')
        
        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.LBFGS([self.p], lr=self.lr, history_size=self.hist)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)
        
    def step(self):
        
        super(TorchAlg,self).step()
        
        def closure():
            self.optimizer.zero_grad()
            value = self.criterion(self.p)
            value.backward()
            return value
        
        self.optimizer.step(closure)
        self.scheduler.step()
        
        # Update current iterate value and update the bundle
        self.cur_x      = self.p.data.numpy().copy()
        self.update_params()
        
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""

import torch
from IPython import embed
from utils.jacobian_hessian import hessian

class Objective:
    def __init__(self, obj_func, requires_grad=True, oracle_output='both'):
        self.obj_func           = obj_func
        self.requires_grad      = requires_grad
        self.x     = None
        self.fx    = None
        
        assert oracle_output in ['f','df','both','hess+']
        self.oracle_output = oracle_output
        
    def call_oracle(self,x):
        
        if type(x) != torch.Tensor:
            try:
                x = torch.tensor(x, dtype=torch.double,
                                 requires_grad=self.requires_grad)
            except:
                raise Exception('Optimization variable must be Pytorch tensor\
                                 or something that could be cast into it such as\
                                 numpy array, list, etc.')
        
        assert len(x.shape) == 1
        
        if (not x.requires_grad) and (self.requires_grad):
            raise Exception('Need to enable gradients on optimization variable.')
        
        # Zero the gradient if x has one
        if x.requires_grad and (x.grad is not None):
            x.grad.zero_grad()
        
        self.x = x
        self.fx = self.obj_func(self.x)
        
        if self.fx.dim() != 0:
            raise Exception('Objective function must outputscalar value')
        
        # Uses auto differentiation to get subgradient

        if self.requires_grad and (self.oracle_output != 'hess+') :
            self.fx.backward()
        
        if self.oracle_output == 'f':
            return self.oracle_f()
        elif self.oracle_output == 'df':
            return self.oracle_df()
        elif self.oracle_output == 'both':
            return {'f'  : self.oracle_f(),
                    'df' : self.oracle_df(),
                    }
        elif self.oracle_output == 'hess+':
            assert type(x) == torch.Tensor
            hess = hessian(self.fx,self.x)
            hess['f'] = self.oracle_f()

            return hess
        
    def oracle_f(self):
        if self.x is None:
            raise Exception('Need to call the oracle first!')
        
        return self.fx.data.numpy()

    def oracle_df(self):
        if self.x is None:
            raise Exception('Need to call the oracle first!')
            
        if self.requires_grad == False:
            raise Exception('Oracle set to no return gradient')
            
        return self.x.grad.data.numpy()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:32:35 2019

@author: Xiaoyan
"""
import torch
import itertools
import numpy as np
import seaborn as sns
from algs.optAlg import OptAlg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from IPython import embed

# We can also make an object that compares multiple paths
class OptPlot:
    def __init__(self,opt_algs=None, plot_lims=None, axis_rot=(0,0), resolution=250):
        
        # List of optimization algorithms
        self.opt_algs = []
        self.x_dim    = None
        self.obj_func = None
        
        # Add to list of algorithms to compare at initialization
        if opt_algs is not None:
            self.add_alg(opt_algs)
            
        self.x1_max = None
        self.x2_max = None
        self.x3_max = None

        self.x1_min = None
        self.x2_min = None
        self.x3_min = None
            
        if plot_lims is not None:
            for key in plot_lims:
                assert key in ['x1_max', 'x2_max', 'x3_max', 'x1_min', 'x2_min', 'x3_min']
                setattr(self, key, plot_lims[key])
        
        self.axis_rot = axis_rot
        plt.style.use('seaborn-white')
        self.resolution = resolution
            
    def add_alg(self,opt_algs):
        
        # Check it is an optimization algorithm
        if type(opt_algs) is not list:
            assert isinstance(opt_algs, OptAlg)
            opt_algs = [opt_algs]
        else:
            # Check every element is an optimization algorithm
            for alg in opt_algs:
                assert isinstance(alg, OptAlg)
            
        # Add to list of optimization algorithms
        self.opt_algs += opt_algs
        
        # Do a check to make sure all algorithms match
        # After checking set the x_dim and obj_func to be the common one
        self.do_check()
        
    def plotPath(self,**kwargs):
        assert self.x_dim is not None
        
        if self.x_dim == 2:
            self.plotPath3D(**kwargs)
        elif self.x_dim == 3:
            self.plotPath4D(**kwargs)
        else:
            raise Exception('Plotting for this type of objective not implemented yet')
    
    # Plots for one dimensional input functions.
    # Coming soon...
    def plotPath2D(self):
        assert len(self.opt_algs) > 0
        assert self.x_dim == 1 # 1D domain for 2D plot
    
    # Plot for objective function of two inputs
    def plotPath3D(self, ax = None):
        assert len(self.opt_algs) > 0
        assert self.x_dim == 2 # 2D domain for 3D plot
        
        # Set unset limits
        if None in [self.x1_max, self.x1_min, self.x2_max, self.x2_min]:
            x1_lim = float('-inf')
            x2_lim = float('-inf')
            
            for alg in self.opt_algs:
                path_max = np.nanmax(np.abs(alg.path_x),axis=0)
                if path_max[0] > x1_lim:
                    x1_lim = path_max[0]
                if path_max[1] > x2_lim:
                    x2_lim = path_max[1]
            
            if self.x1_max == None:
                self.x1_max = 1.2 * x1_lim
            if self.x1_min == None:
                self.x1_min = -1.2 * x1_lim
            if self.x2_max == None:
                self.x2_max = 1.2 * x2_lim
            if self.x2_min == None:
                self.x2_min = -1.2 * x2_lim
        
        x1 = np.linspace(self.x1_min,self.x1_max,self.resolution)
        x2 = np.linspace(self.x2_min,self.x2_max,self.resolution)
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        
        def f(x1, x2):
            val = self.obj_func(torch.tensor([x1,x2],dtype=torch.double))
            try:
                return val.data.numpy()
            except:
                return val
        vf = np.vectorize(f)
        fx_grid = vf(x1_grid, x2_grid)
        if np.nanmin(fx_grid) < 0:
            fx_grid -= np.nanmin(fx_grid)

        # Plot objective
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            plot_now = True
        else:
            assert isinstance(ax, Axes3D)
            plot_now = False

        ax.plot_wireframe(x1_grid,x2_grid,fx_grid, rstride=5, cstride=5, alpha=0.3, linewidths=0.1, colors='black')
        ax.plot_surface(x1_grid,x2_grid,fx_grid,rstride = 5, cstride = 5, cmap = 'jet', alpha = .3, edgecolor = 'none' )
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.view_init(self.axis_rot[0], self.axis_rot[1])
        ax.zaxis._set_scale('log')

        palette = itertools.cycle(sns.hls_palette(len(self.opt_algs), l=.3, s=.8))
        markers = itertools.cycle(('*', '.', 'X', '^', 'D')) 
        
        # Plot optimization path
        for alg in self.opt_algs:
            ax.plot(alg.path_x[:,0],alg.path_x[:,1], alg.path_fx,
                    color = next(palette), marker = next(markers), alpha = .6, label = alg.name)
        plt.legend()
        plt.show(block=False)

    # Plot for objective function of two inputs
    def plotPath4D(self, ax=None):
        assert len(self.opt_algs) > 0
        assert self.x_dim == 3  # 3D domain, 4D with function (unplotted)

        # Set unset limits
        if None in [self.x1_max, self.x1_min, self.x2_max, self.x2_min, self.x3_max, self.x3_min]:
            x1_lim = float('-inf')
            x2_lim = float('-inf')
            x3_lim = float('-inf')

            for alg in self.opt_algs:
                path_max = np.nanmax(np.abs(alg.path_x), axis=0)
                if path_max[0] > x1_lim:
                    x1_lim = path_max[0]
                if path_max[1] > x2_lim:
                    x2_lim = path_max[1]
                if path_max[2] > x3_lim:
                    x3_lim = path_max[2]

            if self.x1_max == None:
                self.x1_max = 1.2 * x1_lim
            if self.x1_min == None:
                self.x1_min = -1.2 * x1_lim
            if self.x2_max == None:
                self.x2_max = 1.2 * x2_lim
            if self.x2_min == None:
                self.x2_min = -1.2 * x2_lim
            if self.x3_max == None:
                self.x3_max = 1.2 * x3_lim
            if self.x3_min == None:
                self.x3_min = -1.2 * x3_lim

        # Plot objective
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            plot_now = True
        else:
            assert isinstance(ax, Axes3D)
            plot_now = False

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.view_init(self.axis_rot[0], self.axis_rot[1])
        palette = itertools.cycle(sns.hls_palette(len(self.opt_algs), l=.3, s=.8))
        markers = itertools.cycle(('*', '.', 'X', '^', 'D'))

        # Plot optimization path
        for alg in self.opt_algs:
            ax.plot(alg.path_x[:, 0], alg.path_x[:, 1], alg.path_x[:, 2],
                    color=next(palette), marker=next(markers), alpha=.6, label=alg.name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(self.x1_min, self.x1_max)
        ax.set_ylim(self.x2_min, self.x2_max)
        ax.set_zlim(self.x3_min, self.x3_max)

        if plot_now:
            plt.legend()
            plt.show()
        else:
            return ax

    # Plot for objective function of two inputs
    def plotValue(self, val='path_fx', title=None, rescaled=False, ax=None):
        assert len(self.opt_algs) > 0
        assert val in ['path_fx','step_size']

        # Set up matplotlib
        if ax is None:
            fig = plt.figure(figsize = (10,10))
            ax  = fig.add_subplot(111)
            plot_now = True
        else:
            plot_now = False
        
        # Plot optimization path
        palette = itertools.cycle(sns.hls_palette(len(self.opt_algs), l=.3, s=.8))
        markers = itertools.cycle(('*', '.', 'X', '^', 'D'))

        # Count iterations and find max/min
        max_iters = float('-inf')
        all_vals = np.array([])
        for alg in self.opt_algs:

            if val == 'step_size':
                alg.step_size = np.diff(alg.path_x,axis=0)
                alg.step_size = np.linalg.norm(alg.step_size,axis=1)
                alg.step_size = np.insert(alg.step_size, 0, np.nan, axis=0)

            if alg.total_iter > max_iters:
                max_iters = alg.total_iter
            all_vals = np.concatenate((all_vals,getattr(alg,val)))

        if val == 'path_fx':
            if (min(all_vals) < 0) or rescaled:
                y_label = 'Shifted Objective Value (log-scale)'
                shift  =  -np.nanmin(all_vals)
            else:
                y_label = 'Objective Value (log-scale)'
                shift = 0
        elif val == 'step_size':
            y_label = 'Step Size'
            shift = 0

        all_vals += shift
        max_f = np.nanmax(all_vals)
        min_f = np.nanmin(all_vals[all_vals != 0])

        for alg in self.opt_algs:
            y = getattr(alg,val) + shift
            y[y==0] = min_f # Just set all 0's to second smallest
            ax.plot(np.arange(alg.total_iter), y,
                    color=next(palette), marker=next(markers),
                    alpha=.4, label=alg.name)

        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(y_label)
        ax.set_xticks(np.round(np.linspace(0,max_iters,10)))

        if title is not None:
            ax.set_title(title)
        # plt.ion()
        ax.set_ylim((min_f,max_f))
        np.set_printoptions(precision=2)
        ylabs = np.geomspace(min_f,max_f,num=5)
        ylabs_prt = ["{0:0.2e}".format(float(i)) for i in ylabs]
        ax.set_yticks(ylabs,ylabs_prt)

        if plot_now:
            plt.legend()
            plt.show()
            # plt.draw()
        else:
            return ax


    # check that all optimization algorithms have the same objective and inputs
    def do_check(self):
        
        # Use the first algorithm's dimension and objective as reference
        ref_alg = self.opt_algs[0]
        
        # Check that dimensions of input are all the same
        # Check that there is actually a path that finished
        for alg in self.opt_algs:
            assert alg.x_dim == ref_alg.x_dim
            assert alg.objective.obj_func == ref_alg.objective.obj_func
            assert alg.opt_x is not None
        
        self.x_dim      = ref_alg.x_dim
        self.obj_func   = ref_alg.objective.obj_func
        
        
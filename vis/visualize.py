#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:32:35 2019

@author: Xiaoyan
"""
import torch
import warnings
import itertools
import numpy as np
import seaborn as sns
import matplotlib
from algs.optAlg import OptAlg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True

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
    def plotPath3D(self, ax = None, domain_sz=2):
        assert len(self.opt_algs) > 0
        assert self.x_dim in [2,3] # 2D domain for 3D plot
        
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

        if domain_sz==3 and (None in [self.x3_max, self.x3_min]):
            x3_lim = float('-inf')

            for alg in self.opt_algs:
                path_max = np.nanmax(np.abs(alg.path_x), axis=0)
                if path_max[2] > x3_lim:
                    x3_lim = path_max[2]

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

        if domain_sz == 2:
            x1 = np.linspace(self.x1_min, self.x1_max, self.resolution)
            x2 = np.linspace(self.x2_min, self.x2_max, self.resolution)
            x1_grid, x2_grid = np.meshgrid(x1, x2)

            def f(x1, x2):
                val = self.obj_func(torch.tensor([x1, x2], dtype=torch.double))
                try:
                    return val.data.numpy()
                except:
                    return val

            vf = np.vectorize(f)
            fx_grid = vf(x1_grid, x2_grid)
            if np.nanmin(fx_grid) < 0:
                fx_grid -= np.nanmin(fx_grid)

            ax.plot_wireframe(x1_grid,x2_grid,fx_grid, rstride=5, cstride=5, alpha=0.3, linewidths=0.1, colors='black')
            ax.plot_surface(x1_grid,x2_grid,fx_grid,rstride = 5, cstride = 5, cmap = 'jet', alpha = .3, edgecolor = 'none' )

        ax.view_init(self.axis_rot[0], self.axis_rot[1])
        if domain_sz == 2:
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.zaxis._set_scale('log')
        elif domain_sz == 3:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        palette = itertools.cycle(sns.hls_palette(len(self.opt_algs), l=.3, s=.8))
        markers = itertools.cycle(('*', '.', 'X', '^', 'D')) 
        
        # Plot optimization path
        for alg in self.opt_algs:
            if domain_sz == 2:
                ax.plot(alg.path_x[:,0],alg.path_x[:,1], alg.path_fx,
                        color = next(palette), marker = next(markers), alpha = .6, label = alg.name)
            elif domain_sz == 3:
                ax.plot(alg.path_x[:, 0], alg.path_x[:, 1], alg.path_x[:, 2],
                        color=next(palette), marker=next(markers), alpha=.6, label=alg.name)

        if domain_sz == 2:
            ax.set_xlim(self.x1_min, self.x1_max)
            ax.set_ylim(self.x2_min, self.x2_max)
        elif domain_sz == 3:
            ax.set_zlim(self.x3_min, self.x3_max)

        if plot_now:
            plt.legend()
            plt.show(block=False)
        else:
            return ax

    # Plot for objective function of two inputs
    def plotValue(self, val_list=['path_fx'], title=None, rescaled=False, fixed_shift=0.0, ax=None,
                  rolling_min=['path_fx','path_diam','path_delta']):
        assert len(self.opt_algs) > 0
        val_list = [val_list] if isinstance(val_list,str) else val_list
        rolling_min = [rolling_min] if isinstance(val_list, str) else rolling_min
        assert np.all([val in ['path_fx', 'step_size', 'path_diam', 'path_delta', 'path_vio', 'path_hess'] for val in val_list])

        lab_dict = {'path_fx': r"$f(x)$: ",
                    'step_size': r"$|x_k - x_{k+1}|$: ",
                    'path_diam': r"diam$(S)$: ",
                    'path_delta': r"$\Theta(S)$: ",
                    'path_vio': r"$Vio.=|Ax - b|: "}

        if 'path_hess' in val_list: # Handle plotting spectrum of hessian
            val_list.remove('path_hess')
            for i in range(self.x_dim):
                val_list += ['path_hess{}'.format(i)]

                lab_dict['path_hess{}'.format(i)] = r"$\sigma_{} (\nabla^2 f(x))$: ".format(i+1)
                for alg in self.opt_algs:
                    if not hasattr(alg,'path_hess'):
                        continue
                    setattr(alg,'path_hess{}'.format(i),alg.path_hess[:,i])

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
        max_f = float('-inf')
        min_f = float('inf')

        for val in val_list:

            all_vals = np.array([])
            for alg in self.opt_algs:
                if not hasattr(alg, val):
                    continue

                if val == 'step_size':
                    alg.step_size = np.diff(alg.path_x,axis=0)
                    alg.step_size = np.linalg.norm(alg.step_size,axis=1)
                    alg.step_size = np.insert(alg.step_size, 0, np.nan, axis=0)

                if alg.total_iter > max_iters:
                    max_iters = alg.total_iter

                alg_val = getattr(alg,val)
                if val in rolling_min:
                    alg_val = np.fmin.accumulate(alg_val)
                all_vals = np.concatenate((all_vals,alg_val))

            if len(all_vals) == 0:
                warnings.warn('The value {} is empty!'.format(val))
                continue

            if val == 'path_fx':
                shift = fixed_shift
                if (min(all_vals) < 0) or rescaled:
                    shift  =  -np.nanmin(all_vals)
            else:
                shift = 0

            prefix = ''
            suffix = ''
            if abs(shift) > 0:
                prefix = 'Shifted ' + prefix

            if val in rolling_min:
                suffix += ' (cumulative min)'

            all_vals += shift
            max_f = max(np.nanmax(all_vals),max_f)
            min_f = min(np.nanmin(all_vals[all_vals != 0]),min_f)

            for alg in self.opt_algs:
                if not hasattr(alg, val):
                    continue
                y = getattr(alg,val) + shift
                y[y==0] = min_f # Just set all 0's to second smallest

                if val in rolling_min:
                    y = np.fmin.accumulate(y)

                ax.plot(np.arange(alg.total_iter), y,
                        color=next(palette), marker=next(markers), alpha=.4,
                        label= prefix + lab_dict[val] + alg.name + suffix)

        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_xticks(np.round(np.linspace(0,max_iters,10)))

        if title is not None:
            ax.set_title(title)
        # plt.ion()
        ax.set_ylim((min_f,max_f))
        np.set_printoptions(precision=2)
        ylabs = np.geomspace(min_f,max_f,num=5)
        ylabs_prt = ["{0:0.2e}".format(float(i)) for i in ylabs]
        ax.set_yticks(ylabs)
        ax.set_yticklabels(ylabs_prt)

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
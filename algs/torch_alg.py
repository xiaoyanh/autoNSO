import torch
import numpy as np
import cvxpy as cp
import torch.optim as optim
import algs.lbfgs as lbfgs

from IPython import embed
from algs.optAlg import OptAlg
from torch.optim.lr_scheduler import StepLR

# Wrappers for pytorch algorithms
class TorchAlg(OptAlg):
    def __init__(self, objective, **kwargs):
        super(TorchAlg, self).__init__(objective, **kwargs)

        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.update_params()

        # Set up criterion and thing to be optimized
        self.criterion = self.objective.obj_func
        self.p = torch.tensor(self.x0, dtype=torch.double, requires_grad=True)

    def step(self):

        super(TorchAlg, self).step()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        value = self.criterion(self.p)
        value.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Update current iterate value and update the bundle
        self.cur_x = self.p.data.numpy().copy()
        self.update_params()

    def update_params(self):

        super(TorchAlg,self).update_params()

        old_fx = self.cur_fx.copy() if (self.cur_fx is not None) else float('inf')
        self.cur_fx = self.objective.call_oracle(self.cur_x)['f']
        self.fx_step = old_fx - self.cur_fx

        if self.path_x is not None:
            self.path_x = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
        else:
            self.path_x = self.cur_x[np.newaxis]
            self.path_fx = self.cur_fx[np.newaxis]

        # Update paths and bundle constraints
        self.cur_iter += 1


# Subgradient method
class Subgradient(TorchAlg):
    def __init__(self, objective, lr=1, decay=0.9, **kwargs):
        super(Subgradient, self).__init__(objective, **kwargs)

        self.lr = lr
        self.decay = decay
        self.name = 'Subgradient'
        self.name += ' (lr=' + str(self.lr) + ',decay=' + str(self.decay) + ')'

        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=0)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)


class Nesterov(TorchAlg):
    def __init__(self, objective, lr=1, decay=0.9, momentum=0.9, **kwargs):
        super(Nesterov, self).__init__(objective, **kwargs)

        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.name = 'Nesterov'
        self.name += (' (lr=' + str(self.lr) + ',decay=' + str(self.decay)
                      + ',mom=' + str(self.momentum) + ')')

        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=self.momentum)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)


class BFGS(TorchAlg):
    def __init__(self, objective, lr=1, hist=float('inf'), linesearch='strong_wolfe', ls_params=None,
                 tolerance_change=1e-9, tolerance_grad=1e-7, **kwargs):
        super(BFGS, self).__init__(objective, **kwargs)

        self.linesearch = linesearch

        self.lr = lr
        self.hist = hist
        self.name = 'BFGS'
        self.name += (' (lr=' + str(self.lr) + ')')

        # This is a modified BFGS from PyTorch
        self.optimizer = lbfgs.LBFGS([self.p], lr=self.lr, history_size=self.hist, line_search_fn=self.linesearch,
                                     ls_params = ls_params, tolerance_change=tolerance_change, tolerance_grad=tolerance_grad)

    def step(self):
        super(TorchAlg, self).step()

        def closure():
            self.optimizer.zero_grad()
            value = self.criterion(self.p)
            value.backward()
            return value

        self.optimizer.step(closure)

        # Update current iterate value and update the bundle
        self.cur_x = self.p.data.numpy().copy()
        self.update_params()

    def save_bundle(self):
        print('Bundled Saving Triggered', flush=True)
        self.saved_bundle = {'bundle': self.path_x[-min(2*len(self.cur_x),len(self.path_x)):],
                             'iter': self.cur_iter,
                             'x'   : self.cur_x.copy()}
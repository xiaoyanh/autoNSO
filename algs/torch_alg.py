import torch
import numpy as np
import cvxpy as cp
import torch.optim as optim

from IPython import embed
from algs.optAlg import OptAlg
from torch.optim.lr_scheduler import StepLR

# Wrappers for pytorch algorithms
class TorchAlg(OptAlg):
    def __init__(self, objective, max_iter=10, x0=None):
        super(TorchAlg, self).__init__(objective, max_iter=max_iter, x0=x0)

        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.update_params()

        # Set up criterion and thing to be optimized
        self.criterion = self.objective.obj_func
        self.p = torch.tensor(self.x0, dtype=torch.float, requires_grad=True)

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

        self.cur_fx = self.objective.call_oracle(self.cur_x)['f']

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
    def __init__(self, objective, max_iter=10, x0=None, lr=1, decay=0.9):
        super(Subgradient, self).__init__(objective, max_iter=max_iter, x0=x0)

        self.lr = lr
        self.decay = decay
        self.name = 'Subgradient'
        self.name += ' (lr=' + str(self.lr) + ',decay=' + str(self.decay) + ')'

        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=0)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)


class Nesterov(TorchAlg):
    def __init__(self, objective, max_iter=10, x0=None, lr=1, decay=0.9, momentum=0.9):
        super(Nesterov, self).__init__(objective, max_iter=max_iter, x0=x0)

        self.lr = lr
        self.decay = decay
        self.momentum = 0.9
        self.name = 'Nesterov'
        self.name += (' (lr=' + str(self.lr) + ',decay=' + str(self.decay)
                      + ',mom=' + str(self.momentum) + ')')

        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=self.momentum)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)


class LBFGS(TorchAlg):
    def __init__(self, objective, max_iter=10, x0=None, lr=1, decay=0.9, hist=100):
        super(LBFGS, self).__init__(objective, max_iter=max_iter, x0=x0)

        self.lr = lr
        self.decay = decay
        self.hist = hist
        self.name = 'LBFGS'
        self.name += (' (lr=' + str(self.lr) + ',decay=' + str(self.decay)
                      + ',hist=' + str(self.hist) + ')')

        self.optimizer = optim.LBFGS([self.p], lr=self.lr, history_size=self.hist)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)

    def step(self):
        super(TorchAlg, self).step()

        def closure():
            self.optimizer.zero_grad()
            value = self.criterion(self.p)
            value.backward()
            return value

        self.optimizer.step(closure)
        self.scheduler.step()

        # Update current iterate value and update the bundle
        self.cur_x = self.p.data.numpy().copy()
        self.update_params()
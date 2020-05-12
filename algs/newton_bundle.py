import torch
import numpy as np

from algs.optAlg import OptAlg

# Subgradient method
class NewtonBundle(OptAlg):
    def __init__(self, objective, max_iter=10, x0=None, k=4):
        objective.oracle_output='hess+'

        super(NewtonBundle, self).__init__(objective, max_iter=max_iter, x0=x0)
        self.criterion = self.objective.obj_func
        self.k = k # bundle size

        # Add start with initial point
        self.cur_x = self.x0
        self.cur_fx = self.criterion(self.cur_x)
        self.update_params()

        # Set up criterion and thing to be optimized

        self.p = torch.tensor(self.x0, dtype=torch.float, requires_grad=True)

    def step(self):

        super(NewtonBundle, self).step()

        # Get current gradient and hessian
        oracle = self.objective.call_oracle(self.cur_x)
        oracle['f']
        oracle['df']
        oracle['d2f']

        # Update current iterate value and update the bundle
        self.cur_x = ... #NEW X
        self.cur_fx = oracle['f']
        self.update_params()

    def update_params(self):

        if self.path_x is not None:
            self.path_x = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
        else:
            self.path_x = self.cur_x[np.newaxis]
            self.path_fx = self.cur_fx[np.newaxis]

        # Update paths and bundle constraints
        self.cur_iter += 1
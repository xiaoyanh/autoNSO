import torch
from algs.optAlg import OptAlg

# Subgradient method
class NewtonBundle(OptAlg):
    def __init__(self, objective, max_iter=10, x0=None):
        super(NewtonBundle, self).__init__(objective, max_iter=max_iter, x0=x0)

        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.update_params()

        # Set up criterion and thing to be optimized
        self.criterion = self.objective.obj_func
        self.p = torch.tensor(self.x0, dtype=torch.float, requires_grad=True)

    def step(self):

        super(NewtonBundle, self).step()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        value = self.objective.call_oracle(self.cur_y)
        value.backward()

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
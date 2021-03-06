import numpy as np
import cvxpy as cp

from IPython import embed
from algs.optAlg import OptAlg
from algs.newton_bundle_aux.params import m_params, g_params

class ProxBundle(OptAlg):
    def __init__(self, objective, mu=1.0, null_k=0.5, ignore_null=False, prune=False, active_thres=1e-12,
                 solver='MOSEK', naive_prune=False, **kwargs):
        super(ProxBundle, self).__init__(objective, **kwargs)

        self.objective.oracle_output = 'both'

        self.constraints    = []
        self.constraint_ind = []
        self.p = cp.Variable(self.x_dim)  # variable of optimization
        self.v = cp.Variable()  # value of cutting plane model
        self.mu = mu
        self.null_k = null_k
        self.name = 'ProxBundle'
        self.prune = prune
        self.naive_prune = naive_prune
        if self.prune:
            if self.naive_prune:
                self.name += ' [Naive]'
            else:
                self.name += ' [Drop Inactive]'
        else:
            self.name += ' [No Drops]'
        self.name += ' (mu=' + str(self.mu) + ',null-k=' + str(self.null_k) + ')'
        self.active_thres = active_thres
        self.solver = solver

        assert self.solver in ['GUROBI','MOSEK']

        print("Prune Bundle: {}".format(self.prune), flush=True)

        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.cur_y = self.x0  # the auxiliary variables will null values
        self.path_y = np.array([],dtype=np.float64).reshape(0,self.x_dim)
        self.dfS = np.array([], dtype=np.float64).reshape(0, self.x_dim)  # gradients
        self.cur_rank = None
        self.path_fx = np.array([], dtype=np.float64).reshape(0, 1)
        self.path_rank = np.array([],dtype=np.float64).reshape(0,1)
        self.total_serious      = 0
        self.total_null         = 0
        self.ignore_null        = ignore_null
        self.latest_null        = 0
        self.is_serious = False

        # Some other useful info
        self.cur_tight = 0
        self.cur_active = np.array([0])
        self.tight_x = []
        self.tight_y = []

        self.update_params(None)

    def step(self):

        super(ProxBundle, self).step()

        prox_objective = self.v + 0.5 * (self.mu / 2.0) * cp.quad_form(self.p - self.cur_x, np.eye(self.x_dim))
        self.p.value = self.cur_y  # Warm-starting
        prob = cp.Problem(cp.Minimize(prox_objective), self.constraints)

        # MOSEK
        if self.solver == 'MOSEK':
            prob.solve(warm_start=True, solver=cp.MOSEK,mosek_params=m_params)
        elif self.solver == 'GUROBI':
            prob.solve(warm_start=True, solver=cp.GUROBI,**g_params)

        # Update current iterate value and update the bundle
        self.cur_y = self.p.value

        # Find number of tight constraints
        self.cur_duals = [self.constraints[i].dual_value for i in range(len(self.constraints))]

        thres = self.active_thres * max(self.cur_duals)
        self.cur_active = np.where([(self.cur_duals[i] > thres) for i in range(len(self.constraints))])[0]
        self.cur_tight = sum(self.cur_active)

        # Check tight set is actually tight
        # if self.cur_iter == 75:
        #     self.check_crit()

        # Update paths and bundle constraints
        self.update_params(self.v.value)

    def update_params(self, expected):

        if self.path_y is not None:
            self.path_y = np.concatenate((self.path_y, self.cur_y[np.newaxis]))
        else:
            self.path_y = self.cur_y[np.newaxis]

        self.tight_y += [self.cur_tight]

        orcl_call = self.objective.call_oracle(self.cur_y)
        cur_fy = orcl_call['f']

        # Whether to take a serious step
        if expected is not None:
            serious = ((self.path_fx[-1] - cur_fy) > self.null_k * (self.path_fx[-1] - expected))
        else:
            serious = True

        if serious:
            self.cur_x = self.cur_y.copy()

            if self.ignore_null:
                self.cur_fx = orcl_call['f'].copy()
                self.update_fx_step()
                self.path_x = np.vstack([self.path_x, self.cur_x])
                self.path_fx = np.vstack([self.path_fx, self.cur_fx])

                self.tight_x += [self.cur_tight]

            self.total_serious += 1
            self.latest_null += 1
            self.is_serious = True
        else:
            if self.is_serious:
                self.latest_null = 0
                self.is_serious = False
            self.latest_null += 1
            self.total_null += 1

        if not self.ignore_null:
            self.path_x = self.path_y

            self.update_fx_step()
            self.path_fx = np.vstack([self.path_fx, self.cur_fx])

        super(ProxBundle, self).update_params() # Check fx_step size save bundle if necessary

        if self.prune: # Remove inactive indices
            if serious:
                if self.naive_prune: # Throw away all constraints after serious step
                    self.constraints = []
                    self.constraint_ind = []
                else:
                    # Remove inactive constraints
                    inactive = np.setdiff1d(np.arange(len(self.constraints)),self.cur_active)[::-1] # Removes in descending order
                    [self.constraints.pop(i) for i in inactive]
                    [self.constraint_ind.pop(i) for i in inactive]

            self.constraint_ind += [self.cur_iter]
        else:
            self.constraint_ind = self.cur_active

        # Even if it is null step, add a constraint to cutting plane model
        self.constraints += [(cur_fy.copy() +
                              orcl_call['df'].copy() @ (self.p - self.cur_y.copy())) <= self.v]

        if expected is not None:
            self.cur_iter += 1 # Count null steps as interations

    def save_bundle(self):

        print('Bundled Saving Triggered', flush=True)

        if self.prune:
            duals = self.cur_duals.copy()
        else:
            duals = np.array(self.cur_duals)[self.constraint_ind].copy()

        self.saved_bundle = {'bundle': self.path_y[self.constraint_ind,:],
                             'iter': self.cur_iter,
                             'x'   : self.cur_x.copy(),
                             'duals' : duals}

        if self.prune and self.naive_prune:
            self.saved_bundle['bundle'] = np.concatenate((self.saved_bundle['bundle'],self.cur_x[np.newaxis]))
            self.saved_bundle['duals'] += [float('inf')]

    def check_crit(self):
        tmp = []
        for i in self.constraint_ind:
            orcl_call = self.objective.call_oracle(self.path_y[i])
            tmp += [orcl_call['f'] + orcl_call['df'] @ (self.cur_y - self.path_y[i])]

        assert np.all([np.isclose(self.v.value,val) for val in tmp])

    def update_fx_step(self):
        old_fx = self.cur_fx.copy() if (self.cur_fx is not None) else float('inf')
        self.cur_fx = self.objective.obj_func(self.cur_x).data.numpy()
        self.fx_step = (old_fx - self.cur_fx)
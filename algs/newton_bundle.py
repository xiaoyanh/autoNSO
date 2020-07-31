import numpy as np

from IPython import embed
from utils.pinv import pinv2
from algs.bundleAlg import BundleAlg
from scipy.sparse import diags

# Bundle Newton Method from Lewis-Wylie 2019
class NewtonBundle(BundleAlg):
    def __init__(self, objective, proj_hess = False,
                 store_hessian=False, adaptive_bundle=False,
                 hessian_type='autograd', **kwargs):

        # Set up criterion
        self.store_hessian = store_hessian
        self.adaptive_bundle = adaptive_bundle
        self.hessian_type = hessian_type
        self.proj_hess    = proj_hess
        objective.oracle_output = 'hess+'

        super(NewtonBundle, self).__init__(objective, **kwargs)

        assert self.hessian_type in ['autograd','cI']
        assert self.solver in ['MOSEK','GUROBI','OSQP','CVXOPT','quadprog','MATLAB']
        assert self.leaving_met in ['delta','ls','grad_dist','cayley_menger']

        print("Project Hessian: {}".format(self.proj_hess),flush=True)

    def step(self):

        super(NewtonBundle, self).step()

        G  = self.D @ self.dfS # See Lewis-Wylie (2019)
        b_l = self.D@(np.einsum('ij,ij->i',self.dfS,self.S) - self.fS)

        if self.proj_hess: # Project hessian. See Lewis-Wylie 2019
            Q, R    = np.linalg.qr(G.T, mode='complete')
            V = Q[:,:(self.k-1)]
            U = Q[:,(self.k-1):]

            p = V @ np.linalg.inv(G@V) @ b_l

            UhU = np.stack([U.T @ self.d2fS[i,:,:] @ U for i in range(self.k)])

            A = np.einsum('s,sij->ij',self.lam_cur,UhU)
            b1 = np.einsum('s,sij,jk,ks->i',self.lam_cur,UhU,U.T,(self.S - p).T)
            b2 = np.einsum('s,ij,js->i',self.lam_cur,U.T,self.dfS.T)
            b  = b1 - b2

            xu = pinv2(A, rcond=self.pinv_cond) @ b
            self.cur_x = U@xu + p
        else:
            A = np.zeros([self.x_dim+self.k,self.x_dim+self.k])

            A[0:self.x_dim,0:self.x_dim] = np.einsum('s,sij->ij', self.lam_cur, self.d2fS)
            A[0:self.x_dim,self.x_dim:(self.x_dim+self.k)] = self.dfS.T
            A[self.x_dim,self.x_dim:(self.x_dim+self.k)]   = 1
            A[(self.x_dim+1):, 0:self.x_dim]               = G

            b =  np.zeros(self.x_dim+self.k)
            b[0:self.x_dim] = np.einsum('s,sij,sj->i',self.lam_cur,self.d2fS,self.S)
            b[self.x_dim]   = 1
            b[self.x_dim+1:] = b_l
            self.cur_x = (pinv2(A, cond=self.pinv_cond) @ b)[0:self.x_dim]

        # self.vio = np.linalg.norm(A @ self.cur_x - b)

        # optimality check
        # self.opt_check(A, b)

        self.post_step()

    def update_params(self):

        if self.store_hessian:
            self.hessian = self.oracle['d2f']

        super(NewtonBundle,self).update_params()

        if self.path_x is not None:
            if self.store_hessian:
                self.path_hess = np.concatenate((self.path_hess, np.linalg.svd(self.hessian,compute_uv=False)[np.newaxis]))
        else:
            if self.store_hessian:
                self.path_hess = np.linalg.svd(self.hessian,compute_uv=False)[np.newaxis]

    def create_paths(self):
        super(NewtonBundle, self).create_paths()

        if self.store_hessian:
            self.path_hess = np.zeros([self.cur_iter + 1, self.x_dim]) * np.nan

    def opt_check(self, A, b):
            mu  = (np.linalg.pinv(A,rcond=self.pinv_cond) @ b)[self.x_dim:self.x_dim+self.k]
            tmp = np.zeros(self.k)
            tmp2 = np.zeros(self.x_dim)
            for i in range(self.k):
                tmp[i] = self.fS[i] + self.dfS[i,:]@(self.cur_x - self.S[i,:])
                tmp2 += self.lam_cur[i] * self.d2fS[i] @ (self.cur_x - self.S[i,:])
                tmp2 += mu[i] * self.dfS[i]

            assert np.all([np.isclose(tmp[0], val) for val in tmp]) # Check active set
            assert np.isclose(np.linalg.norm(tmp2),0) # Check first order cond
            assert np.isclose(sum(mu),1) # Check duals

    def update_k(self):
        self.k = self.S.shape[0]
        self.D = diags([1, -1], offsets=[0, 1], shape=(self.k - 1, self.k)).toarray()

        met_dict = {'delta' : r'$\Theta$',
                    'grad_dist' : r'min$|\cdot - \nabla f(x)|$',
                    'cayley_menger'   : 'Cayley-Menger'}

        self.name = 'NewtonBundle '
        self.name += '(met='+met_dict[self.leaving_met]+')'
        self.name += '(bund-sz=' + str(self.k)
        self.name += ', t0='+str(self.start_iter)
        if self.proj_hess:
            self.name += ' U-projected'
        self.name += ')'
        if self.hessian_type == 'cI':
            self.name += '(First-Order)'

        print('Bundle Size Set to {}'.format(self.k), flush=True)
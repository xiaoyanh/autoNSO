import torch
from IPython import embed
from obj.objective import Objective
from torch import abs, max, sum, norm, einsum, stack, symeig, tensor, Tensor

def simple2D(x):
    return max(abs(x[0]),(0.5 * x[1]**2))
Simple2D = Objective(simple2D)

# Below are example objective functions from Lewis-Wylie 2019 (https://arxiv.org/abs/1907.11742)

# Creates a strongly convex objective function for particular n and k
def stronglyconvex(n=50, k=10, seed=0, **kwargs):
    torch.random.manual_seed(seed)

    c = torch.randn(k,dtype=torch.double)
    g = torch.randn(k,n,dtype=torch.double)
    tmp = torch.randn(k,n,n,dtype=torch.double)
    H = stack([tmp[i,:,:].T @ tmp[i,:,:] for i in range(k)])

    def sc_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        term1 = g@x
        term2 = 0.5 * stack([x.T @ H[i,:,:] @ x for i in range(k)])
        term3 = (1./24.) * (norm(x)**4) * c
        return max(term1+term2+term3)

    return Objective(sc_function, **kwargs)

# Creates a nonconvex objective function for particular n and k
def nonconvex(n=50, k=10, seed=0, **kwargs):
    torch.random.manual_seed(seed)

    c = torch.randn(k,dtype=torch.double)
    g = torch.randn(k,n,dtype=torch.double)
    tmp = torch.randn(k,n,n,dtype=torch.double)
    H = stack([tmp[i,:,:].T @ tmp[i,:,:] for i in range(k)])

    def nc_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        term1 = g@x
        term2 = 0.5 * stack([x.T @ H[i,:,:] @ x for i in range(k)])
        term3 = (1./24.) * (norm(x)**4) * c
        return sum(abs(term1 + term2 + term3)[0:k])

    return Objective(nc_function, **kwargs)

# Creates a partly objective function for particular n and m
def partlysmooth(n=50, m=25, seed=0, **kwargs):
    torch.random.manual_seed(seed)

    tmp = torch.randn(n+1,m,m,dtype=torch.double)
    A = stack([tmp[i, :, :].T + tmp[i, :, :] for i in range(n+1)])

    def ps_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        mat = A[0,:,:] + einsum('i,ijk->jk',x,A[1:,:,:])
        return symeig(mat,eigenvectors=True)[0][-1] # eigenvalues in ascending order

    return Objective(ps_function, **kwargs)

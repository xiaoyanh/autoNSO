import torch
from obj.objective import Objective

def simple2D(x):
    return max(abs(x[0]),(0.5 * x[1]**2))
Simple2D = Objective(simple2D)

# Creates a strongly convex objective function for particular n and k
def stronglyconvex(n=50, k=10, seed=0):
    torch.random.manual_seed(seed)

    c = torch.randn(k)
    g = torch.randn(k,n)
    tmp = torch.randn(k,n,n)
    H = torch.stack([tmp[i,:,:].T @ tmp[i,:,:] for i in range(k)])

    def sc_function(x):
        if type(x) != torch.Tensor: # If non-tensor passed in, no gradient will be used
            x = torch.tensor(x, dtype=torch.float, requires_grad=False)
        assert len(x) == n

        term1 = g@x
        term2 = 0.5 * torch.stack([x.T @ H[i,:,:] @ x for i in range(k)])
        term3 = (1./24) * (torch.norm(x)**4) * c
        return torch.max(term1+term2+term3)

    return Objective(sc_function)

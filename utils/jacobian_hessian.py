# Taken from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7

import torch

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x, return_grad=True):
    gradient = jacobian(y, x, create_graph=True)

    if not return_grad:
        return jacobian(gradient, x)
    else:
        return {'df' : gradient.data.numpy(),
                'hessian'  : jacobian(gradient, x).data.numpy()}
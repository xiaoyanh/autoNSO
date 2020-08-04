import numpy as np

from IPython import embed

def hess_approx_cI(hess, sig_type='max', mu=float('inf')):
    if sig_type == 'sigI':
        return np.linalg.norm(hess, ord=2) * np.eye(hess.shape[0])
    elif sig_type == 'sigMinI':
        s = np.linalg.svd(hess, compute_uv=False, hermitian=True)
        return min(s[s > max(s)*1e-2]) * np.eye(hess.shape[0])
    elif sig_type == 'sigCapAuto':
        u, s, vt = np.linalg.svd(hess, compute_uv=True, hermitian=True)
        s[s > mu] = mu
        s[s < (1.0/mu)] = 1.0/mu
        return u @ np.diag(s) @ vt
    elif sig_type in ['muI','cI']:
        return mu * np.eye(hess.shape[0])
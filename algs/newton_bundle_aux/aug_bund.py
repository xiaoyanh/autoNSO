import numpy as np
from IPython import embed

from algs.newton_bundle_aux.get_lambda import get_lam

def create_bundle(obj, bundle_prune, warm_start, start_type):

    # Add higher order info results
    obj.fS = np.zeros(obj.S.shape[0])
    obj.dfS = np.zeros([obj.S.shape[0], obj.x_dim])
    obj.d2fS = np.zeros([obj.S.shape[0], obj.x_dim, obj.x_dim])
    for i in range(obj.S.shape[0]):
        oracle = obj.objective.call_oracle(obj.S[i, :])
        obj.fS[i] = oracle['f']
        obj.dfS[i, :] = oracle['df']
        obj.d2fS[i, :, :] = oracle['d2f']

    if warm_start and start_type == 'bundle' and (bundle_prune is not None):
        assert bundle_prune in ['lambda', 'svd', 'log_lambda', 'log_svd', 'svd2', 'duals']
        print('Preprocessing bundle with {}.'.format(bundle_prune), flush=True)

        def active_from_vec(rank, vec):
            if obj.proj_hess:
                rank = min(rank, obj.x_dim)
            return np.argsort(vec)[-rank:]

        def geo_gap(vec, exclude_first=True):
            sorted = np.argsort(abs(np.diff(np.log10(np.sort(vec)[::-1]))))
            ind = -2 if exclude_first else -1
            return sorted[ind] + 1

        if bundle_prune == 'svd':
            sig = np.linalg.svd(obj.dfS, compute_uv=False)
            rank = int(1 * sum(sig > max(sig) * obj.rank_thres))
            active = active_from_vec(rank, warm_start['duals'])
        elif bundle_prune == 'svd2':
            sig = np.linalg.svd(np.concatenate((obj.dfS, np.ones(obj.k)[:, np.newaxis]), axis=1), compute_uv=False)
            rank = int(1 * sum(sig > max(sig) * obj.rank_thres))
            active = active_from_vec(rank, warm_start['duals'])
        elif bundle_prune == 'duals':
            assert obj.k is not None
            rank = obj.k
            active = active_from_vec(rank, warm_start['duals'])
        elif bundle_prune == 'lambda':
            _, tmp_lam = get_lam(obj.dfS, solver=obj.solver, eng=obj.eng)
            if obj.solver == 'MATLAB':
                rank = sum((tmp_lam > 0))
            else:
                rank = sum(tmp_lam > obj.rank_thres * max(tmp_lam))
            active = active_from_vec(rank, tmp_lam)
        elif bundle_prune == 'log_lambda':
            _, tmp_lam = get_lam(obj.dfS, solver=obj.solver, eng=obj.eng)
            rank = geo_gap(tmp_lam, exclude_first=True)
            active = active_from_vec(rank, tmp_lam)
        elif bundle_prune == 'log_svd':
            sig = np.linalg.svd(obj.dfS, compute_uv=False)
            rank = geo_gap(sig, exclude_first=True)
            active = active_from_vec(rank, warm_start['duals'])

        obj.S = obj.S[active, :]
        obj.fS = obj.fS[active]
        obj.dfS = obj.dfS[active, :]
        obj.d2fS = obj.d2fS[active, :]
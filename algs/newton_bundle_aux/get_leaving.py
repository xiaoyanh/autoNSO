import numpy as np
import multiprocessing
from IPython import embed
from joblib import Parallel, delayed
from utils.cayley_menger import simplex_vol
from algs.newton_bundle_aux.get_lambda import get_lam, get_LS

def get_leaving(obj, oracle):
    if obj.leaving_met == 'delta':
        jobs_delta, jobs_lambda = get_lam(obj.dfS, new_df=oracle['df'], solver=obj.solver, eng=obj.eng)
        k_sub = np.argmin(jobs_delta)
        obj.lam_cur = jobs_lambda[k_sub, :]
    elif obj.leaving_met == 'ls':
        ls_size = lambda i: get_LS(obj.S, obj.fS, obj.dfS,
                                   sub_ind=i, new_S=obj.cur_x, new_fS=oracle['f'], new_df=oracle['df'])
        jobs = Parallel(n_jobs=min(multiprocessing.cpu_count(), obj.k))(delayed(ls_size)(i) for i in range(obj.k))
        k_sub = np.argmax(jobs)
    elif obj.leaving_met == 'grad_dist':
        k_sub = np.argmin(np.linalg.norm(obj.dfS - oracle['df'],axis=1))
    elif obj.leaving_met == 'cayley_menger':
        simplex_vol(obj.dfS)
        pass

    if obj.leaving_met == 'delta' and jobs_delta[k_sub] >= obj.cur_delta and obj.adaptive_bundle:
        obj.S = np.concatenate((obj.S, obj.cur_x[np.newaxis]))
        obj.fS = np.concatenate((obj.fS, obj.cur_fx[np.newaxis]))
        obj.dfS = np.concatenate((obj.dfS, oracle['df'][np.newaxis]))
        obj.d2fS = np.concatenate((obj.d2fS, oracle['d2f'][np.newaxis]))
        obj.update_k()

        # old_delta = obj.cur_delta.copy()

        obj.cur_delta, obj.lam_cur = get_lam(obj.dfS, solver=obj.solver, eng=obj.eng)
        k_sub = None

    return k_sub
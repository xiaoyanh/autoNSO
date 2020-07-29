import numpy as np

def const_scal(vec_raw, vec_tar):
    assert len(vec_raw) == len(vec_tar)

    k = np.nanmedian(vec_tar / vec_raw)
    return np.around(k,2), (vec_raw * k)
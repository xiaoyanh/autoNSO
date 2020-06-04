import numpy as np
from numba import njit

# @njit(parallel=True)
def get_diam(arr,axis=0):

    if axis == 1:
        arr = arr.T

    nrows = arr.shape[0]
    diam = 0
    for i in range(nrows):
        for j in range(i):
            diam = max(np.linalg.norm(arr[i,:] - arr[j,:]),diam)
    return diam
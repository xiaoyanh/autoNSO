from IPython import embed
import numpy as np
import math
from numba import njit

from sklearn.metrics import pairwise_distances

# Adapted from https://pydoc.net/PyMF/0.1.9/pymf.vol/

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@njit
def fast_factorial(n):
    if n > 20:
        return math.gamma(n+1)
    return LOOKUP_TABLE[n]

@njit(parallel=True)
def simplex_vol(vecs):
    k = vecs.shape[0]
    d = np.zeros((k,k),dtype=np.float64)
    for i in range(k):
        for j in range(k):
            if i == j:
                pass
            else:
               d[i,j] = np.linalg.norm(vecs[i,:] - vecs[j,:])

    # compute the CMD determinant of the euclidean distance matrix d
    # -> d should not be squared!
    D = np.ones((d.shape[0] + 1, d.shape[0] + 1))
    D[0, 0] = 0.0
    D[1:, 1:] = d ** 2
    j = D.shape[0] - 2
    f1 = (-1.0) ** (j + 1) / ((2 ** j) * ((fast_factorial(j)) ** 2))
    cmd = f1 * np.linalg.det(D)
    # sometimes, for very small values "cmd" might be negative ...
    return np.sqrt(np.abs(cmd))
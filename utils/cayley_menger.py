from IPython import embed
import numpy as np
from math import factorial
from sklearn.metrics import pairwise_distances

# Adapted from https://pydoc.net/PyMF/0.1.9/pymf.vol/

def simplex_vol(vecs):
    d = pairwise_distances(vecs)

    # compute the CMD determinant of the euclidean distance matrix d
    # -> d should not be squared!
    D = np.ones((d.shape[0] + 1, d.shape[0] + 1))
    D[0, 0] = 0.0
    D[1:, 1:] = d ** 2
    j = np.float64(D.shape[0] - 2)
    f1 = (-1.0) ** (j + 1) / ((2 ** j) * ((factorial(j)) ** 2))
    cmd = f1 * np.linalg.det(D)
    # sometimes, for very small values "cmd" might be negative ...
    return np.sqrt(np.abs(cmd))
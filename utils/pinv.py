import numpy as np
import scipy.linalg.decomp_svd as decomp_svd

from IPython import embed
from scipy.linalg.decomp import _asarray_validated

def pinv2(a, rank=None, cond=None, rcond=None, return_rank=False, check_finite=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.
    Calculate a generalized inverse of a matrix using its
    singular-value decomposition and including all 'large' singular
    values.
    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    cond, rcond : float or None
        Cutoff for 'small' singular values; singular values smaller than this
        value are considered as zero. If both are omitted, the default value
        ``max(M,N)*largest_singular_value*eps`` is used where ``eps`` is the
        machine precision value of the datatype of ``a``.
        .. versionchanged:: 1.3.0
            Previously the default cutoff value was just ``eps*f`` where ``f``
            was ``1e3`` for single precision and ``1e6`` for double precision.
    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix. Returned if `return_rank` is True.
    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    Examples
    --------
    >>> from scipy import linalg
    >>> a = np.random.randn(9, 6)
    >>> B = linalg.pinv2(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True
    """
    a = _asarray_validated(a, check_finite=check_finite)
    u, s, vh = decomp_svd.svd(a, full_matrices=False, check_finite=False)

    if rank is None:
        if rcond is not None:
            cond = np.max(s) * rcond
        if cond in [None, -1]:
            t = u.dtype.char.lower()
            cond = np.max(s) * max(a.shape) * np.finfo(t).eps
        rank = np.sum(s > cond)
    elif rank == float('inf'):
        assert (cond is not None)
        rank = len(s)
        s[s<cond] = cond

    print(rank, flush='True')
    u = u[:, :rank]
    u /= s[:rank]
    B = np.transpose(np.conjugate(np.dot(u, vh[:rank])))

    if return_rank:
        return B, rank
    else:
        return B
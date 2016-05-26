"""A few miscellaneous cython routines to speed up critical operations.
"""

from cython.parallel import prange, parallel
cimport cython

import numpy as np
cimport numpy as np

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b


# A routine for quickly calculating the noise part of the banded
# covariance matrix for the Wiener filter.
@cython.boundscheck(False)
cpdef _band_wiener_covariance(double [:, ::1] Rn, double [::1] Ni, int bw):

    cdef double [:, ::1] Ci = np.zeros((bw+1, Rn.shape[0]), dtype=np.float64)

    cdef unsigned int N, M
    cdef unsigned int alpha, beta, betap, j, alpha_start

    cdef double t

    N = Rn.shape[0]
    M = Rn.shape[1]

    # Loop over the band array indices to generate each one (opposite
    # order for faster parallelisation)
    for beta in prange(N, nogil=True):

        # Calculate alphas to start at
        alpha_start = int_max(0, bw - beta)

        for alpha in range(alpha_start, bw+1):
            betap = alpha + beta - bw
            t = 0.0
            for j in range(M):
                t = t + Rn[betap, j] * Rn[beta, j] * Ni[j]
            Ci[alpha, beta] = t

    return np.asarray(Ci)


def _unpack_product_array_fast(cython.numeric[::1] utv, cython.numeric[:, ::1] mat, cython.integral[::1] feeds, int nfeed):
    """Fast unpacking of a product array.

    The output array must be preallocated and passed in.

    Parameters
    ----------
    utv : np.ndarray[(nfeed+1)*nfeed/2]
        Upper triangular vector to unpack.
    mat : np.ndarray[lfeed, lfeed]
        The output square array to unpack into.
    feeds : np.ndarray[lfeed]
        Indices of feeds to unpack into array.
    nfeed : int
        Number of feeds contained in upper triangle.
    """
    cdef int lfeed = len(feeds)

    cdef int i, j, fi, fj, pi

    for i in range(lfeed):
        for j in range(i, lfeed):

            fi = feeds[i]
            fj = feeds[j]

            pi = (nfeed * (nfeed+1) / 2) - ((nfeed-fi) * (nfeed-fi + 1) / 2) + (fj - fi)

            mat[i, j] = utv[pi]

        for j in range(i):

            fi = feeds[j]
            fj = feeds[i]

            pi = (nfeed * (nfeed+1) / 2) - ((nfeed-fi) * (nfeed-fi + 1) / 2) + (fj - fi)

            mat[i, j] = utv[pi].conjugate()

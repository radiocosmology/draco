"""A few miscellaneous Cython routines to speed up critical operations.
"""

from cython.parallel import prange, parallel
cimport cython

import numpy as np
cimport numpy as np

from libc.stdint cimport int16_t, uint32_t

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b


# A routine for quickly calculating the noise part of the banded
# covariance matrix for the Wiener filter.
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef _band_wiener_covariance(double [:, ::1] Rn, double [::1] Ni,
                              int[::1] start_ind, int[::1] end_ind, int bw):

    cdef double [:, ::1] Ci = np.zeros((bw+1, Rn.shape[0]), dtype=np.float64)

    cdef int N, M
    cdef int alpha, beta, betap, j, alpha_start, si, ei

    cdef double t

    N = Rn.shape[0]
    M = Rn.shape[1]

    # Loop over the band array indices to generate each one (opposite
    # order for faster parallelisation)
    for beta in prange(N, nogil=True):

        # Calculate alphas to start at
        alpha_start = int_max(0, bw - beta)

        si = start_ind[beta]
        ei = end_ind[beta]

        for alpha in range(alpha_start, bw+1):
            betap = alpha + beta - bw
            t = 0.0
            for j in range(si, ei + 1):
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



@cython.boundscheck(False)
@cython.wraparound(False)
def _calc_redundancy(float[:, ::1] input_flags, int16_t[:, ::1] prod_map, uint32_t[::1] stack_index,
                     int nstack, float[:, ::1] redundancy):
    """Quickly calculate redundancy.

    Parameters
    ----------
    input_flags : np.ndarray[input, time]
        The input flags. Must be zeros or ones.
    prod_map : np.ndarray[prod, 2]
        The product map.
    stack_index : np.ndarray[prod]
        The stack map.
    nstack : int
        Number of stacks.
    redundancy : np.ndarray[nstack, ntime]
        Array in which to fill out the redundancy of each stack.
    """
    
    cdef int istack
    cdef int ia, ib
    cdef int ii, jj
    
    cdef int ninput = input_flags.shape[0]
    cdef int ntime = input_flags.shape[1]

    cdef int nt
    
    # Check the array shapes
    # Need to construct rshape as redundancy.shape, has extra entries as it's of a memoryview
    rshape = (redundancy.shape[0], redundancy.shape[1])
    if rshape != (nstack, ntime):
        raise RuntimeError("redundancy array shape %s incorrect, expected %s" %
                           (repr(rshape), repr((nstack, ntime))))
        
    # Check that we don't index out of bounds from the prod_map
    if np.min(prod_map) < 0 or np.max(prod_map) >= ninput:
        raise RuntimeError("Input index in prod_map out of bounds.")

    with nogil:

        # Loop over all products
        for ii in range(prod_map.shape[0]):

            istack = stack_index[ii]
            ia = prod_map[ii, 0]
            ib = prod_map[ii, 1]

            # Make sure stack index is positive and less than the number of unique baselines.
            # Negative index or index with large value can indicate a product was
            # not included in the stack.
            if not ((istack >= 0) and (istack < nstack)):
                continue

            # Fill out the redundancy array
            # This loop could be OpenMP parallelised, but ntime is typically so small the overhead isn't worth it.
            for jj in range(ntime):

                # Increment the redundancy counter for this unique baseline if both inputs good
                redundancy[istack, jj] += input_flags[ia, jj] * input_flags[ib, jj]


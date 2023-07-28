"""A few miscellaneous Cython routines to speed up critical operations."""

from cython.parallel import prange, parallel
from cython.view cimport array as cvarray
cimport cython

import numpy as np
cimport numpy as np

from libc.stdint cimport int16_t, int32_t
from libc.stdlib cimport malloc, free
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport fabs

ctypedef double complex complex128
ctypedef float complex complex64

cdef extern from "complex.h" nogil:
    #complex64 conj(complex64)
    complex128 conj(complex128)
    double creal(complex128)
    double cimag(complex128)

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b


# A routine for quickly calculating the noise part of the banded
# covariance matrix for the Wiener filter.
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef _band_wiener_covariance(double [:, ::1] Rn, double [::1] Ni,
                              int[::1] start_ind, int[::1] end_ind, int bw):

    cdef double [:, ::1] Ci = np.zeros((bw+1, Rn.shape[0]), dtype=np.float64)

    cdef int alpha, beta, betap, j, alpha_start, si, ei

    cdef double t

    cdef int N = Rn.shape[0]

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
            for j in range(si, ei):
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

            pi = (nfeed * (nfeed+1) // 2) - ((nfeed-fi) * (nfeed-fi + 1) // 2) + (fj - fi)

            mat[i, j] = utv[pi]

        for j in range(i):

            fi = feeds[j]
            fj = feeds[i]

            pi = (nfeed * (nfeed+1) // 2) - ((nfeed-fi) * (nfeed-fi + 1) // 2) + (fj - fi)

            mat[i, j] = utv[pi].conjugate()



@cython.boundscheck(False)
@cython.wraparound(False)
def _calc_redundancy(
    float[:, ::1] input_flags,
    int16_t[:, ::1] prod_map,
    int32_t[::1] stack_index,
    int nstack,
    float[:, ::1] redundancy
):
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

    if stack_index.shape[0] != prod_map.shape[0]:
        raise ValueError(
            f"Number of prod_map rows ({prod_map.shape[0]}) must match stack_index "
            f"length ({stack_index.shape[0]})."
        )

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


cdef extern from "complex.h" nogil:
    double complex cexp(double complex)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef beamform(float complex [:, :, ::1] vis,
               double[:, :, ::1] weight,
               double dec, double lat,
               double[::1] cosha, double[::1] sinha,
               double[:, ::1] u, double[:, ::1] v,
               int[::1] f_index, int[::1] ra_index):
    """ Fringestop visibility data and sum over products.

    CAUTION! For efficiency reasons this routine does not
    normalize the sum over products. This is to avoid dividing
    here and multiplyig again for further stacking in the time
    axis later on.

    If no stacking in the time axis is made further in your code
    you should do

                    formed_beam / np.sum(weight, axis=-1)

    to get a proper normalization.

    Parameters
    ----------
    vis : complex np.ndarray[freq, RA/time, product/stack]
        Visibility data. Notice this is not in the usual order.
        This order reduces data striding.
    weight : double np.ndarray[freq, RA/time, product/stack]
        The weights to be used for adding products.
    dec : double
        Source declination.
    lat : double
        Latitude of observation.
    cosha : double np.ndarray[HA]
        Cosine of hour angle array
    sinha : double np.ndarray[HA]
        Sine of hour angle array
    u : double np.ndarray[freq, product/stack]
        X-direction (EW) baseline in wavelengths
    v : double np.ndarray[freq, product/stack]
        Y-direction (NS) baseline in wavelengths
    f_index : int np.ndarray[freq_to_process]
        Indices in the frequencies to process
    ra_index : int np.ndarray[HA]
        Indicies in the RA axis of the HA in cosha, sinha
    """

    cdef double cosdec, sindec, coslat, sinlat
    cdef double fsphase
    cdef int nfreq, nra, nprod
    cdef int ii, jj, kk
    cdef int fi, ri
    cdef double pi
    nfreq, nra, nprod = len(f_index), len(ra_index), vis.shape[2]
    # To store the formed beams. Will only be populated at f_index
    # frequency entries. Zero otherwise.
    cdef double[:, ::1] formed_beam = np.zeros((vis.shape[0], nra), dtype=np.float64)
    cdef double phase, ut, vt, st, ct

    pi = np.pi
    cosdec, sindec = cos(dec), sin(dec)
    coslat, sinlat = cos(lat), sin(lat)

    for ii in prange(nfreq, nogil=True):
        fi = f_index[ii]

        for jj in range(nra):

            ri = ra_index[jj]

            formed_beam[fi, jj] = 0.0

            ut = 2.0 * pi * cosdec * sinha[jj]
            vt = -2.0 * pi * (coslat * sindec - sinlat * cosdec * cosha[jj])

            for kk in range(nprod):
                phase = u[fi, kk] * ut + v[fi, kk] * vt
                st = sin(phase)
                ct = cos(phase)
                formed_beam[fi, jj] += weight[fi, ri, kk] * (vis[fi, ri, kk] * (ct + 1j * st)).real

    return np.asarray(formed_beam)


cdef extern from "float.h" nogil:
    double DBL_MAX
    double FLT_MAX

ctypedef fused real_or_complex:
    double
    double complex
    float
    float complex


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef _fast_var(
    real_or_complex[:, :, ::1] arr,
    double[:, ::1] out
):
    """Fast parallel variance evaluation.

    This is specialised to dim=3 arrays with the variance taken along axis=1. This implementation uses the Youngs and Cramer single pass algorithm.

    Parameters
    ----------
    arr
        The array to take the variance of.
    out
        The array to write the variance into.
    """
    cdef int N1, N2, N3
    cdef int ii, jj, kk
    cdef real_or_complex t
    cdef double t2
    cdef real_or_complex* T
    cdef int size = sizeof(T)

    N1 = arr.shape[0]
    N2 = arr.shape[1]
    N3 = arr.shape[2]

    if arr.shape[0] != out.shape[0] or arr.shape[2] != out.shape[1]:
        raise ValueError("Input and output array shapes incompatible.")

    with nogil, parallel():

        T = <real_or_complex*>malloc(N3 * sizeof(real_or_complex))

        for ii in prange(N1):

            for kk in range(N3):
                out[ii, kk] = 0.0
                T[kk] = arr[ii, 0, kk]

            for jj in range(1, N2):
                for kk in range(N3):
                    t = arr[ii, jj, kk]
                    T[kk] += t
                    t = (jj + 1) * t - T[kk]
                    if (
                        (real_or_complex is cython.doublecomplex) or
                        (real_or_complex is cython.floatcomplex)
                    ):
                        t2 = creal(t)**2 + cimag(t)**2
                    else:
                        t2 = t * t
                    out[ii, kk] += t2 / (jj * (jj + 1))


            for kk in range(N3):
                out[ii, kk] /= N2

        free(T)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef _fast_cov(
    real_or_complex[:, ::1] arr,
    real_or_complex[:, ::1] out,
    bint bias = False,
):
    """Fast parallel covariance evaluation.

    This is specialised to dim=2 arrays with samples along the last axis, and the
    variables along the first.

    Parameters
    ----------
    arr
        The array to take the covariance of.
    out
        The array to write the covariance into.
    bias
        If False (default) normalise by N - 1. Otherwise by N.
    """

    cdef int nd = arr.shape[0]
    cdef int nsamp = arr.shape[1]

    cdef int ii, jj, kk, iijj

    cdef int norm = nsamp if bias else nsamp - 1

    cdef real_or_complex ti, tj

    if out.shape[0] != nd or out.shape[1] != nd:
        raise ValueError(f"Out array does not have the required shape ({nd}, {nd}).")

    cdef real_or_complex* means = <real_or_complex*>malloc(sizeof(real_or_complex) * nd)
    cdef real_or_complex* tarr = <real_or_complex*>malloc(sizeof(real_or_complex) * nd)

    #for ii in prange(nd, nogil=True):
    #for ii in range(nd):
    #    for jj in range(nsamp):
    #        means[ii] += arr[ii, jj]
#
    #for ii in range(nd):
    #    means[ii] /= nsamp
#
    ##for iijj in prange(nd * nd, nogil=True):
    #for iijj in range(nd * nd):
#
    #    ii = iijj // nd
    #    jj = iijj % nd
#
    #    out[ii, jj] = 0
#
    #    for kk in range(nsamp):
#
    #        ti = arr[ii, kk] - means[ii]
    #        tj = arr[jj, kk] - means[jj]
#
    #        if (
    #            (real_or_complex is cython.doublecomplex) or
    #            (real_or_complex is cython.floatcomplex)
    #        ):
    #            out[ii, jj] += ti * conj(tj)
    #        else:
    #            out[ii, jj] += ti * tj;
#
    #    out[ii, jj] /= norm

    for ii in range(nd):
        for jj in range(nsamp):
            means[ii] += arr[ii, jj]

    for ii in range(nd):
        means[ii] /= nsamp

    for ii in range(nd):
        for jj in range(nd):
            out[ii, jj] = 0.0

    #for iijj in prange(nd * nd, nogil=True):
    for kk in range(nsamp):

        for ii in range(nd):
            tarr[ii] = arr[ii, kk] - means[ii]

        for ii in range(nd):
            for jj in range(nd):

                if (
                    (real_or_complex is cython.doublecomplex) or
                    (real_or_complex is cython.floatcomplex)
                ):
                    out[ii, jj] += tarr[ii] * conj(tarr[jj])
                else:
                    out[ii, jj] += tarr[ii] * tarr[jj];

    for ii in range(nd):
        for jj in range(nd):
            out[ii, jj] /= norm

    free(means)
    free(tarr)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef _mean_sub(
    real_or_complex[:, ::1] arr,
):
    """In-place subtraction of the mean along the last axis.

    Parameters
    ----------
    arr
        The array to subtract the mean of.
    """

    cdef int nd = arr.shape[0]
    cdef int nsamp = arr.shape[1]

    cdef int ii, jj

    cdef real_or_complex* means = <real_or_complex*>malloc(sizeof(real_or_complex) * nd)

    #for ii in prange(nd, nogil=True):
    with nogil:
        for ii in range(nd):
            for jj in prange(nsamp):
                means[ii] += arr[ii, jj]

        for ii in range(nd):
            means[ii] /= nsamp

        for ii in range(nd):
            for jj in prange(nsamp):
                arr[ii, jj] -= means[ii]

    free(means)
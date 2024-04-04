"""Routines for regridding irregular data using a Lanczos/Wiener filtering approach.

This is described in some detail in `doclib:173
<http://bao.chimenet.ca/doc/cgi-bin/general/documents/display?Id=173>`_.
"""

from typing import List, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as ss

from ..util import _fast_tools


def band_wiener(R, Ni, Si, y, bw):
    r"""Calculate the Wiener filter assuming various bandedness properties.

    In particular this asserts that a particular element in the filtered
    output will only couple to the nearest `bw` elements. Equivalently, this
    is that the covariance matrix will be band diagonal. This allows us to use
    fast routines to generate the solution.

    Note that the inverse noise estimate returned is :math:`\mathrm{diag}(\mathbf{R}^T
    \mathbf{N}^{-1} \mathbf{R})` and not the full Bayesian estimate including a
    contribution from the signal covariance.

    Parameters
    ----------
    R : np.ndarray[m, n]
        Transfer matrix for the Wiener filter.
    Ni : np.ndarray[k, n]
        Inverse noise matrix. Noise assumed to be uncorrelated (i.e. diagonal matrix).
    Si : np.narray[m]
        Inverse signal matrix. Signal model assumed to be uncorrelated (i.e. diagonal
        matrix).
    y : np.ndarray[k, n]
        Data to apply to.
    bw : int
        Bandwidth, i.e. how many elements couple together.

    Returns
    -------
    xhat : np.ndarray[k, m]
        Filtered data.
    nw : np.ndarray[k, m]
        Estimate of variance of each element.
    """
    Ni = np.atleast_2d(Ni)
    y = np.atleast_2d(y)

    k = Ni.shape[0]
    m = R.shape[0]

    # Initialise arrays
    xh = np.zeros((k, m), dtype=y.dtype)
    nw = np.zeros((k, m), dtype=np.float32)

    # Multiply by noise weights inplace to reduce memory usage (destroys original)
    y *= Ni

    # Calculate dirty estimate (and output straight into xh)
    R_s = R.astype(np.float32)
    np.dot(y, R_s.T, out=xh)

    # Calculate the start and end indices of the summation
    start_ind = (R != 0).argmax(axis=-1).astype(np.int32)
    end_ind = R.shape[-1] - (R[..., ::-1] != 0).argmax(axis=-1)
    end_ind = np.where((R == 0).all(axis=-1), 0, end_ind).astype(np.int32)

    # Iterate through and solve noise
    for ki in range(k):
        # Upcast noise weights to float type
        Ni_ki = Ni[ki].astype(np.float64)

        # Calculate the Wiener noise weighting (i.e. inverse covariance)
        Ci = _fast_tools._band_wiener_covariance(R, Ni_ki, start_ind, end_ind, bw)

        # Set the noise estimate before adding in the signal contribution. This avoids
        # the issue that the inverse-noise estimate becomes non-zero even when the data
        # was entirely missing
        nw[ki] = Ci[-1]

        # Add on the signal covariance part
        Ci[-1] += Si

        # Solve for the Wiener estimate
        xh[ki] = la.solveh_banded(Ci, xh[ki])

    return xh, nw


def lanczos_kernel(x, a):
    """Lanczos interpolation kernel.

    Parameters
    ----------
    x : array_like
        Point separation.
    a : integer
        Lanczos kernel width.

    Returns
    -------
    kernel : np.ndarray
    """
    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), np.zeros_like(x))


def lanczos_forward_matrix(x, y, a=5, periodic=False):
    """Lanczos interpolation matrix.

    Parameters
    ----------
    x : np.ndarray[m]
        Points we have data at. Must be regularly spaced.
    y : np.ndarray[n]
        Point we want to interpolate data onto.
    a : integer, optional
        Lanczos width parameter.
    periodic : boolean, optional
        Treat input points as periodic.

    Returns
    -------
    matrix : np.ndarray[m, n]
        Lanczos regridding matrix. Apply to data with `np.dot(matrix, data)`.
    """
    dx = x[1] - x[0]

    sep = (x[np.newaxis, :] - y[:, np.newaxis]) / dx

    if periodic:
        n = len(x)
        sep = np.where(np.abs(sep) > n // 2, n - np.abs(sep), sep)

    return lanczos_kernel(sep, a)


def lanczos_inverse_matrix(x, y, a=5, cond=1e-1):
    """Regrid data using a maximum likelihood inverse Lanczos.

    Parameters
    ----------
    x : np.ndarray[m]
        Points to regrid data onto. Must be regularly spaced.
    y : np.ndarray[n]
        Points we have data at. Irregular spacing.
    a : integer, optional
        Lanczos width parameter.
    cond : float
        Relative condition number for pseudo-inverse.

    Returns
    -------
    matrix : np.ndarray[m, n]
        Lanczos regridding matrix. Apply to data with `np.dot(matrix, data)`.
    """
    lz_forward = lanczos_forward_matrix(x, y, a)
    return la.pinv(lz_forward, rcond=cond)


def rebin_matrix(tra: np.ndarray, ra: np.ndarray, width_t: float = 0) -> np.ndarray:
    """Construct a matrix to rebin the samples.

    Parameters
    ----------
    tra
        The samples we have in the time stream.
    ra
        The target samples we want in the sidereal stream.
    width_t
        The width of a time sample. Set to zero to do a nearest bin assignment.

    Returns
    -------
    R
        A matrix to perform the rebinning.
    """
    R = np.zeros((ra.shape[0], tra.shape[0]))

    inds = np.searchsorted(ra, tra)

    # Estimate the typical width of an RA bin
    width_ra = np.median(np.abs(np.diff(ra)))

    lowest_ra = ra[0] - width_ra / 2
    highest_ra = ra[-1] + width_ra / 2

    # NOTE: this is a bit of a hack to avoid zero division by zeros, but should have
    # the required effect of giving 1 if the time sample is inside a bin, and zero
    # otherwise.
    if width_t == 0:
        width_t = 1e-10

    # NOTE: this can probably be done more efficiently, but we typically only need
    # to do this once per day
    for ii, (jj, t) in enumerate(zip(inds, tra)):

        lower_edge = t - width_t / 2.0
        upper_edge = t + width_t / 2.0

        # If we are in here we have weight to assign to the sample below
        if upper_edge > lowest_ra and jj < len(ra):
            ra_edge = ra[jj] - width_ra / 2
            wh = np.clip((upper_edge - ra_edge) / width_t, 0.0, 1.0)
            R[jj, ii] = wh

        if lower_edge < highest_ra and jj > 0:
            ra_edge = ra[jj - 1] + width_ra / 2
            wl = np.clip((ra_edge - lower_edge) / width_t, 0.0, 1.0)
            R[jj - 1, ii] = wl

    return R


def taylor_coeff(
    x: np.ndarray,
    N: int,
    M: int,
    Ni: np.ndarray,
    Si: float,
    period: Union[float, None] = None,
    xc: Union[np.ndarray, None] = None,
) -> List[ss.csr_array]:
    """Return a set of sparse matrices that estimates expansion coefficients.

    Parameters
    ----------
    x
        Positions of each element.
    N
        Number of positions each side to estimate from.
    M
        The number of terms in the expansion.
    Ni
        The weight for each position. The inverse noise variance if interpreted as a
        Wiener filter.
    Si
        A regulariser. The inverse signal variance if interpreted as a Wiener filter.
    period
        If set, assume the axis is periodic with this period.
    xc
        An optional parameter giving the location to return the coefficients at. If not
        set, just use the locations of each individual sample.


    Returns
    -------
    matrices
        A set of sparse matrices that will estimate the coefficents at each location.
        Each matrix is for a different coefficient.
    """
    nx = x.shape[0]

    ind = np.arange(nx)[:, np.newaxis] + np.arange(-N, N + 1)[np.newaxis, :]

    xc = x if xc is None else xc

    # If periodic then just wrap back around
    if period is not None:
        ind = ind % nx
        xf = x[ind] - xc[:, np.newaxis]
        x = ((x + period / 2) % period) - period / 2
        Na = Ni[ind]

    # If not then set the weights for out of bounds entries to zero
    else:
        mask = (ind < 0) | (ind >= nx)
        ind = np.where(mask, 0, ind)
        xf = x[ind] - x[:, np.newaxis]
        Na = Ni[ind]
        Na[mask] = 0.0

    # Create the required matrices at each location
    X = np.stack([xf**m for m in range(M)], axis=2)
    XhNi = (X * Na[:, :, np.newaxis]).transpose(0, 2, 1)
    XhNiX = XhNi @ X

    # Calculate the covariance part of the filter
    Ci = np.identity(M) * Si + XhNiX
    C = np.zeros_like(Ci)
    for i in range(nx):
        C[i] = la.inv(Ci[i])

    W = C @ XhNi

    # Create the indptr array designating the start and end of the indices for each row
    # in the csr_array
    indptr = (2 * N + 1) * np.arange(nx + 1, dtype=int)
    return [
        ss.csr_array((W[:, i].ravel(), ind.ravel(), indptr), shape=(nx, nx))
        for i in range(M)
    ]

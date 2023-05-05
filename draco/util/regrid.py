"""Routines for regridding irregular data using a Lanczos/Wiener filtering approach.

This is described in some detail in `doclib:173
<http://bao.chimenet.ca/doc/cgi-bin/general/documents/display?Id=173>`_.
"""

import numpy as np
import scipy.linalg as la

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

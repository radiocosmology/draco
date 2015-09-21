"""
=================================================================
General routines for regridding data (:mod:`~ch_pipeline.regrid`)
=================================================================

.. currentmodule:: ch_pipeline.regrid

Routines for regridding irregular using a Lanczos/Wiener filtering approach.
This is described in some detail in `doclib:173
<http://bao.phas.ubc.ca/doc/cgi-bin/general/documents/display?Id=173>`_.

Tasks
=====

.. autosummary::
    :toctree: generated/

    band_wiener
    lanczos_kernel
    lanczos_forward_matrix
    lanczos_inverse_matrix
"""

import numpy as np
import scipy.linalg as la

from . import _fast_tools


def band_wiener(R, Ni, Si, y, bw):
    """Calculate the Wiener filter assuming various bandedness properties.

    In particular this asserts that a particular element in the filtered
    output will only couple to the nearest `bw` elements. Equivalently, this
    is that the covariance matrix will be band diagonal. This allows us to use
    fast routines to generate the solution.

    Parameters
    ----------
    R : np.ndarray[m, n]
        Transfer matrix for the Wiener filter.
    Ni : np.ndarray[k, n]
        Inverse noise matrix. Noise assumed to be uncorrelated (i.e. diagonal matrix).
    Si : np.narray[m]
        Inverse signal matrix. Signal model assumed to be uncorrelated (i.e. diagonal matrix).
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

    Ni = np.atleast_2d(Ni) #.astype(np.float64)
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

    # Iterate through and solve noise
    for ki in range(k):

        # Upcast noise weights to float type
        Ni_ki = Ni[ki].astype(np.float64)

        # Calculate the Wiener noise weighting (i.e. inverse covariance)
        Ci = _fast_tools._band_wiener_covariance(R, Ni_ki, bw)

        # Add on the signal covariance part
        Ci[-1] += Si

        # Solve for the Wiener estimate
        xh[ki] = la.solveh_banded(Ci, xh[ki])
        nw[ki] = Ci[-1]

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

    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x/a), np.zeros_like(x))


def lanczos_forward_matrix(x, y, a=5):
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
    dx = x[1] - x[0]

    sep = (x[np.newaxis, :] - y[:, np.newaxis]) / dx

    lz_forward = lanczos_kernel(sep, a)

    return lz_forward


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
    lz_inverse = la.pinv(lz_forward, rcond=cond)

    return lz_inverse

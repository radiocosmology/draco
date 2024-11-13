"""Routines for regridding irregular data using a Lanczos/Wiener filtering approach.

This is described in some detail in `doclib:173
<http://bao.chimenet.ca/doc/cgi-bin/general/documents/display?Id=173>`_.
"""

from typing import Union

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
    Si : np.ndarray[k, m] | float
        Inverse signal variance. Signal assumed to be uncorrelated (i.e. diagonal matrix).
    y : np.ndarray[k, n]
        Data to apply to.
    bw : int
        Bandwidth, i.e. how many elements couple together.

    Returns
    -------
    xh : np.ndarray[k, m]
        Filtered signal
    nw : np.ndarray[k, m]
        Estimate of inverse noise variance of each element
    """
    # Project the signal through the inverse transfer matrix
    xh = wiener_projection(R, y, Ni)

    # Deconvolve the projected signal and noise
    return wiener_deconvolve(R, Ni, Si, xh, bw)


def wiener_projection(R, y, Ni=None, inplace=False):
    r"""Calculate the dirty estimate of a signal given the transfer matrix and inverse noise.

    :math:`\mathbf{R}^{T} \mathbf{N}^{-1} \mathbf{y}`

    Parameters
    ----------
    R : np.ndarray[m, n]
        Transfer matrix
    Ni : np.ndarray[k, n]
        Inverse noise matrix
    y : np.ndarray[k, n]
        Measured signal/data
    inplace : bool
        If True, multiply Ni in place.

    Returns
    -------
    xhat : np.ndarray[k, m]
        Dirty estimate of the signal y
    """
    y = np.atleast_2d(y)

    # Multiply by noise weights inplace to reduce
    # memory usage (destroys original)
    if Ni is not None:
        if inplace:
            y *= np.atleast_2d(Ni)
        else:
            y = y * np.atleast_2d(Ni)

    return y @ R.T.astype(y.real.dtype)


def wiener_noise_covariance(R, Ni, bw):
    """Make the band wiener noise covariance matrix.

    This is a large matrix so unless it is explicitly needed in its
    entirety, use `wiener_deconvolve` to solve the deconvolution
    without generating the entire matrix.

    Parameters
    ----------
    R : np.ndarray[m, n]
        Transfer matrix for the Wiener filter.
    Ni : np.ndarray[..., n]
        Inverse noise matrix. Noise assumed to be uncorrelated (i.e. diagonal matrix)
    bw : int
        Bandwidth, i.e. how many elements couple together.

    Returns
    -------
    Ci : np.ndarray[..., bw, m]
        Signal covariance matrix for each PCA mode
    """
    Ni = np.atleast_2d(Ni)
    shape_ = Ni.shape
    Ni = Ni.reshape(-1, Ni.shape[-1])
    Ci = np.zeros((Ni.shape[0], bw + 1, R.shape[0]), dtype=np.float64)

    # Calculate the start and end indices of the summation
    start_ind, end_ind = _get_band_inds(R)

    for ki in range(Ni.shape[0]):
        Ni_ki = Ni[ki].astype(np.float64)
        # Calculate the Wiener noise weighting (i.e. inverse covariance)
        # for each frequency and mode
        Ci[ki] = _fast_tools._band_wiener_covariance(R, Ni_ki, start_ind, end_ind, bw)

    return Ci.reshape((*shape_[:-1], *Ci.shape[-2:]))


def wiener_deconvolve(R, Ni, Si, xh, bw):
    r"""Solve for a filtered signal given a dirty estimate, transfer matrix, and noise.

    Solve :math:`\mathbf{C}^{-1} \mathbf{\hat{v}} = \mathbf{w}`,
    where `w` is the dirty estimate of the signal.

    Parameters
    ----------
    R : np.ndarray[m, n]
        Transfer matrix for the Wiener filter.
    Ni : np.ndarray[k, n]
        Inverse noise matrix. Noise assumed to be uncorrelated (i.e. diagonal matrix).
    Si : np.narray[k, m]
        Inverse signal matrix. Signal model assumed to be uncorrelated (i.e. diagonal
        matrix).
    xh : np.ndarray[k, m]
        Dirty signal estimate. The deconvolved signal is saved to this array and will
        overwrite it
    bw : int
        Bandwidth, i.e. how many elements couple together.

    Returns
    -------
    xh : np.ndarray[k, m]
        Filtered data.
    nw : np.ndarray[k, m]
        Noise realization
    """
    Ni = np.atleast_2d(Ni)
    xh = np.atleast_2d(xh)

    # If Si is given as a fixed value, add correct dimension
    # and broadcast to the correct shape
    Si = np.atleast_2d(Si)
    Si = np.broadcast_to(Si, xh.shape)

    # Make the output noise array
    nw = np.zeros_like(xh, dtype=np.float32)

    # Calculate the start and end indices of the summation
    start_ind, end_ind = _get_band_inds(R)

    # Iterate through the frequency-baseline axis and solve noise
    for ki in range(Ni.shape[0]):
        # Upcast noise weights to float type
        Ni_ki = Ni[ki].astype(np.float64)
        # Calculate the Wiener noise weighting (i.e. inverse covariance)
        Ci = _fast_tools._band_wiener_covariance(R, Ni_ki, start_ind, end_ind, bw)
        # Get the noise estimate as well
        nw[ki] = Ci[-1]
        # Add on the signal covariance and solve
        Ci[-1] += Si[ki]

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
    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.0)


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


def grad_1d(
    x: np.ndarray, si: np.ndarray, mask: np.ndarray, period: Union[float, None] = None
) -> np.ndarray:
    """Gradient with boundary samples wrapped.

    Parameters
    ----------
    x
        Data to calculate the gradient for.
    si
        Positions of the samples. Must be monotonically increasing.
    mask
        Boolean mask, True where a sample is flagged.
    period
        Period of `samples`. Default is None, which produces a
        non-periodic gradient.

    Returns
    -------
    gradient
        Gradient of `x`. Gradient is set to zero where any sample in
        the calculation was flagged.
    mask
        Boolean mask corresponding to samples for which a proper
        gradient could not be calculated. True where a sample
        is flagged.
    """
    if period is not None:
        # Wrap each array, accounting for the periodicity
        # in sample positions
        x = np.concatenate(([x[-1]], x, [x[0]]))
        mask = np.concatenate(([mask[-1]], mask, [mask[0]]))
        # Account for the possibility of multiple periods
        shift = np.ceil(si[-1] / period) * period
        si = np.concatenate(([si[-1] - shift], si, [si[0] + shift]))
        # Return with wrapped samples removed
        sel = slice(1, -1)
    else:
        # Calculate the gradient using `np.gradient` first order
        # one-sided difference at the boundaries
        sel = slice(None)

    # Extend the flagged values such that any gradient which
    # includes a flagged sample is set to 0. This effectively
    # involves masking any sample where an adjacent sample is masked
    mask |= np.concatenate(([False], mask[:-1])) | np.concatenate((mask[1:], [False]))

    # `np.gradient` will produce NaNs if the sample separation is zero -
    # i.e., if there are two adjacent zeros in `si`. Explicitly set
    # these values to zero
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.gradient(x, si)

    mask |= ~np.isfinite(grad)
    grad[mask] = 0.0

    return grad[sel], mask[sel]


def taylor_coeff(
    x: np.ndarray,
    N: int,
    M: int,
    Ni: np.ndarray,
    Si: float,
    period: Union[float, None] = None,
    xc: Union[np.ndarray, None] = None,
) -> list[ss.csr_array]:
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


def _get_band_inds(R: np.ndarray) -> tuple:
    """Get the indices of the band edge for a band diagonal matrix.

    Parameters
    ----------
    R
        Band diagonal matrix

    Returns
    -------
    start_ind : np.ndarray[int]
        left indices of the band
    end_ind : np.ndarray[int]
        right indices of the band
    """
    start_ind = (R != 0).argmax(axis=-1).astype(np.int32)
    end_ind = R.shape[-1] - (R[..., ::-1] != 0).argmax(axis=-1)
    end_ind = np.where((R == 0).all(axis=-1), 0, end_ind).astype(np.int32)

    return start_ind, end_ind

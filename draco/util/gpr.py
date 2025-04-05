"""Routines for Gaussian Process Regression."""

import numpy as np
import scipy.linalg as la
from caput.tools import invert_no_zero
from scipy.spatial.distance import cdist

from . import kernels


def resample(
    data: np.ndarray,
    weight: np.ndarray,
    xi: np.ndarray,
    xo: np.ndarray,
    cutoff_dist: float = 1.0,
    flagn: int = 0,
    **kernel_params,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a dataset using gaussian process regression.

    Parameters
    ----------
    data
        Data array. Iterate over the first axis and interpolate the second.
        Samples are assumed to be constant over the third axis.
    weight
        Data weights, broadcastable to the shape of `data`.
    xi
        Samples where data points have been measured
    xo
        Samples we want to interpolate onto.
    cutoff_dist
        Maximum distance from nearest unflagged input sample to keep
        output samples.
    flagn
        Propagate flag cutoff to `n` closest sample. Goes into `np.partition`.
        Value of 0 is equivalent to the closest sample. Default is 0.
    kernel_params
        Kernel type and structure parameters.

    Returns
    -------
    xout
        Resampled data array.
    wout
        Resampled weight array.
    """
    # Get the kernels based on the kernel parameters, assuming
    # a gaussian kernel if no name is provided
    kernel_type = kernel_params.pop("name", "gaussian")
    svar = kernel_params.pop("svar", 0.0)
    # Create the source kernel and add a noise variance term
    # TODO: it should be configurable whether this should be obtained
    # from the weights or from a constant value
    Ki = kernels.get_kernel(name=kernel_type, N=xi, **kernel_params)
    Ki += np.eye(Ki.shape[0]) * svar
    # Create the interpolate kernel
    Ks = kernels.get_kernel(name=kernel_type, N=(xo, xi), **kernel_params)

    # Cast to 32-bit precision
    Ki = Ki.astype(np.float32)
    Ks = Ks.astype(np.float32)

    xout, wout = interpolate_noisefree(data, weight, Ki, Ks)

    # Flag samples depending on how far we want to interpolate
    inp_mask = ~np.all(weight == 0, axis=-1)
    mask = flag_sample_dist(xi, xo, inp_mask, cutoff_dist, n=flagn)

    # Apply the sample mask
    xout *= mask[..., np.newaxis]
    wout *= mask[..., np.newaxis]

    return xout, wout


def interpolate_noisefree(
    data: np.ndarray,
    weight: np.ndarray,
    K: np.ndarray,
    Kstar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate an input array assuming no noise.

    Iterate the first axis and interpolate the second.

    Parameters
    ----------
    data
        Data array. Iterate over the first axis and interpolate over the second.
    weight
        Weight array, with the same first two axes and `data`.
    K
        Source samples kernel.
    Kstar
        Projection kernel.

    Returns
    -------
    xout
        Interpolated data array.
    woud
        Interpolated weight array.
    """
    # Get the square of each kernel for use each loop
    Kstar2 = Kstar**2

    # Make the output arrays
    nsamp = Kstar.shape[0]
    xout = np.zeros((data.shape[0], nsamp, *data.shape[2:]), dtype=data.dtype)
    wout = np.zeros((weight.shape[0], nsamp, *weight.shape[2:]), dtype=weight.dtype)

    complex_data = np.iscomplexobj(data)

    # Iterate the first axis and interpolate the second
    for ii in range(data.shape[0]):
        wi = np.ascontiguousarray(weight[ii])
        xi = np.ascontiguousarray(data[ii])
        mi = ~np.all(wi == 0, axis=-1)

        if mi.sum() < 3:
            continue

        # Get the cholesky of the kernel to invert
        kb = _make_band_diagonal(K[mi][:, mi])

        # Interpolate the data. If `xi` is complex, solve as a real view
        if complex_data:
            xi = xi.view(np.float32)

        alpha = la.solveh_banded(kb, xi[mi], check_finite=False, lower=True)
        yp = Kstar[:, mi] @ alpha

        if complex_data:
            # Convert back to complex
            yp = yp.reshape(xout.shape[1:] + (2,))
            xout[ii] = yp[..., 0] + 1j * yp[..., 1]
        else:
            xout[ii] = yp

        # Interpolate variances
        # TODO: check that this is correct
        vi = invert_no_zero(wi[mi])
        beta = la.solveh_banded(kb**2, vi, check_finite=False, lower=True)
        wout[ii] = invert_no_zero(Kstar2[:, mi] @ beta)

    # Assuming that weights should not be negative, flag out any samples
    # where that happened.
    wout[wout < 0] = 0.0
    xout[wout < 0] = 0.0

    return xout, wout


def _make_band_diagonal(x, tol=1e-3):
    """Convert a full band diagonal kernel into just the lower band.

    Used to feed into `la.solveh_banded.`
    """
    N = x.shape[0]
    M = np.sum(x > tol, axis=-1).max()

    banded = np.zeros((M, N), dtype=x.dtype)

    for i in range(M):
        banded[i, : N - i] = x.diagonal(i)

    return banded


def flag_sample_dist(
    xi: np.ndarray, xo: np.ndarray, mask: np.ndarray, cutoff: float, n: int = 0
) -> np.ndarray:
    """Mask output samples which are more than some distance from input samples.

    Parameters
    ----------
    xi
        Samples where data points have been measured
    xo
        Samples we want to interpolate onto
    mask
        Mask corresponding to measured samples. `True` values are assumed
        to be unflagged.
    cutoff
        Flag any output sample which is farther (in number of _input_ samples)
        away from the `n`th nearest unflagged input sample than this value.
    n
        `n`th closest sample to consider when applying cutoff. This is fed into
        `np.partition`. A value of `0` is relative to the nearest sample. Default
        is 0.

    Returns
    -------
    out
        Mask with flagged samples set to `False`
    """
    dist = cdist(xo[:, np.newaxis], xi[:, np.newaxis], metric="euclidean")
    # Divide by the sample width to get the distance in number of input samples
    dist /= np.median(np.abs(np.diff(xi)))

    out = np.empty((mask.shape[0], xo.shape[0]), dtype=bool)

    # Iterate over the first axis of `mask`
    for ii in range(mask.shape[0]):
        mi = mask[ii]

        if not np.any(mi):
            out[ii] = False
            continue

        out[ii] = np.partition(dist[:, mi], n, axis=-1)[:, n] < cutoff

    return out

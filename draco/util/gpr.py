"""Routines for Gaussian Process Regression."""

import numpy as np
import scipy.linalg as la

from . import kernels


def resample(
    data: np.ndarray,
    weight: np.ndarray,
    xi: np.ndarray,
    xo: np.ndarray,
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
    kernel_params
        Kernel type and structure parameters

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

    return interpolate_noisefree(data, weight, Ki, Ks)


def interpolate_noisefree(
    data: np.ndarray, weight: np.ndarray, K: np.ndarray, Kstar: np.ndarray
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
        kb2 = kb**2

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

        # Interpolate weights
        # TODO: I don't think that this is the correct way to
        # do this
        beta = la.solveh_banded(kb2, wi[mi], check_finite=False, lower=True)
        wout[ii] = Kstar2[:, mi] @ beta

        # Get a mask and apply it to the data and weights
        err = la.solveh_banded(kb2, np.ones_like(kb[0]), check_finite=False, lower=True)
        err = (Kstar2[:, mi] @ err) > 0.97

        xout[ii] *= err[:, np.newaxis]
        wout[ii] *= err[:, np.newaxis]

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

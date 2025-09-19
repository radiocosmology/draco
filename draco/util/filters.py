"""Utility functions for filtering data.

Includes implementations of median filtering and high-pass filtering.
"""

import numpy as np
import numpy.typing as npt
from caput import tools, weighted_median
from scipy import linalg as la
from scipy import signal

from .tools import window_generalised

__all__ = [
    "highpass_weighted_convolution_filter",
    "lowpass_weighted_convolution_filter",
    "medfilt",
    "null_filter",
]


def lowpass_weighted_convolution_filter(
    data: npt.NDArray[np.floating | np.complexfloating],
    weight: npt.NDArray[np.floating | np.complexfloating],
    samples: npt.NDArray[np.floating | np.complexfloating],
    cutoff: float,
    axis: int = -1,
) -> np.ndarray[np.floating | np.complexfloating]:
    """Apply a low-pass weighted convolution filter along an axis.

    Parameters
    ----------
    data
        The data to filter.
    weight
        The weights to apply.
    samples
        The sample times.
    cutoff
        The cutoff frequency.
    axis
        The axis to apply the filter along.
    """
    # Broadcast the kernel to the right dimension
    bcast_sl = [np.newaxis] * data.ndim
    bcast_sl[axis] = Ellipsis
    bcast_sl = tuple(bcast_sl)

    # Median sampling rate
    fs = 1 / np.median(abs(np.diff(samples)))
    # Order is sample frequency over cutoff frequency. Ensure order is odd
    order = int(np.ceil(fs / cutoff) // 2 * 2 + 1)
    # Make the window. Flattop seems to work well here
    kernel = signal.firwin(order, cutoff, window="flattop", fs=fs)[bcast_sl]

    # Low-pass filter the visibilities. `oaconvolve` is significantly
    # faster than the standard convolve method
    vw_lp = signal.oaconvolve(data * weight, kernel, mode="same")
    ww_lp = signal.oaconvolve(weight, kernel, mode="same")

    return vw_lp * tools.invert_no_zero(ww_lp)


def highpass_weighted_convolution_filter(
    data: npt.NDArray[np.floating | np.complexfloating],
    weight: npt.NDArray[np.floating | np.complexfloating],
    samples: npt.NDArray[np.floating | np.complexfloating],
    cutoff: float,
    axis: int = -1,
) -> np.ndarray[np.floating | np.complexfloating]:
    """Apply a crude high-pass weighted convolution filter along an axis.

    Filter is applied by subtracting a low-pass filtered version of the data.

    Parameters
    ----------
    data
        The data to filter.
    weight
        The weights to apply.
    samples
        The sample times.
    cutoff
        The cutoff frequency.
    axis
        The axis to apply the filter along.
    """
    datalp = lowpass_weighted_convolution_filter(
        data, weight, samples, cutoff, axis=axis
    )

    return data - datalp


def medfilt(
    x: npt.NDArray[np.floating | np.complexfloating],
    mask: npt.NDArray[np.bool_],
    size: tuple[int],
    *args,
) -> npt.NDArray[np.floating | np.complexfloating]:
    """Apply a moving median filter to masked data.

    Parameters
    ----------
    x
        Data to filter.
    mask
        Mask of data to filter out.
    size
        Size of the window in each dimension.
    args
        Additional arguments to pass to the moving weighted median

    Returns
    -------
    y
        The masked data. Data within the mask is undefined.
    """
    if np.iscomplexobj(x):
        return medfilt(x.real, mask, size) + 1.0j * medfilt(x.imag, mask, size)

    # Copy and do initial masking
    x = np.ascontiguousarray(x.astype(np.float64))
    w = np.ascontiguousarray((~mask).astype(np.float64))

    return weighted_median.moving_weighted_median(x, w, size, *args)


def null_filter(
    samples: npt.NDArray[np.floating],
    cutoff: float,
    mask: npt.NDArray[np.bool_],
    num_modes: int = 200,
    tol: float = 1e-8,
    window: str | bool = True,
    type_: str = "high",
    lapack_driver: str = "gesvd",
) -> np.ndarray[np.floating, 2]:
    """Create a high-pass or low-pass filter by nulling Fourier modes.

    Parameters
    ----------
    samples
        Samples we have data at.
    cutoff
        Fourier inverse cut to apply.
    mask
        Samples to mask out.
    num_modes
        Number of fourier samples to use in the range -cutoff to +cutoff.
    tol
        Cutoff value for singular values.
    window
        Apply a window function to the data while filtering.
    type_
        Whether to apply a high-pass or low-pass filter. Options are
        `high` or `low`. Default is `high`.
    lapack_driver
        Which lapack driver to use in the SVD. Options are 'gesvd' or 'gesdd'.
        'gesdd' is generally faster, but seems to experience convergence issues.
        Default is 'gesvd'.

    Returns
    -------
    filter
        The filter as a 2D matrix.
    """
    if type_ not in {"high", "low"}:
        raise ValueError(f"Filter type must be one of [high, low]. Got {type_}")

    fmodes = np.linspace(-cutoff, cutoff, num_modes)

    # Construct the Fourier matrix
    F = mask[:, np.newaxis] * np.exp(
        2.0j * np.pi * fmodes[np.newaxis, :] * samples[:, np.newaxis]
    )

    if window:
        # Construct the window function
        x = (samples - samples.min()) / np.ptp(samples)
        window = "nuttall" if window is True else window
        w = window_generalised(x, window=window)

        F *= w[:, np.newaxis]

    # Use an SVD to figure out the set of significant modes spanning the delays
    # we are wanting to get rid of.
    # NOTE: we've experienced some convergence failures in here which ultimately seem
    # to be the fault of MKL (see https://github.com/scipy/scipy/issues/10032 and links
    # therein). This seems to be limited to the `gesdd` LAPACK routine, so we can get
    # around it by switching to `gesvd`.
    u, sig, _ = la.svd(F, full_matrices=False, lapack_driver=lapack_driver)
    nmodes = np.sum(sig > tol * sig.max())
    p = u[:, :nmodes]

    # Construct a projection matrix for the filter
    proj = p @ p.T.conj()

    if type_ == "high":
        proj = np.identity(samples.size) - proj

    # Multiply in the mask and window (if applicable)
    proj *= mask[np.newaxis, :]

    if window:
        proj *= w[np.newaxis, :]

    return proj

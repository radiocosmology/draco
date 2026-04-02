"""Collection of routines for RFI excision."""

import numpy as np
import numpy.typing as npt
from scipy.ndimage import correlate1d


def sumthreshold_py(
    data,
    max_m=16,
    start_flag=None,
    threshold1=None,
    remove_median=True,
    correct_for_missing=True,
    variance=None,
    rho=None,
    axes=None,
    only_positive=False,
):
    """SumThreshold outlier detection algorithm.

    See https://andreoffringa.org/pdfs/SumThreshold-technical-report.pdf
    for description of the algorithm.

    Parameters
    ----------
    data : np.ndarray[:, :]
        The data to flag.
    max_m : int, optional
        Maximum size to expand to.
    start_flag : np.ndarray[:, :], optional
        A boolean array of the initially flagged data.
    threshold1 : float, optional
        Initial threshold. By default use the 95 percentile.
    remove_median : bool, optional
        Subtract the median of the full 2D dataset. Default is True.
    correct_for_missing : bool, optional
        Correct for missing counts
    variance : np.ndarray[:, :], optional
        Estimate of the uncertainty on each data point.
        If provided, then correct_for_missing=True should be set
        and threshold1 should be provided in units of "sigma".
    rho : float, optional
        Controls the dependence of the threshold on the window size m,
        specifically threshold = threshold1 / rho ** log2(m).
        If not provided, will use a value of 1.5 (0.9428)
        when correct_for_missing is False (True).  This is to maintain
        backward compatibility.
    axes : tuple | int, optional
        Axes of `data` along which to calculate. Flagging is done in
        the order in which `axes` is provided. By default, loop over
        all axes in reverse order.
    only_positive : bool, optional
        Only flag positive excursions, do not flag negative excursions.

    Returns
    -------
    mask : np.ndarray[:, :]
        Boolean array, with `True` entries marking outlier data.
    """
    data = np.copy(data)

    # If the variance was provided, then we will need to take the
    # square root of the sum of the variances prior to thresholding.
    # Make sure correct_for_missing is set to True:
    if variance is not None:
        correct_for_missing = True

    # If rho was not provided, then use the backwards-compatible values
    if rho is None:
        rho = 0.9428 if correct_for_missing else 1.5

    # Function that determines if we flag all excursions or only positive excursions
    def get_sign(x):
        return x if only_positive else np.abs(x)

    # Determine what axes to flag over
    if axes is None:
        # Iterate over axes in reverse order
        axes = list(range(data.ndim))[::-1]
    elif isinstance(axes, int):
        axes = (axes,)

    # By default, flag anything that is not finite (inf and NaN)
    flag = ~np.isfinite(data)

    if start_flag is not None:
        flag += start_flag

    if remove_median:
        data -= np.median(data[~flag])

    if threshold1 is None:
        if variance is not None:
            raise RuntimeError(
                "If variance is provided, then must also "
                "provide starting threshold in units of sigma."
            )
        threshold1 = np.percentile(data[~flag], 95.0)

    m = 1

    while m <= max_m:
        threshold = threshold1 / rho ** (np.log2(m))

        # The centre of the window for even windows is the bin right to the left of
        # centre. I want the origin at the leftmost bin
        centre = (m - 1) // 2

        # Convolution with the kernel is effectively just a windowed sum
        kernel = np.ones(m, dtype=np.float64)

        # Loop over axes in order
        for axis in axes:
            # Update the data mask
            data[flag] = 0.0
            count = (~flag).astype(np.float64) if variance is None else ~flag * variance

            # Convolve the data and counts with the kernel, extending the
            # boundaries based on the edge values. Setting `origin=centre`
            # results in a peak at the *rightmost* edge of the region to
            # be flagged.
            dconv = correlate1d(data, kernel, origin=centre, axis=axis, mode="nearest")
            cconv = correlate1d(count, kernel, origin=centre, axis=axis, mode="nearest")

            if correct_for_missing:
                cconv = cconv**0.5

            temp_flag = get_sign(dconv) > cconv * threshold
            # Extend the mask to symmetrically cover flagged samples. `origin` is
            # -centre if m is odd and -centre-1 if m is even. This extends the
            # flag to the *left* and correctly centers the flagged region
            origin = m % 2 - centre - 1
            flag += correlate1d(
                temp_flag, kernel, origin=origin, axis=axis, mode="nearest"
            )

        m *= 2

    return flag


# This routine might be substituted by a faster one later
sumthreshold = sumthreshold_py


def _sir_lastaxis(
    basemask: npt.NDArray[np.bool_], eta: float = 0.2, axis=-1
) -> npt.NDArray[np.bool]:
    """Numpy implementation of the scale-invariant rank (SIR) operator.

    For more information, see arXiv:1201.3364v2.

    Parameters
    ----------
    basemask
        Array with the threshold mask previously generated.
        1 (True) for flagged points, 0 (False) otherwise.
    eta
        Aggressiveness of the method: with eta=0, no additional samples are
        flagged and the function returns basemask. With eta=1, all samples
        will be flagged. The authors in arXiv:1201.3364v2 seem to be convinced
        that 0.2 is a mostly universally optimal value, but no optimization
        has been done on CHIME data.
    axis
        Axis along which to apply the SIR operator. Default is -1,
        which applies it along the last axis.

    Returns
    -------
    mask
        The mask after the application of the (SIR) operator. Same shape and
        type as basemask.
    """
    # Move the filtetr axis to the end and copy to ensure contiguous array
    basemask = np.moveaxis(basemask, axis, -1).copy()

    M = np.zeros((*basemask.shape[:-1], basemask.shape[-1] + 1), dtype=np.float64)
    # Copy the basemask into the appropriate slice of M,
    # which will automatically cast to float64. Use `M` to
    # avoid an additional allocation
    M[..., 1:] = basemask
    M[..., 1:] += eta - 1.0

    # cumulative sum stored directly into `M` to avoid
    # an additional allocation
    # NOTE: neither `cumsum` not `accumulate` seem to use
    # multiple threads, so this seems like a fairly easy place
    # for significant performance improvement
    np.cumsum(M[..., 1:], axis=-1, out=M[..., 1:])

    MP = np.minimum.accumulate(M, axis=-1)[..., :-1]
    # Re-use `M` again
    np.maximum.accumulate(M[..., -2::-1], axis=-1, out=M[..., -2::-1])

    basemask |= M[..., 1:] >= MP

    return np.moveaxis(basemask, -1, axis)


# Alias for backward compatibility
sir1d = _sir_lastaxis


def scale_invariant_rank(
    basemask: npt.NDArray[np.bool_],
    eta: float | tuple[float, ...] = 0.2,
    axis: int | tuple[int, ...] = -1,
) -> npt.NDArray[np.bool]:
    """Apply the scale-invariant rank (SIR) operator along one or more axes.

    This is a wrapper for `sir1d`.  It loops over the provided axes, applying
    `sir1d` along each axis in turn.  It returns the logical OR of these masks.

    Parameters
    ----------
    basemask
        The previously generated threshold mask.
        1 (True) for masked points, 0 (False) otherwise.
    eta
        Aggressiveness of the method: with eta=0, no additional samples are
        flagged and the function returns basemask. With eta=1, all samples
        will be flagged. If a tuple is provided, it must have the same length
        as `axis`, and the corresponding value will be used for each axis.
    axis
        Axis or axes along which to apply the SIR operator. Default is -1,
        which applies it along the last axis.

    Returns
    -------
    mask
        The mask after the application of the SIR operator.
    """
    if basemask.ndim < 1:
        raise ValueError("basemask must have at least one dimension.")

    if isinstance(axis, int):
        axis = (axis,)

    if isinstance(eta, float | int):
        eta = (eta,) * len(axis)

    if len(eta) != len(axis):
        raise ValueError(
            "If eta is a tuple, it must have the same length as axis."
            f"Got len(eta)={len(eta)} and len(axis)={len(axis)}."
        )

    # Avoids an unnecessary copy in the case where only one axis is requested
    newmask = _sir_lastaxis(basemask, eta=eta[0], axis=axis[0])

    # Iterate over the axes, applying sir1d to the
    # base mask along each axis
    for ax, et in zip(axis[1:], eta[1:]):
        newmask |= _sir_lastaxis(basemask, eta=et, axis=ax)

    return newmask


def sir(
    basemask: npt.NDArray[np.bool_],
    eta: float = 0.2,
    only_freq: bool = False,
    only_time: bool = False,
) -> npt.NDArray[np.bool]:
    """Apply the SIR operator over the frequency and time axes for each product.

    This is a wrapper for `sir1d`.  It loops over times, applying `sir1d`
    across the frequency axis.  It then loops over frequencies, applying `sir1d`
    across the time axis.  It returns the logical OR of these two masks.

    Parameters
    ----------
    basemask
        The previously generated threshold mask.
        1 (True) for masked points, 0 (False) otherwise.
    eta
        Aggressiveness of the method: with eta=0, no additional samples are
        flagged and the function returns basemask. With eta=1, all samples
        will be flagged.
    only_freq
        Only apply the SIR operator across the frequency axis.
    only_time
        Only apply the SIR operator across the time axis.

    Returns
    -------
    mask
        The mask after the application of the SIR operator.
    """
    import warnings

    warnings.warn(
        "The sir function is deprecated and will be removed in a future release. "
        "Please use the updated `scale_invariant_rank` function with `axis=(0, -1)`.",
        DeprecationWarning,
    )

    if basemask.ndim != 3:
        raise ValueError(
            "basemask must be a 3D array with [freq, prod, time] axes. "
            f"Got {basemask.ndim}D array instead."
        )

    if only_freq and only_time:
        raise ValueError("Only one of only_freq and only_time can be True.")

    nfreq, nprod, ntime = basemask.shape

    newmask = basemask.astype(bool).copy()

    for pp in range(nprod):
        if not only_time:
            for tt in range(ntime):
                newmask[:, pp, tt] |= _sir_lastaxis(basemask[:, pp, tt], eta=eta)

        if not only_freq:
            for ff in range(nfreq):
                newmask[ff, pp, :] |= _sir_lastaxis(basemask[ff, pp, :], eta=eta)

    return newmask

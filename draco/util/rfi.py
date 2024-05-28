"""Collection of routines for RFI excision."""

import numpy as np
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
        If provided, then correct_missing=True should be set
        and threshold1 should be provided in units of "sigma".
    rho : float, optional
        Controls the dependence of the threshold on the window size m,
        specifically threshold = threshold1 / rho ** log2(m).
        If not provided, will use a value of 1.5 (0.9428)
        when correct_missing is False (True).  This is to maintain
        backward compatibility.
    axes : tuple | int, optional
        Axes of `data` along which to calculate. Flagging is done in
        the order in which `axes` is provided. By default, loop over
        all axes in reverse order.

    Returns
    -------
    mask : np.ndarray[:, :]
        Boolean array, with `True` entries marking outlier data.
    """
    data = np.copy(data)

    # If the variance was provided, then we will need to take the
    # square root of the sum of the variances prior to thresholding.
    # Make sure correct_missing is set to True:
    if variance is not None:
        correct_missing = True

    # If rho was not provided, then use the backwards-compatible values
    if rho is None:
        rho = 0.9428 if correct_missing else 1.5

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

            temp_flag = np.abs(dconv) > cconv * threshold
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


# Scale-invariant rank (SIR) functions
def sir1d(basemask, eta=0.2):
    """Numpy implementation of the scale-invariant rank (SIR) operator.

    For more information, see arXiv:1201.3364v2.

    Parameters
    ----------
    basemask : numpy 1D array of boolean type
        Array with the threshold mask previously generated.
        1 (True) for flagged points, 0 (False) otherwise.
    eta : float
        Aggressiveness of the method: with eta=0, no additional samples are
        flagged and the function returns basemask. With eta=1, all samples
        will be flagged. The authors in arXiv:1201.3364v2 seem to be convinced
        that 0.2 is a mostly universally optimal value, but no optimization
        has been done on CHIME data.

    Returns
    -------
    mask : numpy 1D array of boolean type
        The mask after the application of the (SIR) operator. Same shape and
        type as basemask.
    """
    n = basemask.size
    psi = basemask.astype(np.float64) - 1.0 + eta

    M = np.zeros(n + 1, dtype=np.float64)
    M[1:] = np.cumsum(psi)

    MP = np.minimum.accumulate(M)[:-1]
    MQ = np.concatenate((np.maximum.accumulate(M[-2::-1])[-2::-1], M[-1, np.newaxis]))

    return (MQ - MP) >= 0.0


def sir(basemask, eta=0.2, only_freq=False, only_time=False):
    """Apply the SIR operator over the frequency and time axes for each product.

    This is a wrapper for `sir1d`.  It loops over times, applying `sir1d`
    across the frequency axis.  It then loops over frequencies, applying `sir1d`
    across the time axis.  It returns the logical OR of these two masks.

    Parameters
    ----------
    basemask : np.ndarray[nfreq, nprod, ntime] of boolean type
        The previously generated threshold mask.
        1 (True) for masked points, 0 (False) otherwise.
    eta : float
        Aggressiveness of the method: with eta=0, no additional samples are
        flagged and the function returns basemask. With eta=1, all samples
        will be flagged.
    only_freq : bool
        Only apply the SIR operator across the frequency axis.
    only_time : bool
        Only apply the SIR operator across the time axis.

    Returns
    -------
    mask : np.ndarray[nfreq, nprod, ntime] of boolean type
        The mask after the application of the SIR operator.
    """
    if only_freq and only_time:
        raise ValueError("Only one of only_freq and only_time can be True.")

    nfreq, nprod, ntime = basemask.shape

    newmask = basemask.astype(bool).copy()

    for pp in range(nprod):
        if not only_time:
            for tt in range(ntime):
                newmask[:, pp, tt] |= sir1d(basemask[:, pp, tt], eta=eta)

        if not only_freq:
            for ff in range(nfreq):
                newmask[ff, pp, :] |= sir1d(basemask[ff, pp, :], eta=eta)

    return newmask

"""Collection of routines for RFI excision."""
import numpy as np
from scipy.ndimage import convolve1d


def sumthreshold_py(
    data,
    max_m=16,
    start_flag=None,
    threshold1=None,
    remove_median=True,
    correct_for_missing=True,
):
    """SumThreshold outlier detection algorithm.

    See http://www.astro.rug.nl/~offringa/SumThreshold.pdf for description of
    the algorithm.

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

    Returns
    -------
    mask : np.ndarray[:, :]
        Boolean array, with `True` entries marking outlier data.
    """
    data = np.copy(data)
    (ny, nx) = data.shape

    if start_flag is None:
        start_flag = np.isnan(data)
    flag = np.copy(start_flag)

    if remove_median:
        data -= np.median(data[~flag])

    if threshold1 is None:
        threshold1 = np.percentile(data[~flag], 95.0)

    m = 1
    while m <= max_m:
        if m == 1:
            threshold = threshold1
        else:
            threshold = threshold1 / 1.5 ** (np.log2(m))

        # The centre of the window for even windows is the bin right to the left of
        # centre. I want the origin at the leftmost bin
        if m == 1:
            centre = 0
        else:
            centre = m // 2 - 1

        ## X-axis

        data[flag] = 0.0
        count = (~flag).astype(np.float64)

        # Convolution of the data
        dconv = convolve1d(
            data, weights=np.ones(m, dtype=float), origin=-centre, axis=1
        )[:, : (nx - m + 1)]

        # Convolution of the counts
        cconv = convolve1d(
            count, weights=np.ones(m, dtype=float), origin=-centre, axis=1
        )[:, : (nx - m + 1)]
        if correct_for_missing:
            cconv = m**0.5 * cconv**0.5
        flag_temp = dconv > cconv * threshold
        flag_temp += dconv < -cconv * threshold
        for ii in range(flag_temp.shape[1]):
            flag[:, ii : (ii + m)] += flag_temp[:, ii][:, np.newaxis]

        ## Y-axis

        data[flag] = 0.0
        count = (~flag).astype(np.float64)
        # Convolution of the data
        dconv = convolve1d(
            data, weights=np.ones(m, dtype=float), origin=-centre, axis=0
        )[: (ny - m + 1), :]
        # Convolution of the counts
        cconv = convolve1d(
            count, weights=np.ones(m, dtype=float), origin=-centre, axis=0
        )[: (ny - m + 1), :]
        if correct_for_missing:
            cconv = m**0.5 * cconv**0.5
        flag_temp = dconv > cconv * threshold
        flag_temp += dconv < -cconv * threshold

        for ii in range(flag_temp.shape[0]):
            flag[ii : ii + m, :] += flag_temp[ii, :][np.newaxis, :]

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

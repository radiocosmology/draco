"""Collection of miscellaneous routines.

Miscellaneous tasks should be placed in :module:`draco.core.misc`.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np

from ._fast_tools import _calc_redundancy


def cmap(i, j, n):
    """Given a pair of feed indices, return the pair index.

    Parameters
    ----------
    i, j : integer
        Feed index.
    n : integer
        Total number of feeds.

    Returns
    -------
    pi : integer
        Pair index.
    """
    if i <= j:
        return (n * (n + 1) // 2) - ((n - i) * (n - i + 1) // 2) + (j - i)
    else:
        return cmap(j, i, n)


def icmap(ix, n):
    """Inverse feed map.

    Parameters
    ----------
    ix : integer
        Pair index.
    n : integer
        Total number of feeds.

    Returns
    -------
    fi, fj : integer
        Feed indices.
    """
    for ii in range(n):
        if cmap(ii, n - 1, n) >= ix:
            break

    i = ii
    j = ix - cmap(i, i, n) + i
    return i, j


def apply_gain(vis, gain, axis=1, out=None, prod_map=None):
    """Apply per input gains to a set of visibilities packed in upper
    triangular format.

    This allows us to apply the gains while minimising the intermediate
    products created.

    Parameters
    ----------
    vis : np.ndarray[..., nprod, ...]
        Array of visibility products.
    gain : np.ndarray[..., ninput, ...]
        Array of gains. One gain per input.
    axis : integer, optional
        The axis along which the inputs (or visibilities) are
        contained.
    out : np.ndarray
        Array to place output in. If :obj:`None` create a new
        array. This routine can safely use `out = vis`.
    prod_map : ndarray of integer pairs
        Gives the mapping from product axis to input pairs. If not supplied,
        :func:`icmap` is used.

    Returns
    -------
    out : np.ndarray
        Visibility array with gains applied. Same shape as :obj:`vis`.
    """

    nprod = vis.shape[axis]
    ninput = gain.shape[axis]

    if prod_map is None and nprod != (ninput * (ninput + 1) // 2):
        raise Exception("Number of inputs does not match the number of products.")

    if prod_map is not None:
        if len(prod_map) != nprod:
            msg = "Length of *prod_map* does not match number of input" " products."
            raise ValueError(msg)
        # Could check prod_map contents as well, but the loop should give a
        # sensible error if this is wrong, and checking is expensive.
    else:
        prod_map = [icmap(pp, ninput) for pp in range(nprod)]

    if out is None:
        out = np.empty_like(vis)
    elif out.shape != vis.shape:
        raise Exception("Output array is wrong shape.")

    # Define slices for use in gain & vis selection & combination
    gain_vis_slice = tuple(slice(None) for i in range(axis))

    # Iterate over input pairs and set gains
    for pp in range(nprod):

        # Determine the inputs.
        ii, ij = prod_map[pp]

        # Fetch the gains
        gi = gain[gain_vis_slice + (ii,)]
        gj = gain[gain_vis_slice + (ij,)].conj()

        # Apply the gains and save into the output array.
        out[gain_vis_slice + (pp,)] = vis[gain_vis_slice + (pp,)] * gi * gj

    return out


def invert_no_zero(x):
    """Return the reciprocal, but ignoring zeros.

    Where `x != 0` return 1/x, or just return 0. Importantly this routine does
    not produce a warning about zero division.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    r : np.ndarray
        Return the reciprocal of x.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 0.0, 1.0 / x)


def extract_diagonal(utmat, axis=1):
    """Extract the diagonal elements of an upper triangular array.

    Parameters
    ----------
    utmat : np.ndarray[..., nprod, ...]
        Upper triangular array.
    axis : int, optional
        Axis of array that is upper triangular.

    Returns
    -------
    diag : np.ndarray[..., ninput, ...]
        Diagonal of the array.
    """

    # Estimate nside from the array shape
    nside = int((2 * utmat.shape[axis]) ** 0.5)

    # Check that this nside is correct
    if utmat.shape[axis] != (nside * (nside + 1) // 2):
        msg = (
            "Array length (%i) of axis %i does not correspond upper triangle\
                of square matrix"
            % (utmat.shape[axis], axis)
        )
        raise RuntimeError(msg)

    # Find indices of the diagonal
    diag_ind = [cmap(ii, ii, nside) for ii in range(nside)]

    # Construct slice objects representing the axes before and after the product axis
    slice0 = (np.s_[:],) * axis
    slice1 = (np.s_[:],) * (len(utmat.shape) - axis - 1)

    # Extract wanted elements with a giant slice
    sl = slice0 + (diag_ind,) + slice1
    diag_array = utmat[sl]

    return diag_array


def calculate_redundancy(input_flags, prod_map, stack_index, nstack):
    """Calculates the number of redundant baselines that were stacked
    to form each unique baseline, accounting for the fact that some fraction
    of the inputs are flagged as bad at any given time.

    Parameters
    ----------
    input_flags : np.ndarray [ninput, ntime]
        Array indicating which inputs were good at each time.
        Non-zero value indicates that an input was good.

    prod_map: np.ndarray[nprod]
        The products that were included in the stack.
        Typically found in the `index_map['prod']` attribute of the
        `containers.TimeStream` or `containers.SiderealStream` object.

    stack_index: np.ndarray[nprod]
        The index of the stack axis that each product went into.
        Typically found in `reverse_map['stack']['stack']` attribute
        of the `containers.Timestream` or `containers.SiderealStream` object.

    nstack: int
        Total number of unique baselines.

    Returns
    -------
    redundancy : np.ndarray[nstack, ntime]
        Array indicating the total number of redundant baselines
        with good inputs that were stacked into each unique baseline.

    """
    ninput, ntime = input_flags.shape
    redundancy = np.zeros((nstack, ntime), dtype=np.float32)

    if not np.any(input_flags):
        input_flags = np.ones_like(input_flags)

    input_flags = np.ascontiguousarray(input_flags)
    pm = np.ascontiguousarray(prod_map.view(np.int16).reshape(-1, 2))
    stack_index = np.ascontiguousarray(stack_index)

    # Call fast cython function to do calculation
    _calc_redundancy(input_flags, pm, stack_index.copy(), nstack, redundancy)

    return redundancy


def polarization_map(index_map, telescope, exclude_autos=True):
    """ Map the visibilities corresponding to entries in
        pol = ['XX', 'YY', 'XY', 'YX'].

        Parameters
        ----------
        index_map : h5py.group or dict
            Index map to map into polarizations. Must contain a `stack`
            entry and an `input` entry.
        telescope : :class: `drift.core.telescope`
            Telescope object containing feed information.
        exclude_autos: bool
            If True (default), auto-correlations are set to -1.

        Returns
        -------
        polmap : array of int
            Array of size `nstack`. Each entry is the index to the
            corresponding polarization in pol = ['XX', 'YY', 'XY', 'YX']

    """
    pol = ['XX', 'YY', 'XY', 'YX']
    nstack = len(index_map['stack'])
    # polmap: indices of each vis product in
    # polarization list: ['XX', 'YY', 'XY', 'YX']
    polmap = np.zeros(nstack, dtype=int)
    # For each entry in stack
    for vi in range(nstack):
        # Product index
        pi = index_map['stack'][vi][0]
        # Inputs that go into this product
        ipt0 = index_map['input']['chan_id'][
                            index_map['prod'][pi][0]]
        ipt1 = index_map['input']['chan_id'][
                            index_map['prod'][pi][1]]

        # Exclude autos if exclude_autos == True
        if (exclude_autos and (ipt0 == ipt1)):
            polmap[vi] = -1
            continue

        # Are feeds on?
        onfeeds = (tools.is_array_on(telescope._feeds[ipt0]) and
                   tools.is_array_on(telescope._feeds[ipt1]))
        # Feeds are off, exclude from polmap
        if not onfeeds:
            polmap[vi] = -1
            continue

        # If feeds are on, find both polarizations and populate polmap
        if onfeeds:
            # Find polarization of first input
            if tools.is_array_x(telescope._feeds[ipt0]):
                polstring = 'X'
            elif tools.is_array_y(telescope._feeds[ipt0]):
                polstring = 'Y'
            else:
                # We should not get to this case.
                polmap[vi] = -1
                continue
            # Find polarization of second input and add it to polstring
            if tools.is_array_x(telescope._feeds[ipt1]):
                polstring += 'X'
            elif tools.is_array_y(telescope._feeds[ipt1]):
                polstring += 'Y'
            else:
                # We should not get to this case.
                polmap[vi] = -1
                continue
            # If conjugate, flip polstring ('XY -> 'YX)
            if telescope.feedconj[ipt0, ipt1]:
                polstring = polstring[::-1]
            # Populate polmap
            polmap[vi] = pol.index(polstring)

    return polmap


def baseline_vector(index_map, telescope):
    """ Baseline vectors in meters.

        Parameters
        ----------
        index_map : h5py.group or dict
            Index map to map into polarizations. Must contain a `stack`
            entry and an `input` entry.
        telescope : :class: `drift.core.telescope`
            Telescope object containing feed information.

        Returns
        -------
        bvec_m : array
            Array of shape (2, nstack). The 2D baseline vector
            (in meters) for each visibility in index_map['stack']
    """
    nstack = len(index_map['stack'])
    # Baseline vectors in meters.
    bvec_m = np.zeros((2, nstack), dtype=np.float64)
    # Compute all baseline vectors.
    for vi in range(nstack):
        # Product index
        pi = index_map['stack'][vi][0]
        # Inputs that go into this product
        ipt0 = index_map['input']['chan_id'][
                            index_map['prod'][pi][0]]
        ipt1 = index_map['input']['chan_id'][
                            index_map['prod'][pi][1]]

        # Beseline vector in meters
        unique_index = telescope.feedmap[ipt0, ipt1]
        bvec_m[:, vi] = telescope.baselines[unique_index]
        # No need to conjugate. Already done in telescope.baselines.
        #if telescope.feedconj[ipt0, ipt1]:
        #    bvec_m[:, vi] *= -1.

    return bvec_m

"""Collection of miscellaneous routines.

Miscellaneous tasks should be placed in :py:mod:`draco.core.misc`.
"""

import numpy as np

# Keep this here for compatibility
from caput.tools import invert_no_zero  # noqa: F401
from numpy.lib.recfunctions import structured_to_unstructured

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


def find_key(key_list, key):
    """Find the index of a key in a list of keys.

    This is a wrapper for the list method `index`
    that can search any interable (not just lists)
    and will return None if the key is not found.

    Parameters
    ----------
    key_list : iterable
        Iterable containing keys to search
    key : object to be searched
        Keys to search for

    Returns
    -------
    index : int or None
        The index of `key` in `key_list`.
        If `key_list` does not contain `key`,
        then None is returned.
    """
    try:
        return [tuple(x) for x in key_list].index(tuple(key))
    except TypeError:
        return list(key_list).index(key)
    except ValueError:
        return None


def find_keys(key_list, keys, require_match=False):
    """Find the indices of keys into a list of keys.

    Parameters
    ----------
    key_list : iterable
        Iterable of keys to search
    keys : iterable
        Keys to search for
    require_match : bool
        Require that `key_list` contain every element of `keys`,
        and if not, raise ValueError.

    Returns
    -------
    indices : list of int or None
        List of the same length as `keys` containing
        the indices of `keys` in `key_list`.  If `require_match`
        is False, then this can also contain None for keys
        that are not contained in `key_list`.
    """
    # Significantly faster than repeated calls to find_key
    try:
        dct = {tuple(kk): ii for ii, kk in enumerate(key_list)}
        index = [dct.get(tuple(key)) for key in keys]
    except TypeError:
        dct = {kk: ii for ii, kk in enumerate(key_list)}
        index = [dct.get(key) for key in keys]

    if require_match and any(ind is None for ind in index):
        raise ValueError("Could not find all of the keys.")

    return index


def find_inputs(input_index, inputs, require_match=False):
    """Find the indices of inputs into a list of inputs.

    This behaves similarly to `find_keys` but will automatically choose the key to
    match on.

    Parameters
    ----------
    input_index : np.ndarray
        Inputs to search
    inputs : np.ndarray
        Inputs to find
    require_match : bool
        Require that `input_index` contain every element of `inputs`,
        and if not, raise ValueError.

    Returns
    -------
    indices : list of int or None
        List of the same length as `inputs` containing
        the indices of `inputs` in `input_inswx`.  If `require_match`
        is False, then this can also contain None for inputs
        that are not contained in `input_index`.
    """
    # Significantly faster than repeated calls to find_key

    if "correlator_input" in input_index.dtype.fields:
        field_to_match = "correlator_input"
    elif "chan_id" in input_index.dtype.fields:
        field_to_match = "chan_id"
    else:
        raise ValueError(
            "`input_index` must have either a `chan_id` or `correlator_input` field."
        )

    if field_to_match not in inputs.dtype.fields:
        raise ValueError(f"`inputs` array does not have a `{field_to_match!s}` field.")

    return find_keys(
        input_index[field_to_match], inputs[field_to_match], require_match=require_match
    )


def apply_gain(vis, gain, axis=1, out=None, prod_map=None):
    """Apply per input gains to a set of visibilities packed in upper triangular format.

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
        gi = gain[(*gain_vis_slice, ii)]
        gj = gain[(*gain_vis_slice, ij)].conj()

        # Apply the gains and save into the output array.
        out[(*gain_vis_slice, pp)] = vis[(*gain_vis_slice, pp)] * gi * gj

    return out


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
    sl = (*slice0, diag_ind, *slice1)
    return utmat[sl]


def calculate_redundancy(input_flags, prod_map, stack_index, nstack):
    """Calculates the number of redundant baselines that were stacked to form each unique baseline.

    Accounts for the fact that some fraction of the inputs are flagged as bad at any given time.

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

    input_flags = np.ascontiguousarray(input_flags.astype(np.float32, copy=False))
    pm = structured_to_unstructured(prod_map, dtype=np.int16)
    stack_index = np.ascontiguousarray(stack_index.astype(np.int32, copy=False))

    # Call fast cython function to do calculation
    _calc_redundancy(input_flags, pm, stack_index, nstack, redundancy)

    return redundancy


def redefine_stack_index_map(telescope, inputs, prod, stack, reverse_stack):
    """Ensure baselines between unmasked inputs are used to represent each stack.

    Parameters
    ----------
    telescope : :class: `drift.core.telescope`
        Telescope object containing feed information.
    inputs : np.ndarray[ninput,] of dtype=('correlator_input', 'chan_id')
        The 'correlator_input' or 'chan_id' of the inputs in the stack.
    prod : np.ndarray[nprod,] of dtype=('input_a', 'input_b')
        The correlation products as pairs of inputs.
    stack : np.ndarray[nstack,] of dtype=('prod', 'conjugate')
        The index into the `prod` axis of a characteristic baseline included in the stack.
    reverse_stack :  np.ndarray[nprod,] of dtype=('stack', 'conjugate')
        The index into the `stack` axis that each `prod` belongs.

    Returns
    -------
    stack_new : np.ndarray[nstack,] of dtype=('prod', 'conjugate')
        The updated `stack` index map, where each element is an index to a product
        consisting of a pair of unmasked inputs.
    stack_flag : np.ndarray[nstack,] of dtype=bool
        Boolean flag that is True if this element of the stack index map is now valid,
        and False if none of the baselines that were stacked contained unmasked inputs.
    """
    # Determine mapping between inputs in the index_map and
    # inputs in the telescope instance
    tel_index = find_inputs(telescope.input_index, inputs, require_match=False)

    # Create a copy of the stack axis
    stack_new = stack.copy()
    stack_flag = np.zeros(stack_new.size, dtype=bool)

    # Loop over the stacked baselines
    for sind, (ii, jj) in enumerate(prod[stack["prod"]]):
        bi, bj = tel_index[ii], tel_index[jj]

        # Check that the represenative pair of inputs are present
        # in the telescope instance and not masked.
        if (bi is None) or (bj is None) or not telescope.feedmask[bi, bj]:
            # Find alternative pairs of inputs using the reverse map
            this_stack = np.flatnonzero(reverse_stack["stack"] == sind)

            # Loop over alternatives until we find an acceptable pair of inputs
            for ts in this_stack:
                tp = prod[ts]
                ti, tj = tel_index[tp[0]], tel_index[tp[1]]
                if (ti is not None) and (tj is not None) and telescope.feedmask[ti, tj]:
                    stack_new[sind]["prod"] = ts
                    stack_new[sind]["conjugate"] = reverse_stack[ts]["conjugate"]
                    stack_flag[sind] = True
                    break
        else:
            stack_flag[sind] = True

    return stack_new, stack_flag


def polarization_map(index_map, telescope, exclude_autos=True):
    """Map the visibilities corresponding to entries in pol = ['XX', 'XY', 'YX', 'YY'].

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
        corresponding polarization in pol = ['XX', 'XY', 'YX', 'YY']
    """
    # Old versions of telescope object don't have the `stack_type`
    # attribute. Assume those are of type `redundant`.
    try:
        teltype = telescope.stack_type
    except AttributeError:
        teltype = None
        msg = (
            "Telescope object does not have a `stack_type` attribute.\n"
            + "Assuming it is of type `redundant`"
        )

    if teltype is not None:
        if not (teltype == "redundant"):
            msg = "Telescope stack type needs to be 'redundant'. Is {0}"
            raise RuntimeError(msg.format(telescope.stack_type))

    # Older data's input map has a simpler dtype
    try:
        input_map = index_map["input"]["chan_id"][:]
    except IndexError:
        input_map = index_map["input"][:]

    pol = ["XX", "XY", "YX", "YY"]
    nstack = len(index_map["stack"])
    # polmap: indices of each vis product in
    # polarization list: ['XX', 'YY', 'XY', 'YX']
    polmap = np.zeros(nstack, dtype=int)
    # For each entry in stack
    for vi in range(nstack):
        # Product index
        pi = index_map["stack"][vi][0]
        # Inputs that go into this product
        ipt0 = input_map[index_map["prod"][pi][0]]
        ipt1 = input_map[index_map["prod"][pi][1]]

        # Exclude autos if exclude_autos == True
        if exclude_autos and (ipt0 == ipt1):
            polmap[vi] = -1
            continue

        # Find polarization of first input
        if telescope.beamclass[ipt0] == 0:
            polstring = "X"
        elif telescope.beamclass[ipt0] == 1:
            polstring = "Y"
        else:
            # Not a CHIME feed or not On. Ignore.
            polmap[vi] = -1
            continue
        # Find polarization of second input and add it to polstring
        if telescope.beamclass[ipt1] == 0:
            polstring += "X"
        elif telescope.beamclass[ipt1] == 1:
            polstring += "Y"
        else:
            # Not a CHIME feed or not On. Ignore.
            polmap[vi] = -1
            continue
        # If conjugate, flip polstring ('XY -> 'YX)
        if telescope.feedconj[ipt0, ipt1]:
            polstring = polstring[::-1]
        # Populate polmap
        polmap[vi] = pol.index(polstring)

    return polmap


def baseline_vector(index_map, telescope):
    """Baseline vectors in meters.

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
    nstack = len(index_map["stack"])
    # Baseline vectors in meters.
    bvec_m = np.zeros((2, nstack), dtype=np.float64)
    # Older data's input map has a simpler dtype
    try:
        input_map = index_map["input"]["chan_id"][:]
    except IndexError:
        input_map = index_map["input"][:]

    # Compute all baseline vectors.
    for vi in range(nstack):
        # Product index
        pi = index_map["stack"][vi][0]
        # Inputs that go into this product
        ipt0 = input_map[index_map["prod"][pi][0]]
        ipt1 = input_map[index_map["prod"][pi][1]]

        # Beseline vector in meters
        unique_index = telescope.feedmap[ipt0, ipt1]
        bvec_m[:, vi] = telescope.baselines[unique_index]
        # No need to conjugate. Already done in telescope.baselines.
        # if telescope.feedconj[ipt0, ipt1]:
        #    bvec_m[:, vi] *= -1.

    return bvec_m


def window_generalised(x, window="nuttall"):
    """A generalised high-order window at arbitrary locations.

    Parameters
    ----------
    x : np.ndarray[n]
        Location to evaluate at. Values outside the range 0 to 1 are zero.
    window : one of {'nuttall', 'blackman_nuttall', 'blackman_harris'}
        Type of window function to return.

    Returns
    -------
    w : np.ndarray[n]
        Window function.
    """
    a_table = {
        "uniform": np.array([1, 0, 0, 0]),
        "hann": np.array([0.5, -0.5, 0, 0]),
        "hanning": np.array([0.5, -0.5, 0, 0]),
        "hamming": np.array([0.53836, -0.46164, 0, 0]),
        "blackman": np.array([0.42, -0.5, 0.08, 0]),
        "nuttall": np.array([0.355768, -0.487396, 0.144232, -0.012604]),
        "blackman_nuttall": np.array([0.3635819, -0.4891775, 0.1365995, -0.0106411]),
        "blackman_harris": np.array([0.35875, -0.48829, 0.14128, -0.01168]),
    }

    a = a_table[window]

    t = 2 * np.pi * np.arange(4)[:, np.newaxis] * x[np.newaxis, :]

    w = (a[:, np.newaxis] * np.cos(t)).sum(axis=0)
    return np.where((x >= 0) & (x <= 1), w, 0)

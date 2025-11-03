"""Routines for Gaussian Process Regression."""

import numpy as np
import scipy.linalg as la
from caput.tools import invert_no_zero

from . import _fast_tools, kernels
from .dpss import _dtype_to_real


def resample(
    data: np.ndarray[np.floating | np.complexfloating],
    weight: np.ndarray[np.floating],
    xi: np.ndarray[np.number],
    xo: np.ndarray[np.number],
    cutoff_dist: float = 1.0,
    cutoff_partition: int = 0,
    kernel_spec: list | tuple | dict = {},
) -> tuple[np.ndarray[np.floating | np.complexfloating], np.ndarray[np.floating]]:
    """Resample a dataset using a GP kernel.

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
    cutoff_partition
        Propagate flag cutoff to `n` closest sample. Goes into `np.partition`.
        Value of 0 is equivalent to the closest sample. Default is 0.
    kernel_spec
        Kernel type and structure parameters. Can be provided as a single
        dict for a single kernel, or a list | tuple of dicts to combine
        multiple kernels.

    Returns
    -------
    xout
        Resampled data array.
    wout
        Resampled weight array.
    """
    # Handle any number of kernel specs as a list
    if not isinstance(kernel_spec, list | tuple):
        kernel_spec = [kernel_spec]

    # Get the kernels based on the kernel parameters
    Ki, Ks = _combine_gp_kernels_from_specs((xo, xi), kernel_spec)

    # Select the samples to interpolate to. The kernel width is required,
    # so extract the widest kernel from the spec(s).
    kwidth = 0.0
    for spec in kernel_spec:
        if (kw := spec.get("width", 0.0)) > kwidth:
            kwidth = kw

    inp_mask = ~np.all(weight == 0, axis=-1)
    xm = _select_interp_samples(xi, xo, inp_mask, kwidth, cutoff_dist, cutoff_partition)

    return interpolate_unweighted(data, weight, Ki, Ks, interp_samples=xm)


def interpolate_unweighted(
    data: np.ndarray[np.floating | np.complexfloating],
    weight: np.ndarray[np.floating],
    K: np.ndarray[np.floating],
    Kstar: np.ndarray[np.floating],
    interp_samples: np.ndarray[bool] | None = None,
) -> tuple[np.ndarray[np.floating | np.complexfloating], np.ndarray[np.floating]]:
    """Interpolate data using a GP kernel, assuming the signal is noise-free.

    Iterate the first axis and interpolate the second.

    Parameters
    ----------
    data
        Signal array. Iterate over the first axis and interpolate over the second.
        The third axis is a flattened combination of any other data axis over
        which the data masking is constant.
    weight
        Inverse-variance weight array, with the same first two axes as `data`.
    K
        Source samples kernel.
    Kstar
        Projection kernel.
    interp_samples
        Boolean array corresponding to a subset of the target samples
        onto which the data is projected. This is equivalent
        to masking certain samples after interpolating, but provides a
        performance improvement because there are fewer unnecessary
        operations. If None, interpolate onto all samples.
        Default is None.

    Returns
    -------
    xout
        Interpolated signal.
    woud
        Interpolated inverse-variance weights.

    Raises
    ------
    scipy.linalg.LinAlgError
        Raised if kernel `K` is not positive-definite. GP kernels are only
        required to be positive semi-definite, but this function requires a
        positive-definite kernel. This can usually be resolved by adding a
        small `epsilon` value to the diagonal of `K`, which is specified by
        the `epsilon` parameter of the `kernel_spec` argument in
        `_build_gp_kernels_from_spec`.
    """
    # Choose a solve method. The banded solver has faster parallel
    # performance, and is significantly faster when the kernel
    # is relatively narrow. I've just hard-coded this flag for now.
    _banded = True

    def solve(ab, b):
        if _banded:
            return la.solveh_banded(ab, b, lower=True, check_finite=False)

        return la.cho_solve(ab, b, check_finite=False)

    def decomp(ab):
        if _banded:
            return kernels.convert_band_diagonal(ab, which="lower")

        return la.cho_factor(ab, lower=True, check_finite=False)

    # Make the output masking array
    if interp_samples is None:
        interp_samples = [slice(None)] * data.shape[0]

    # If the signal is complex, interpolate the real and imaginary
    # components independently, assuming the same error
    data_dtype = data.dtype
    interp_dtype = _dtype_to_real(data_dtype)

    # Make the output arrays
    nsamp = Kstar.shape[0]
    xout = np.zeros((data.shape[0], nsamp, data.shape[-1]), dtype=data.dtype)
    wout = np.zeros((weight.shape[0], nsamp, weight.shape[-1]), dtype=weight.dtype)

    # Iterate the first axis and interpolate the second
    for ii in range(data.shape[0]):
        # Get the target sample mask and skip if all
        # samples are ignored
        mt = interp_samples[ii]

        if not isinstance(mt, slice) and not np.any(mt):
            continue

        # Get source and target sample masks
        wi = weight[ii]
        mi = np.any(wi > 0, axis=-1)

        # Decompose the kernel into a form which is faster to invert
        kd = decomp(K[mi][:, mi])
        # Solve for A = K_{s} K_{d}^{-1}, taking advantage
        # of symmetry in K_{d}
        A = solve(kd, Kstar[mt][:, mi].T).T.astype(np.float64, copy=False)
        # Get kernel main band bounds
        start, end = kernels._get_band_inds(A, tol=1.0e-8)

        # Iterate the last axis, tranpose for faster iteration
        # The fast cython utils only support 1D arrays at the moment
        # Invert noise weights to get variance
        vi = invert_no_zero(wi[mi].T)
        xi = np.ascontiguousarray(data[ii][mi].T[..., np.newaxis])

        # Allocate a temporary array to handle real or complex data
        # The banded matmul only supports 1D double-type arrays
        tmp = np.zeros(xout[0, mt].view(interp_dtype).shape, dtype=interp_dtype)

        # Propagate noise covariance by solving N_{f} = A N_{i} A^{H}
        for jj in range(vi.shape[0]):
            vij = vi[jj].astype(np.float64)

            if not np.any(vij > 0):
                continue

            # We could get the true inverse covariance diagonal:
            # wout[ii, mt, jj] = np.diag(la.solveh_banded(ncov, eye, lower=False))
            # with `eye` defined as a N x N identity. This is an order of
            # magnitude slower than just inverting the diagonal. Maybe there's
            # a faster way to get the inverse.
            ncov = _fast_tools._linear_covariance_banded(A, vij, start, end, bw=0)
            # This gets inverted later on, in a single operation
            # outside of the loop
            wout[ii, mt, jj] = ncov[-1]

            # Multiply the data. Using this banded matmul is
            # considerably faster than using numpy matmul.
            # It's worth checking that this is still true every
            # now and then. The equivalent operation would just be
            # xout[ii, mt] = (A @ xi).view(data_dtype)
            # outside of this second loop
            xij = xi[jj].view(interp_dtype).T
            # Multiply the real and imaginary components individually,
            # if the data is complex
            for kk in range(xij.shape[0]):
                # There's a hidden type conversion here, from `float64`
                # to `interp_dtype`
                tmp[:, kk] = _fast_tools._matmul_banded(
                    A, xij[kk].astype(np.float64), start, end
                )
            # Write to the output data array with the original data
            # dtype (depending on whether it's real or complex)
            xout[ii, mt, jj] = tmp.view(data_dtype)[:, 0]

    # Invert the variance to get weights. Faster to do one
    # operation here instead of inverting in the for loop
    invert_no_zero(wout, out=wout)

    # Weights shouldn't be negative - this is probably
    # numerical error in a small number of samples
    xout[wout < 0] = 0.0
    wout[wout < 0] = 0.0

    return xout, wout


def _select_interp_samples(
    xi: np.ndarray[np.number],
    xo: np.ndarray[np.number],
    mask: np.ndarray[bool],
    kwidth: int,
    cutoff: float,
    partition: int = 0,
) -> np.ndarray[bool]:
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
    kwidth
        Width of the interpolating kernel. Select only samples
    cutoff
        Select only output samples closer (in number of _input_ samples)
        than this value to the `n`th nearest unflagged input sample.
    partition
        `n`th closest sample to consider when applying cutoff. This is fed into
        `np.partition`. A value of `0` is relative to the nearest sample. Default
        is 0.

    Returns
    -------
    out
        Mask with flagged samples set to `False`
    """
    dist = np.subtract.outer(xo, xi)
    # Divide by the sample width to get the distance in number of input samples
    dist /= np.median(np.abs(np.diff(xi)))

    out = np.empty((mask.shape[0], xo.shape[0]), dtype=bool)

    kw_cutoff = kwidth - 1

    # Iterate over the first axis of `mask`
    for ii in range(mask.shape[0]):
        mi = mask[ii]

        if not np.any(mi):
            out[ii] = False
            continue

        dmi = dist[:, mi]

        pdist = np.min(dmi, axis=-1, where=dmi > 0, initial=kw_cutoff)
        ndist = np.max(dmi, axis=-1, where=dmi < 0, initial=-kw_cutoff)

        out[ii] = np.maximum(pdist, abs(ndist)) < kw_cutoff
        out[ii] &= np.partition(abs(dmi), partition, axis=-1)[:, partition] < cutoff

    return out


def _combine_gp_kernels_from_specs(
    samples: tuple, kernel_params: list[dict] | tuple[dict] | dict
) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]] | tuple[None, None]:
    """Helper to combine multiple kernels from different specs."""
    if not isinstance(kernel_params, list | tuple):
        kernel_params = [kernel_params]

    Ki = None
    Ks = None
    epsilon = None

    for kspec in kernel_params:
        # Remove the epsilon argument and accumulate on
        # the combined kernel
        var = kspec.pop("epsilon", 0.0)
        # Build each individual kernel
        ki, ks = _build_gp_kernels_from_spec(samples, kspec)

        if Ki is None:
            Ki = ki
            Ks = ks
            epsilon = np.zeros(Ki.shape[0], dtype=Ki.dtype)
        else:
            Ki *= ki
            Ks *= ks

        # Accumulate epsilon
        epsilon[:] += var

    np.einsum("ii->i", Ki)[:] += epsilon

    return Ki, Ks


def _build_gp_kernels_from_spec(
    samples: tuple, kernel_spec: dict
) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]:
    """Build a single kernel from a spec dictionary."""
    # Do not modify the input spec dict
    kernel_spec = kernel_spec.copy()
    # Format the input samples
    xi = samples[0]

    if isinstance(xi, np.ndarray):
        dx = np.median(np.abs(np.diff(xi)))
    elif isinstance(xi, int):
        dx = xi
    else:
        raise TypeError(
            "Invalid type for `samples`. "
            f"Expected `int` or `np.ndarray, got {type(xi)}."
        )

    # Scale the width by sample spacing
    width = kernel_spec.pop("width", 1.0) * dx
    name = kernel_spec.pop("name", "gaussian")
    epsilon = kernel_spec.pop("epsilon", 0.0)

    Ki = kernels.get_kernel(name=name, N=samples[1], width=width, **kernel_spec)
    np.einsum("ii->i", Ki)[:] += epsilon

    Ks = kernels.get_kernel(name=name, N=samples, width=width, **kernel_spec)

    return Ki.astype(np.float64, copy=False), Ks.astype(np.float64, copy=False)

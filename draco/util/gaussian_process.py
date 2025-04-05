"""Routines for Gaussian Process Regression."""

import numpy as np
import scipy.linalg as la
from caput.tools import invert_no_zero
from scipy.spatial.distance import cdist

from . import _fast_tools, kernels
from .dpss import _dtype_to_real


def resample(
    data: np.ndarray,
    weight: np.ndarray,
    xi: np.ndarray,
    xo: np.ndarray,
    cutoff_dist: float = 1.0,
    cutoff_partition: int = 0,
    kernel_spec: list | tuple | dict = {},
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
    # Get the kernels based on the kernel parameters
    Ki, Ks = _combine_kernels_from_specs((xo, xi), kernel_spec)

    # Flag samples depending on how far we want to interpolate. A target
    # sample must be both `kwidth - 1` samples from an unflagged input
    # sample on either side, and `cutoff_dist` samples from the `nth`
    # nearest unflagged input sample, where `n` is `cutoff_partition`
    # Pull `kwidth` from the kernel spec
    ks = [kernel_spec] if isinstance(kernel_spec, dict) else kernel_spec
    kwidth = 0.0
    for spec in ks:
        if (kw := spec.get("width", 0.0)) > kwidth:
            kwidth = kw

    inp_mask = ~np.all(weight == 0, axis=-1)
    tsamp = flag_abs_dist(xi, xo, inp_mask, cutoff_dist, cutoff_partition)
    tsamp &= flag_sample_dist(xi, xo, inp_mask, int(kwidth - 1))

    xout, wout = interpolate_unweighted(data, weight, Ki, Ks, tsamp=tsamp)

    return xout, wout


def interpolate_unweighted(
    data: np.ndarray,
    weight: np.ndarray,
    K: np.ndarray,
    Kstar: np.ndarray,
    tsamp: np.ndarray | None = None,
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
    tsamp
        Boolean array corresponding to a subset of the target samples
        onto which the data is projected. This is equivalent
        to masking certain samples after interpolating, but provides a
        performance improvement because there are fewer unnecessary
        operations. If None, interpolate onto all projection samples.
        Default is None.

    Returns
    -------
    xout
        Interpolated data array.
    woud
        Interpolated weight array.
    """
    # Figure out which banded solver to use
    _solveh = kernels.is_hermitian_positive_definite(K)

    which_band = "lower" if _solveh else "full"

    def solver(ab, b):
        if _solveh:
            return la.solveh_banded(ab, b, lower=True, check_finite=False)

        ndiag = ab.shape[0] // 2

        return la.solve_banded((ndiag, ndiag), ab, b, check_finite=False)

    # Make the output masking array
    if tsamp is None:
        tsamp = [slice(None)] * data.shape[0]

    # Use real views for `solve`, even if data is complex
    data_dtype = data.dtype
    interp_dtype = _dtype_to_real(data_dtype)

    # Make the output arrays
    nsamp = Kstar.shape[0]
    xout = np.zeros((data.shape[0], nsamp, *data.shape[2:]), dtype=data.dtype)
    wout = np.zeros((weight.shape[0], nsamp, *weight.shape[2:]), dtype=weight.dtype)

    # Iterate the first axis and interpolate the second
    for ii in range(data.shape[0]):
        # Get the target sample mask and skip if all
        # samples are ignored
        mt = tsamp[ii]

        if not isinstance(mt, slice) and not np.any(mt):
            continue

        # Get source and target sample masks
        wi = weight[ii]
        mi = np.any(wi > 0, axis=-1)

        # Extract only the parts of the data that we care
        # about and convert to a real view if complex
        xi = np.ascontiguousarray(data[ii][mi]).view(interp_dtype)

        # Convert the kernel to lower band-diagonal form,
        # which is significantly faster to invert
        kb = kernels.convert_band_diagonal(K[mi][:, mi], which=which_band)
        ks = Kstar[mt][:, mi]

        # Solve y = K_{s} K_{b}^{-1} x for the interpolated data
        alpha = solver(kb, xi)
        xout[ii, mt] = (ks @ alpha).view(data_dtype)

        # Interpolate variances by solving
        # C_{f} = A C_{i} A^{H}
        # where A^{H} = A^{T} == K_{b}^{-1} K_{s}
        # taking advantage of symmetry in K_{b} and the
        # bandedness of the kernel function
        A = solver(kb, ks.T).T.astype(np.float64)
        start, end = kernels._get_band_inds(A, tol=1.0e-4)

        # Estimate bandwidth and ensure it's odd
        bw = np.median(end - start).astype(np.int16)
        bw = bw - (bw % 2) + 1

        # Tranpose here for faster iteration
        vi = invert_no_zero(wi[mi].T)
        # Have to iterate over the last axis here, otherwise we end
        # up with a N x N x K covariance, where N is sum(mt) and K is
        # the number of samples in the last axis
        for jj in range(vi.shape[0]):
            vij = vi[jj]

            if not np.any(vij > 0):
                continue

            Ci = _fast_tools._linear_covariance_banded(
                A, vij.astype(np.float64), start, end, bw
            )
            wout[ii, mt, jj] = Ci[-1]
            # We could get the true inverse covariance:
            # wout[ii, mt, jj] = np.diag(la.solveh_banded(Ci, eye, lower=False))
            # given `eye` is defined as a N x N identity. This is an order of
            # magnitude slower than just inverting the diagonal, so at the moment
            # it doesn't seem reasonable. We could probably come up with some faster
            # cython code to take advantage of only needing the diagonal or something

    # Invert the variance to get weights. Faster to do one
    # operation here than to invert in the loops
    invert_no_zero(wout, out=wout)

    # Weights shouldn't be negative - this is probably
    # numerical error in a small number of samples
    wout[wout < 0] = 0.0
    xout[wout < 0] = 0.0

    return xout, wout


def flag_abs_dist(
    xi: np.ndarray, xo: np.ndarray, mask: np.ndarray, cutoff: float, partition: int = 0
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
    partition
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

        out[ii] = np.partition(dist[:, mi], partition, axis=-1)[:, partition] < cutoff

    return out


def flag_sample_dist(
    xi: np.ndarray, xo: np.ndarray, mask: np.ndarray, cutoff: float
) -> np.ndarray:
    """Mask output samples which are more than some distance from input samples.

    Unlike `flag_abs_dist`, this requires an output sample to be within a
    certain distance of unflagged input samples both before _and_ after the
    output sample. The result is a slightly more aggressive mask which does
    a better job at suppressing interpolation errors at the edge of wide
    masked bands.

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

    Returns
    -------
    out
        Mask with flagged samples set to `False`
    """
    dist = np.subtract.outer(xo, xi)
    # Divide by the sample width to get the distance in number of input samples
    dist /= np.median(np.abs(np.diff(xi)))

    out = np.empty((mask.shape[0], xo.shape[0]), dtype=bool)

    # Iterate over the first axis of `mask`
    for ii in range(mask.shape[0]):
        mi = mask[ii]

        if not np.any(mi):
            out[ii] = False
            continue

        dmi = dist[:, mi]

        pdist = np.min(dmi, axis=-1, where=dmi > 0, initial=cutoff)
        ndist = np.max(dmi, axis=-1, where=dmi < 0, initial=-cutoff)

        out[ii] = np.maximum(pdist, np.abs(ndist)) < cutoff

    return out


def _combine_kernels_from_specs(
    samples: tuple, kernel_params: list[dict] | tuple[dict] | dict
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Helper to combine multiple kernels from different specs."""
    kernel_params = (
        [kernel_params]
        if not isinstance(kernel_params, list | tuple)
        else kernel_params
    )

    Ki = None
    Ks = None
    svar = None

    for kspec in kernel_params:
        # Remove the svar argument and accumulate on
        # the combined kernel
        var = kspec.pop("svar", 0.0)
        # Build each individual kernel
        ki, ks = _build_kernel_from_spec(samples, kspec)

        if Ki is None:
            Ki = ki
            Ks = ks
            svar = np.zeros(Ki.shape[0], dtype=Ki.dtype)
        else:
            Ki *= ki
            Ks *= ks

        # Accumulate svar
        svar[:] += var

    np.einsum("ii->i", Ki)[:] += svar

    return Ki, Ks


def _build_kernel_from_spec(samples: tuple, kernel_spec) -> np.ndarray:
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
    svar = kernel_spec.pop("svar", 0.0)

    # Forward kernel
    Ki = kernels.get_kernel(name=name, N=samples[1], width=width, **kernel_spec)
    np.einsum("ii->i", Ki)[:] += svar
    # Projection kernel
    Ks = kernels.get_kernel(name=name, N=samples, width=width, **kernel_spec)

    return Ki.astype(np.float32), Ks.astype(np.float32)

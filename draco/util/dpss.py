"""Routines for DPSS inpainting."""

import numpy as np
from caput.algorithms import invert_no_zero
from scipy import interpolate
from scipy import linalg as la


def make_covariance(
    samples: np.ndarray,
    halfwidths: list[float],
    centres: list[float],
) -> np.ndarray:
    """Make a signal covariance model.

    Assumes the signal is a sum of top-hats in fourier
    space with centres at `centres` and half-widths
    in `halfwidths`.

    Parameters
    ----------
    samples
        Samples corresponding to the signal measurements
    halfwidths
        List of window half-widths in units defined by the
        fourier inverse of `samples`. Must be the same length
        as `centres`.
    centres
        List of window centres in units defined by the fourier
        inverse of `samples`. Must be the sample length as
        `halfwidths`.

    Returns
    -------
    cov
        Model for the signal covariance. If all entries in
        `centres` are zero, this is a real array. Otherwise,
        it is complex.
    """
    if np.isscalar(halfwidths):
        halfwidths = [halfwidths]

    if np.isscalar(centres):
        centres = [centres]

    if len(centres) != len(halfwidths):
        raise ValueError(
            "`halfwidths` and `centres` must be the same length. "
            f"Got halfwidths={halfwidths}, centres={centres}"
        )

    # Create a grid of the outer difference of samples
    ds = np.subtract.outer(samples, samples)
    cov = np.zeros(ds.shape, dtype=np.complex128)

    for ct, hw in zip(centres, halfwidths):
        cov += np.exp(-2.0j * np.pi * ct * ds) * np.sinc(2.0 * hw * ds)

    # If the covariance is entirely real, return a real-type
    # array since this is always computationally faster to use
    if np.isreal(cov).all():
        cov = np.ascontiguousarray(cov.real)

    return cov


def get_basis(
    cov: np.ndarray, threshold: float = 1e-12, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Compute the nth order Slepian sequence (DPSS).

    Order `n` is selected as the number of eigenvalues of
    `cov` greater than threshold when normalized by the
    largest eigenvalue.

    Parameters
    ----------
    cov
        Signal covariance model, constructed as a sum of
        top-hats.
    threshold
        Eigenvalue cutoff relative to the largest eigenvalue.
        Default is 1e-12.
    dtype
        Data type of the output sequence. This casting happens
        AFTER the eigen decomposition and eigenvalue
        threshold cut. If `cov` is complex, a datatype will
        correspond to the datatype of the REAL component - i.e. if
        `cov` is complex and `dtype=np.float32`, output sequence
        will have type `np.complex64`.

    Returns
    -------
    basis
        Eigenvectors corresponding to eigenvalues larger than
        `threshold` times the largest eigenvalue.
    """
    # Using the `evd` driver seems to be significantly faster
    # than other drivers for the types of matrices we're
    # using here. If this ever seems to become a rate-limiting
    # step, try using the `evr` driver instead.
    evals, evecs = la.eigh(cov, check_finite=False, driver="evd")
    idx = np.argsort(evals)[::-1]

    # Sort the values and vectors by decreasing
    # eigenvalue magnitude
    evals = evals[idx]
    evecs = evecs[:, idx]

    nmodes = (evals > threshold * evals.max()).sum()

    # Figure out the output datatype
    if np.iscomplexobj(evecs):
        dtype = _dtype_to_complex(dtype)
    else:
        dtype = _dtype_to_real(dtype)

    return evecs[:, :nmodes].astype(dtype)


def project(x: np.ndarray, Ni: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project noise-weighted data into the DPSS basis.

    This can be thought of as the right half of a wiener filter.


    Parameters
    ----------
    x
        Data to be interpolated.
    Ni
        Inverse noise variance associated with each sample,
        with masked values set to zero.
    A
        dpss basis, assumed to be the output of `get_basis`.

    Returns
    -------
    xproj
        Noise-weighted data projected into the dpss basis.
    """
    # Make sure these arrays are at least 2d,
    # which is helpful to make these operations consistant
    x, _ = atleast_Nd(x, 2)
    Ni, _ = atleast_Nd(Ni, 2)

    # Assuming `A` is given directly as the output of
    # `get_basis`, the conjugate transpose is needed
    AT = A.T.conj()

    return AT @ (Ni * x)


def solve(
    xp: np.ndarray, Ni: np.ndarray, A: np.ndarray, Si: float = 1e-3
) -> list[np.ndarray, np.ndarray]:
    """Apply the inpainting operator to data.

    Returns the inpainted data and corresponding inverse
    variance weight estimate. Only the diagonal of the
    covariance is returned, for computational reasons.

    Parameters
    ----------
    xp
        Noise-weighted data projected into the dpss basis,
        assumed to be the output of `project`
    Ni
        Inverse-variance noise weights.
    A
        dpss basis, assumed to be the output of `get_basis`.
    Si
        Regularizer, treated as the expected typical inverse
        signal variance in a Wiener filter. The default value
        of 1e-3 seems to work quite well, so it should be
        changed with caution.

    Returns
    -------
    xinp
        Inpainted data.
    winp
        Inverse of the diagonal of the uncertainty matrix.
    """
    AT = A.T.conj()
    # Check the shape of `xp` and `Ni` and move axes
    # if necessary to match `A`. Keep the original axis
    # position so it can be reverted later. It's faster to
    # copy than it is to iterate over a non-contiguous array
    xp, si = _check_shape(xp, AT, copy=True)
    Ni, _ = _check_shape(Ni, A, copy=True)
    # Ensure that `xp` and `Ni` are at least two
    # dimensional so we can iterate over the first axis
    xp, inv = atleast_Nd(xp, N=2, lax=0)
    Ni, _ = atleast_Nd(Ni, N=2, lax=0)

    b = np.zeros_like(xp)

    # Figure out which datatype to use depending
    # on whether or not `A` is complex
    cho_dtype = np.float32
    if np.iscomplexobj(A):
        cho_dtype = _dtype_to_complex(cho_dtype)

    # Iterate over the first axis
    for ii in range(xp.shape[0]):
        Ni_ii = Ni[ii].astype(A.dtype)

        if np.all(Ni_ii == 0):
            continue

        ATNi = AT * Ni_ii[np.newaxis]

        # Make the covariance matrix
        Ci = ATNi @ A
        # Add a diagonal regulariser to the covariance.
        # In a weiner filter, this is the signal covariance.
        # Use einsum trick to get a view of the diagonal.
        np.einsum("ii->i", Ci)[:] += Si
        # Cholesky decomposition. This is faster than
        # doing a standard solve, and significantly
        # faster than an inverse
        CiL = la.cho_factor(Ci.astype(cho_dtype), lower=False, check_finite=False)

        # Solve for the data projection coefficient
        b[ii] = la.cho_solve(CiL, xp[ii], check_finite=False)

        # Solve for beta part of the the inpainting operator
        # F = A (Si^{-1} + A^{H} Ni A)^{-1} A^{H} Ni
        # where F = A @ beta
        # Only save the diagonal component of the resulting
        # covariance. This can be done faster using a fancy
        # einsum than by doing the individual matrix mults,
        # but it's still the slowest step here so it might
        # be worth writing some custom cython or numba
        beta = la.cho_solve(CiL, ATNi, check_finite=False)

        betaT = beta.T.conj()
        N_ii = invert_no_zero(Ni_ii)
        var = np.einsum("ik,kj,j,jm,mi->i", A, beta, N_ii, betaT, AT, optimize="greedy")

        # Save out the inverse variance weights. Technically,
        # we should be making the full covariance matrix,
        # inverting that, then taking the resuling diagonal,
        # but that isn't computationally feasible
        Ni[ii] = invert_no_zero(var)

    # Construct the interpolated data
    x = A @ np.moveaxis(b[inv], -1, si)

    return x, np.moveaxis(Ni[inv], -1, si)


def accumulate_variance(wo: np.ndarray, wi: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pchip interpolate and accumulate weights.

    Pchip seems to behave reasonably well around boundaries
    and large spikes.

    Parameters
    ----------
    wo
        Non-inpainted inverse variance weights. These
        will be interpolated
    wi
        Inpainted weights, assumed to be output from `solve`
    W
        Boolean masking matrix, where False values correspond
        to flagged data in `wo` which need to be interpolated.

    Returns
    -------
    wacc
        Inverse variance weights accumulated after interpolating.
    """
    # Interpolate variances. This seems to produce
    # better edge behaviour
    vo = invert_no_zero(wo)
    vi = invert_no_zero(wi)

    samples = np.arange(vo.shape[0])

    # Iterate over the last axis
    for ii in range(vo.shape[-1]):
        sel = W[:, ii]

        if sel.sum() < 2:
            # Need a minimum number of samples
            # to interpolate
            continue

        # Interpolate. Pchip seems to work reasonably well
        # over small gaps without producing any crazy effects
        # over large gaps and edges.
        pchip = interpolate.PchipInterpolator(
            samples[sel], vo[:, ii][sel], extrapolate=True
        )
        # Handle extrapolation errors.
        wint = pchip(samples)
        wint[wint < 0] = 0
        # Accumulate
        vi[:, ii] += wint

    return invert_no_zero(vi)


def flag_above_cutoff(W: np.ndarray, fc: float | None = None) -> np.ndarray:
    """Mask to flag inpainted gaps wider than the cutoff `fc`.

    Calculates the width of each flagged region along the last axis,
    and returns a mask which is False for gaps larger than those
    that can be reasonably interpolated by a covariance with a
    certain cutoff.

    Parameters
    ----------
    W
        Original mask array, False where samples are flagged.
    fc
        Cutoff width, in units of samples. Gaps above this
        value are flagged. A `None` value will return the
        original mask

    Returns
    -------
    mask
        Mask with gaps larger than `fc` flagged, where
        False corresponds to flagged values.
    """
    if fc is None:
        return W

    M = ~W
    dist = np.zeros_like(W, dtype=np.float32)
    # Get the rising and falling edge of flagged regions
    rise = np.diff(M, axis=0, prepend=False) & M
    rise = rise[:-1]
    fall = np.diff(W, axis=0, append=False) & M

    # Get the leftmost and rightmost data boundaries,
    # assuming that we can't extrapolate
    lbound = np.argmax(W, axis=0)
    rbound = W.shape[0] - np.argmax(W[::-1], axis=0) - 1

    # This works, but maybe isn't optimally fast
    for ii in range(M.shape[-1]):
        rind = np.flatnonzero(rise[:, ii])
        find = np.flatnonzero(fall[:, ii])

        for ri, fi in zip(rind, find):
            dist[ri : fi + 1, ii] = fi - ri

        dist[: lbound[ii], ii] = 2 * fc
        dist[rbound[ii] :, ii] = 2 * fc

    return dist < fc


def filter(
    x: np.ndarray, Ni: np.ndarray, A: np.ndarray, W: np.ndarray, Si: float = 1e-3
) -> np.ndarray:
    """Filter using a DPSS basis over the first axis.

    Parameters
    ----------
    x
        Data to filter. Can be Real or Complex.
    Ni
        Inverse variance noise weights for each sample in `x`.
    A
        dpss basis, assumed to be the output of `get_basis`.
    W
        Mask array, where False values will be replaced by
        inpainted data.
    Si
        Regularizer, treated as the expected typical inverse
        signal variance in a Wiener filter. The default value
        of 1e-3 seems to work quite well, so it should be
        changed with caution.

    Returns
    -------
    xfilt
        Filtered data.
    wfilt
        Filtered inverse variance weights.
    """
    # Subtract the mean data before inpainting and
    # re-add at the end
    xhat = np.sum(x * W, axis=0, keepdims=True)
    xhat *= invert_no_zero(np.sum(W, axis=0, keepdims=True))

    # Make the data projection
    xp = project(x - xhat, Ni, A)
    # Make the inpainted data
    xfilt, wfilt = solve(xp, Ni, A, Si)

    # Interpolate and accumulate variances
    wfilt = accumulate_variance(Ni, wfilt, W)

    # Re-add the mean
    xfilt += xhat

    return xfilt, wfilt


def inpaint(
    x: np.ndarray, Ni: np.ndarray, A: np.ndarray, W: np.ndarray, Si: float = 1e-3
) -> np.ndarray:
    """Inpaint using a DPSS basis over the first axis.

    Parameters
    ----------
    x
        Data to interpolate. Can be Real or Complex.
    Ni
        Inverse variance noise weights for each sample in `x`.
    A
        dpss basis, assumed to be the output of `get_basis`.
    W
        Mask array, where False values will be replaced by
        inpainted data.
    Si
        Regularizer, treated as the expected typical inverse
        signal variance in a Wiener filter. The default value
        of 1e-3 seems to work quite well, so it should be
        changed with caution.

    Returns
    -------
    xinp
        Inpainted data. Samples where `W` is True are not changed
        from the input data.
    winp
        Inpainted inverse variance weights. Samples where `W` is True
        are not changed from the input data.
    """
    xinp, winp = filter(x, Ni, A, W, Si)

    xinp[W] = x[W]
    winp[W] = Ni[W]

    return xinp, winp


def atleast_Nd(x: np.ndarray, N: int, lax: int = -1):
    """Ensure that an array is at least `N` dimensional.

    Unlike `np.atleast_2d` and similar, this allows the user
    to select the location where new axes are inserted. New axes
    are always grouped together. If `x.ndim` is greater or equal to
    `N`, `x` is returned unchanged.

    Parameters
    ----------
    x
        Array to expend dimensions.
    N
        Desired dimension of `x`.
    lax
        Axis to the left of where the new axes
        are added. Default is -1, so new axes are
        added to the end.

    Returns
    -------
    xn
        `x` with extra dimensions added, if needed.
    inv
        Index tuple which inverses the operation.
    """
    # Desired dimensionality is already satisfied
    if x.ndim >= N:
        return x, (slice(None),) * x.ndim

    # Create an indexer for the new axes
    newdims = (np.newaxis,) * (N - x.ndim)

    # Create an indexer to place the new axes
    # according to `lax`
    if lax == -1:
        lax = x.ndim

    slobj = (slice(None),) * max(x.ndim - lax, 0)

    add = (..., *newdims, *slobj)
    inv = (..., *(0 for _ in newdims), *slobj)

    return x[add], np.s_[inv]


def _check_shape(x: np.ndarray, A: np.ndarray, copy: bool = False):
    """Set the last axis of `x` to match the first axis of `A`."""
    sval = A.shape[0]

    try:
        si = x.shape.index(sval)
    except ValueError as exc:
        raise ValueError(f"Shape mismatch. x: {x.shape}, A: {A.shape}.") from exc

    # Move the target axis to the end
    if x.shape[-1] != sval:
        x = np.moveaxis(x, si, -1)

    # Make a copy. This is helpful for speed since
    # the transposed array may not be c-contiguous,
    # but should be used with caution if `x` is large
    if copy:
        x = x.copy(order="C")

    return x, si


def _dtype_to_real(dtype: np.dtype) -> np.dtype:
    """Get a real dtype with matching precision."""
    return np.dtype(dtype).type(0).real.dtype


def _dtype_to_complex(dtype: np.dtype) -> np.dtype:
    """Get a complex dtype with matching precision."""
    _map = {
        "float32": np.complex64,
        "float64": np.complex128,
    }

    return np.dtype(_map[_dtype_to_real(dtype).name])

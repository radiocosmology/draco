"""Routines for DPSS inpainting."""

import numpy as np
from scipy import linalg as la


def make_covariance(
    samples: np.ndarray, halfwidths: list[float], centres: list[float]
) -> np.ndarray:
    """Make a signal covariance model.

    Assumes the signal is a sum of top-hats in fourier
    space with centres at `centres` as half-widths
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
    if np.isreal(cov):
        cov = np.ascontiguousarray(cov.real)

    return cov


def get_sequence(cov: np.ndarray, threshold: float = 1e-12) -> np.ndarray:
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

    Returns
    -------
    sequence
        Eigenvectors corresponding to eigenvalues larger than
        `threshold` times the largest eigenvalue. This is the nth
        order Slepian sequence, or nth order DPSS.
    """
    evals, evecs = la.eigh(cov, check_finite=False, driver="evr")
    idx = np.argsort(evals)[::-1]

    # Sort the values and vectors by decreasing
    # eigenvalue magnitude
    evals = evals[idx]
    evecs = evecs[:, idx]

    nmodes = (evals > threshold * evals.max()).sum()

    return evecs[:, :nmodes]


def project(x: np.ndarray, Ni: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project noise-weighted data into the DPSS basis.

    This can be thought of as the right half of a wiener filter.
    """
    # Make sure these arrays are at least 2d,
    # which is helpful to make these operations consistant
    x = atleast_Nd(x, 2)
    Ni = atleast_Nd(Ni, 2)

    # Assuming `A` is given directly as the output of
    # py:function:`sequence`, the conjugate transpose
    # is needed here
    AT = A.T.conj()

    return AT @ (Ni * x)


def solve(
    xp: np.ndarray, Ni: np.ndarray, A: np.ndarray, Si: float = 1e-3
) -> list[np.ndarray, np.ndarray]:
    """Compute the DPSS coefficients."""
    # Check the shape of `xp` and `Ni` and move axes
    # if necessary to match `A`. Keep the original axis
    # position so it can be reverted later. It's faster to
    # copy than it is to iterate over a non-contiguous array
    xp, si = _check_shape(xp, A, copy=True)
    Ni, _ = _check_shape(Ni, A, copy=True)
    # Ensure that `xp` and `Ni` are at least two
    # dimensional so we can iterate over the first axis
    xp, inv = atleast_Nd(xp, N=2, lax=0)
    Ni, _ = atleast_Nd(Ni, N=2, lax=0)
    # Save out the approximate wiener filter noise weights
    nw = np.zeros_like(xp, dtype=np.float32)

    AT = A.T.conj()

    # Iterate over the last axis
    for ii in range(xp.shape[0]):
        Ni_ii = Ni[ii].astype(np.float64)
        # Make the covariance matrix and get the
        # Cholesky decomposition. This is faster than
        # doing a standard solve
        Ci = AT @ (Ni_ii[:, np.newaxis] * A)
        Cidiag = np.einsum("ii->i", Ci)
        # Save weights in case they're needed later
        nw[ii] = Cidiag
        # Add a diagonal regulariser to the covariance.
        # In a weiner filter, this is the signal covariance.
        # Use an einsum trick to get a view of the diagonal.
        Cidiag[:] += Si
        # Decompose and solve
        CiL = la.cho_factor(Ci, lower=False, check_finite=False)

        xp[ii] = la.cho_solve(CiL, xp[ii], check_finite=False)

    # Return arrays to their original shape
    xp = np.moveaxis(xp[inv], -1, si)
    nw = np.moveaxis(nw[inv], -1, si)

    return xp, nw


def inpaint(
    A: np.ndarray, b: np.ndarray, x: np.ndarray = None, W: np.ndarray = None
) -> np.ndarray:
    """Inpaint using a DPSS basis and a set of coefficients.

    If `x` and `W` are not None, assume that these are the
    observed data and a mask, and keep the observed data
    anywhere that `W` is True.
    """
    # Raise an error if only one of the data
    # and data indexing arrays are provided
    if (x is not None) ^ (W is not None):
        raise ValueError(
            f"Only one of `x` and `W` were given. x={x}, W={W}. "
            "Must provide either both or neither."
        )

    # Make the inpainted data
    xinp = A @ b
    # Keep the existing values. Typically this is what
    # would be desired when inpainting.
    if x is not None:
        # I think this can be done with direct indexing
        # since all arrays have the same shape, but it's
        # possible that calling `.ravel()` will be needed
        xinp[W] = x[W]

    return xinp


def atleast_Nd(x: np.ndarray, N: int, lax: int = -1):
    """Ensure that an array is at least `n` dimensional.

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
    inverse
        If True, return the slice that recovers the
        original array dimension.

    Returns
    -------
    xn
        `x` with extra dimensions added, if needed.
    """
    # Desired dimensionality is already satisfied
    if x.ndim >= N:
        return x

    # Create an indexer for the new axes
    newdims = (np.newaxis,) * (N - x.ndim)

    # Create an indexer to place the new axes
    # according to `lax`
    if lax == -1:
        lax = x.ndim

    slobj = (slice(None),) * max(x.ndim - lax, 0)

    inv = np.s_[..., *(0 for _ in newdims), *slobj]

    return x[..., *newdims, *slobj], inv


def _check_shape(x: np.ndarray, A: np.ndarray, copy: bool = True):
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

"""Routines for creating kernel/covariance matrices."""

import numpy as np
from scipy import linalg as la
from scipy.spatial.distance import cdist

__all__ = [
    "convert_band_diagonal",
    "euclidean_difference_kernel",
    "gaussian_kernel",
    "get_kernel",
    "is_hermitian_positive_definite",
    "matern_kernel",
    "moving_average_inverse_kernel",
    "periodic_kernel",
    "rational_kernel",
    "squared_difference_kernel",
]


def get_kernel(name: str, **kernel_params):
    """Get a covariance matrix by name.

    Parameters
    ----------
    name : str
        Name of the covariance function.
    kernel_params : dict
        Extra keyword arguments to pass to the kernel function.

    Notes
    -----
    The `banded` argument is no longer honoured, and a full kernel
    array is returned. The full kernel can be converted to
    band-diagonal form with `convert_band_diagonal`.
    """
    if "banded" in kernel_params:
        import warnings

        warnings.warn("The `banded` keyword is not longer used")

    kdict = {
        "gaussian": gaussian_kernel,
        "rational": rational_kernel,
        "matern": matern_kernel,
        "periodic": periodic_kernel,
        "moving_average": moving_average_inverse_kernel,
    }

    kernelfunc = kdict.get(name.lower())

    if kernelfunc is None:
        raise ValueError(
            f"Invalid kernel type: '{name}'. " f"Valid kernels: {list(kdict.keys())}"
        )

    return kernelfunc(**kernel_params)


# =======
# Kernels
# =======


def gaussian_kernel(
    N: int | tuple | np.ndarray, width: int | float | tuple, alpha: float, **kwargs
):
    """Return a gaussian kernel.

    Parameters
    ----------
    N : int | tuple
        Number of samples over which to generate the kernel.
        If this is a length-2 tuple, assume that this is a
        correlation between two different sets of indices.
    width : int | tuple
        Standard deviation of the kernel.
    alpha : float
        Square root of the kernel variance.
    kwargs : dict
        Unused keyword arguemnts, required for compatibilty
        when calling with `get_kernel`.

    Returns
    -------
    C : np.ndarray
        Gaussian covariance matrix of shape (N[0], N[1]). If
        N is an integer, the covariance is square with shape (N, N).
    """
    dist = squared_difference_kernel(N, width)

    return (alpha**2) * np.exp(-0.5 * dist)


def rational_kernel(
    N: int | tuple | np.ndarray,
    width: int | float | tuple,
    alpha: float,
    a: float,
    **kwargs,
) -> np.ndarray:
    """Return a rational kernel.

    Parameters
    ----------
    N : int | tuple
        Number of samples over which to generate the kernel.
        If this is a length-2 tuple, assume that this is a
        correlation between two different sets of indices.
    width : int | tuple
        Width of the kernel.
    alpha : float
        Square root of the kernel variance.
    a : float
        Kernel scale weighting parameter.
    kwargs : dict
        Unused keyword arguemnts, required for compatibilty
        when calling with `get_kernel`.

    Returns
    -------
    C : np.ndarray
        Rational covariance matrix of shape (N[0], N[1]). If
        N is an integer, the covariance is square with shape (N, N).
    """
    dist = squared_difference_kernel(N, width)

    return (alpha**2) * (1 + dist / (2 * a)) ** -a


def matern_kernel(
    N: int | tuple | np.ndarray,
    width: int | float | tuple,
    alpha: float,
    nu: float,
    **kwargs,
) -> np.ndarray:
    """Return a matern kernel.

    Parameters
    ----------
    N : int | tuple
        Number of samples over which to generate the kernel.
        If this is a length-2 tuple, assume that this is a
        correlation between two different sets of indices.
    width : int | tuple
        Width of the kernel.
    alpha : float
        Square root of the kernel variance.
    nu : float
        Smoothness parameter. Larger values produce smoother kernels.
        Currently, only values of 1.5 (once differentiable) and 2.5
        (twice differentiable) are supported.
    kwargs : dict
        Unused keyword arguemnts, required for compatibilty
        when calling with `get_kernel`.

    Returns
    -------
    C : np.ndarray
        Matern covariance matrix of shape (N[0], N[1]). If
        N is an integer, the covariance is square with shape (N, N).
    """
    if nu not in {1.5, 2.5}:
        raise ValueError(
            f"Invalid value `nu`={nu}. "
            "Only values of (1.5, 2.5) are currently supported."
        )

    dist = euclidean_difference_kernel(N, width)

    if nu == 1.5:
        dist *= np.sqrt(3)
        C = 1.0 + dist
        C *= np.exp(-dist)
    elif nu == 2.5:
        dist *= np.sqrt(5)
        C = 1.0 + dist + dist**2 / 3.0
        C *= np.exp(-dist)

    # Scale
    C *= alpha**2

    return C


def periodic_kernel(
    N: int | tuple | np.ndarray,
    width: int | float | tuple,
    alpha: float,
    p: float,
    **kwargs,
) -> np.ndarray:
    """Return a periodic kernel, aka Exp-Sine-Squared.

    Parameters
    ----------
    N
        Number of samples over which to generate the kernel.
        If this is a length-2 tuple, assume that this is a
        correlation between two different sets of indices.
    width
        Width of the kernel.
    alpha
        Square root of the kernel variance.
    p
        Periodicity of the kernel.
    kwargs : dict
        Unused keyword arguemnts, required for compatibilty
        when calling with `get_kernel`.

    Returns
    -------
    C : np.ndarray
        Periodic covariance matrix of shape (N[0], N[1]). If
        N is an integer, the covariance is square with shape (N, N).
    """
    dist = euclidean_difference_kernel(N, width)

    C = np.sin(np.pi * dist / p)
    C = np.exp(-2 * C**2)

    # Scale
    C *= alpha**2

    return C


def moving_average_inverse_kernel(
    N: int, width: int, alpha: float, periodic: bool = True
) -> np.ndarray:
    """A smoothness prior on the values at given locations.

    This calculates the average in a window of `width` points, and then applies a
    Gaussian with precision `alpha` and this average as the mean for each point. For a
    `width` of 3 this is effectively a constraint on the second derivative.

    Parameters
    ----------
    N : int | tuple
        Number of samples over which to generate the kernel.
        If this is a length-2 tuple, assume that this is a
        correlation between two different sets of indices.
    width : int | tuple
        Width of the kernel.
    alpha : float
        Smoothness precision.
    periodic : bool
        Assume the function is periodic and wrap.

    Returns
    -------
    Ci : np.ndarray
        Inverse covariance matrix of shape (N[0], N[1]) representing a
        window average. If N is an integer, the covariance is square
        with shape (N, N).
    """
    # Calculate the matrix for the moving average
    W = np.zeros((N, N))
    for i in range(N):

        ll, ul = i - (width - 1) // 2, i + (width + 1) // 2
        if not periodic:
            ll, ul = max(0, ll), min(ul, N)
        v = np.arange(ll, ul)

        W[i][v] = 1.0 / len(v)

    IW = np.identity(N) - W

    return alpha * (IW.T @ IW)


# ==================
# Distance functions
# ==================


def squared_difference_kernel(
    N: int | tuple | np.ndarray, width: int | float | tuple
) -> np.ndarray:
    """Create a distance matrix for a kernel using squared difference.

    N : int | tuple
        Number of samples over which to generate the kernel.
        If this is a length-2 tuple, assume that this is a
        correlation between two different sets of indices.
    width : int | tuple
        Width of the kernel along each axis. For a gaussian,
        this is the standard deviation.

    Returns
    -------
    diff : np.ndarray
        Array of normalized distances.
    """
    # If only a single integer is provided, assume
    # a square covariance matrix
    if isinstance(N, int | np.ndarray):
        N = (N, N)

    if isinstance(width, int | float):
        width = (width, width)

    if len(N) != 2 or len(width) != 2:
        raise ValueError(f"Invalid parameters. Got N={N} and width={width}.")

    i0 = np.arange(N[0]) if isinstance(N[0], int) else N[0]
    i1 = np.arange(N[1]) if isinstance(N[1], int) else N[1]

    i0 = i0 / width[0]
    i1 = i1 / width[1]

    return np.subtract.outer(i0, i1) ** 2


def euclidean_difference_kernel(
    N: int | tuple | np.ndarray, width: int | float | tuple
) -> np.ndarray:
    """Create a distance matrix for a kernel using euclidean difference.

    N : int | tuple
        Number of samples over which to generate the kernel.
        If this is a length-2 tuple, assume that this is a
        correlation between two different sets of indices.
    width : int | tuple
        Width of the kernel along each axis. For a gaussian,
        this is the standard deviation.

    Returns
    -------
    diff : np.ndarray
        Array of normalized distances.
    """
    if isinstance(N, int | np.ndarray):
        N = (N, N)

    if isinstance(width, int | float):
        width = (width, width)

    if len(N) != 2 or len(width) != 2:
        raise ValueError(f"Invalid parameters. Got N={N} and width={width}.")

    i0 = np.arange(N[0]) if isinstance(N[0], int) else N[0]
    i1 = np.arange(N[1]) if isinstance(N[1], int) else N[1]

    i0 = i0 / width[0]
    i1 = i1 / width[1]

    return cdist(i0[:, np.newaxis], i1[:, np.newaxis], metric="euclidean")


# =========
# Utilities
# =========


def is_hermitian_positive_definite(x: np.ndarray) -> bool:
    """Check if a matrix is Hermitian positive-definite.

    Parameters
    ----------
    x
        Array to check.

    Returns
    -------
    result
        True if `x` is hermitian positive-definite
    """
    if not la.ishermitian(x):
        return False

    try:
        la.cholesky(x, lower=False)
    except la.LinAlgError:
        return False

    return True


def convert_band_diagonal(
    x: np.ndarray, tol: float = 1.0e-8, which: str = "full"
) -> np.ndarray:
    """Convert a full band diagonal kernel into just the lower band.

    Used to feed into `la.solveh_banded.`

    Parameters
    ----------
    x
        `n x n` symmetric band-diagonal matrix
    tol
        Smallest value to consider when finding the
        band edge. Default is 1.0e-5.
    which
        Which band to extract. Options are
        {"lower", "upper", "full"}. Default is "full".

    Returns
    -------
    xb
        Band diagonal matrix of shape (n,u+l+1), (n,u+1), or (n,l+1)
    """
    if which == "full":
        return _bd_sym(x, tol)
    if which in {"upper", "lower"}:
        return _bd_sym_ul(x, tol, lower=which == "lower")

    raise ValueError(
        f"Got invalid argument `which`={which}. "
        "Only `full`, `upper`, or `lower` are accepted."
    )


def _bd_sym(x: np.ndarray, tol: float) -> np.ndarray:
    """Full band of a symmetric band-diagonal matrix."""
    N = x.shape[0]
    M = np.sum(x > tol, axis=-1).max() // 2 + 1

    banded = np.zeros((2 * M - 1, N), dtype=x.dtype)

    banded[M - 1 :] = _bd_sym_ul(x, tol, lower=True)
    banded[: M - 1] = _bd_sym_ul(x, tol, lower=False)[1:]

    return banded


def _bd_sym_ul(x: np.ndarray, tol: float, lower: bool = False) -> np.ndarray:
    """Upper or lower band of a symmetric band-diagonal matrix.

    Should be symmetric positive-definite.
    """
    N = x.shape[0]
    M = np.sum(x > tol, axis=-1).max() // 2 + 1

    banded = np.zeros((M, N), dtype=x.dtype)

    for ii in range(M):
        if lower:
            banded[ii, : N - ii] = x.diagonal(ii)
        else:
            banded[-ii, ii:] = x.diagonal(-ii)

    return banded


def _get_band_inds(R: np.ndarray, tol: float = 1.0e-4) -> tuple:
    """Get the indices of the band edge for a band diagonal matrix.

    Parameters
    ----------
    R
        Band diagonal matrix
    tol
        Cutoff threshold to consider values to be
        outside of the main band.

    Returns
    -------
    start_ind : np.ndarray[int]
        left indices of the band.
    end_ind : np.ndarray[int]
        right indices of the band.
    """
    u = abs(R) > tol

    start_ind = np.argmax(u, axis=-1)
    end_ind = R.shape[-1] - np.argmax(u[..., ::-1], axis=-1)
    end_ind[~np.any(u, axis=-1)] = 0

    return start_ind.astype(np.int32), end_ind.astype(np.int32)

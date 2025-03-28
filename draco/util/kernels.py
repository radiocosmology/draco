"""Routines for creating kernel/covariance matrices."""

import numpy as np
from scipy import linalg as la
from scipy.spatial.distance import cdist


def get_kernel(name: str, **kernel_params):
    """Get a covariance matrix by name.

    Parameters
    ----------
    name : str
        Name of the covariance function.
    kernel_params : dict
        Extra keyword arguments to pass to the kernel function.
    """
    kdict = {
        "gaussian": gaussian_kernel,
        "rational": rational_kernel,
        "matern": matern_kernel,
        "lanczos": lanczos_kernel,
        "moving_average": moving_average_inverse_kernel,
    }

    kernelfunc = kdict.get(name.lower())

    if kernelfunc is None:
        raise ValueError(
            f"Invalid kernel type: '{name}'. " f"Valid kernels: {list(kdict.keys())}"
        )

    kernel = kernelfunc(**kernel_params)

    # Return a band diagonal kernel
    if kernel_params.get("banded", False):
        # Make sure this is a valid option
        if (kernel.shape[0] != kernel.shape[1]) or not np.allclose(kernel, kernel.T):
            raise ValueError(
                "Cannot convert non-symmetric matrix to band diagonal. "
                f"Got kernel with shape {kernel.shape}."
            )

        tol = kernel_params.get("tol")
        args = {"tol": tol} if tol is not None else {}
        kernel = get_band_diagonal(kernel, **args)

    return kernel


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


def lanczos_kernel(
    N: int | tuple | np.ndarray, width: int | float | tuple, alpha: float, **kwargs
) -> np.ndarray:
    """Return a lanczos kernel.

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
    kwargs
        Unused keyword arguemnts, required for compatibilty
        when calling with `get_kernel`.

    Returns
    -------
    C
        Lanczos covariance matrix of shape (N[0], N[1]). If
        N is an integer, the covariance is square with shape (N, N).
    """
    dist = euclidean_difference_kernel(N, width)

    # Figure out the width and sample spacing used
    # in order to create the lengthened sinc window
    xi = N[0] if isinstance(N, tuple) else N
    dxi = np.median(np.abs(np.diff(xi))) if isinstance(xi, np.ndarray) else 1.0

    a = width[0] if isinstance(width, tuple) else width
    a /= dxi

    C = np.where(
        abs(dist) < 1.0, np.sinc(dist * a / np.pi) * np.sinc(dist / np.pi), 0.0
    )

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


def get_band_diagonal(
    x: np.ndarray, tol: float = 1.0e-4, which: str = "full"
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
        `n x k` lower band of `x`.
    """
    N = x.shape[0]
    M = np.sum(x > tol, axis=-1).max() // 2 + 1

    if which == "full":
        idx = range(-M, M)
    elif which == "lower":
        idx = range(0, M)
    elif which == "upper":
        idx = range(-M + 1, 1)
    else:
        raise TypeError(
            f"Got invalid argument for `which`: {which}. "
            "Options are [`lower`, `upper`, `full`]."
        )

    banded = np.zeros((len(idx), N), dtype=x.dtype)

    for i in idx:
        if i >= 0:
            banded[i, : N - i] = x.diagonal(i)
        else:
            banded[i, -i:] = x.diagonal(i)

    # Correct for indexing
    if which == "full":
        banded = np.fft.fftshift(banded, axes=0)
    elif which == "upper":
        banded = np.roll(banded, shift=-1, axis=0)

    return banded


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

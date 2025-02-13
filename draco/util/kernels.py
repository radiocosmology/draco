"""Routines for creating kernel/covariance matrices."""

from warnings import warn

import numpy as np
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

        N = kernel_params["width"]
        M = kernel.shape[0]
        K = np.ones((N, M), dtype=kernel.dtype)

        for i in range(N):
            K[i, : M - i] = kernel.diagonal(i)

        kernel = K

    return kernel


def _warn_unused_kwargs(kwargs):
    """Warn if any unused arguments."""
    if len(kwargs):
        warn(f"Unused keyword arguments: {kwargs}.")


def squared_difference_kernel(N: int | tuple, width: int | tuple) -> np.ndarray:
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
    if isinstance(N, int):
        N = (N, N)

    if isinstance(width, int):
        width = (width, width)

    if len(N) != 2 or len(width) != 2:
        raise ValueError(f"Invalid parameters. Got N={N} and width={width}.")

    i0 = np.arange(N[0]) / width[0]
    i1 = np.arange(N[1]) / width[1]

    return np.subtract.outer(i0, i1) ** 2


def euclidean_difference_kernel(N: int | tuple, width: int | tuple) -> np.ndarray:
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
    if isinstance(N, int):
        N = (N, N)

    if isinstance(width, int):
        width = (width, width)

    if len(N) != 2 or len(width) != 2:
        raise ValueError(f"Invalid parameters. Got N={N} and width={width}.")

    # The extra axis is required to use `cdist`
    i0 = np.arange(N[0])[:, np.newaxis] / width[0]
    i1 = np.arange(N[1])[:, np.newaxis] / width[1]

    return cdist(i0, i1, metric="euclidean")


def gaussian_kernel(N: int | tuple, width: int | tuple, alpha: float, **kwargs):
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
    N: int | tuple, width: int | tuple, alpha: float, a: float, **kwargs
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
    N: int | float, width: int | float, alpha: float, nu: float, **kwargs
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
        C = np.sqrt(3) * dist
        C = (1.0 + C) * np.exp(-C)
    elif nu == 2.5:
        C = np.sqrt(5) * dist
        C = (1.0 + C + C**2 / 3.0) * np.exp(-C)

    return (alpha**2) * C


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

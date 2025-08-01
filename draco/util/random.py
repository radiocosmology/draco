"""Utilities for drawing random numbers."""

import numpy as np
from caput.random import default_rng


def complex_normal(loc=0.0, scale=1.0, size=None, dtype=None, rng=None, out=None):
    """Get a set of complex normal variables.

    By default generate standard complex normal variables.

    Parameters
    ----------
    size : tuple
        Shape of the array of variables.
    loc : np.ndarray or complex float, optional
        The mean of the complex output. Can be any array which broadcasts against
        an array of `size`.
    scale : np.ndarray or float, optional
        The standard deviation of the complex output. Can be any array which
        broadcasts against an array of `size`.
    dtype : {np.complex64, np.complex128}, optional
        Output datatype.
    rng : np.random.Generator, optional
        Generator object to use.
    out : np.ndarray[shape], optional
        Array to place output directly into.

    Returns
    -------
    out : np.ndarray[shape]
        Complex gaussian variates.
    """
    # Validate/set size argument
    if size is None and out is None:
        size = (1,)
    elif out is not None and size is None:
        size = out.shape
    elif out is not None and size is not None and out.shape != size:
        raise ValueError(
            f"Shape of output array ({out.shape}) != size argument ({size}"
        )

    # Validate/set dtype argument
    if dtype is None and out is None:
        dtype = np.complex128
    elif dtype is None and out is not None:
        dtype = out.dtype.type
    elif out is not None and dtype is not None and out.dtype.type != dtype:
        raise ValueError(
            f"Dtype of output array ({out.dtype.type}) != dtype argument ({dtype}"
        )

    if rng is None:
        rng = default_rng()

    _type_map = {
        np.complex64: np.float32,
        np.complex128: np.float64,
    }

    if dtype not in _type_map:
        raise ValueError(
            f"Only dtype must be complex64 or complex128. Got dtype={dtype}."
        )

    if out is None:
        out = np.ndarray(size, dtype=dtype)

    # Fill the complex array by creating a real type view of it
    rtype = _type_map[dtype]
    rsize = (*size[:-1], size[-1] * 2)
    rng.standard_normal(rsize, dtype=rtype, out=out.view(rtype))

    # Use inplace ops for scaling and adding to avoid intermediate arrays
    rscale = scale / 2**0.5
    out *= rscale

    # Don't bother with the additions if not needed
    if np.any(loc != 0.0):
        out += loc

    return out


def standard_complex_normal(shape, dtype=None, rng=None):
    """Get a set of standard complex normal variables.

    Parameters
    ----------
    shape : tuple
        Shape of the array of variables.
    dtype : {np.complex64, np.complex128}, optional
        Output datatype.
    rng : np.random.Generator, optional
        Generator object to use.

    Returns
    -------
    out : np.ndarray[shape]
        Complex gaussian variates.
    """
    return complex_normal(size=shape, dtype=dtype, rng=rng)


def standard_complex_wishart(m, n, rng=None):
    """Draw a standard Wishart matrix.

    Parameters
    ----------
    m : integer
        Number of variables (i.e. size of matrix).
    n : integer
        Number of measurements the covariance matrix is estimated from.
    rng : np.random.Generator, optional
        Random number generator to use.

    Returns
    -------
    B : np.ndarray[m, m]
    """
    if rng is None:
        rng = default_rng()

    # Fill in normal variables in the lower triangle
    T = np.zeros((m, m), dtype=np.complex128)
    T[np.tril_indices(m, k=-1)] = (
        rng.standard_normal(m * (m - 1) // 2)
        + 1.0j * rng.standard_normal(m * (m - 1) // 2)
    ) / 2**0.5

    # Gamma variables on the diagonal
    for i in range(m):
        T[i, i] = rng.gamma(n - i) ** 0.5

    # Return the square to get the Wishart matrix
    return np.dot(T, T.T.conj())


def complex_wishart(C, n, rng=None):
    """Draw a complex Wishart matrix.

    Parameters
    ----------
    C : np.ndarray[:, :]
        Expected covaraince matrix.
    n : integer
        Number of measurements the covariance matrix is estimated from.
    rng : np.random.Generator, optional
        Random number generator to use.

    Returns
    -------
    C_samp : np.ndarray
        Sample covariance matrix.
    """
    import scipy.linalg as la

    # Find Cholesky of C
    L = la.cholesky(C, lower=True)

    # Generate a standard Wishart
    A = standard_complex_wishart(C.shape[0], n, rng=rng)

    # Transform to get the Wishart variable
    return np.dot(L, np.dot(A, L.T.conj()))

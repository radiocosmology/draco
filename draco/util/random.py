"""Utilities for drawing random numbers."""

import contextlib

import numpy as np
import zlib

from caput import config
from ..core import task


_rng = None
_default_bitgen = np.random.SFC64


def default_rng():
    """Returns an instance of the default random number generator to use.

    This creates a randomly seeded generator using the fast SFC64 bit generator
    underneath. This is only initialise on the first call, subsequent calls will
    return the same Generator.

    Returns
    -------
    rng : np.random.Generator
    """
    global _rng

    if _rng is None:
        _rng = np.random.Generator(_default_bitgen())

    return _rng


def complex_normal(size=None, loc=0.0, scale=1.0, dtype=None, rng=None, out=None):
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
    rsize = size[:-1] + (size[-1] * 2,)
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
    return complex_normal(shape, dtype=dtype, rng=rng)


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


@contextlib.contextmanager
def mpi_random_seed(seed, extra=0, gen=None):
    """Use a specific random seed and return to the original state on exit.

    This is designed to work for MPI computations, incrementing the actual seed of
    each process by the MPI rank. Overall each process gets the numpy seed:
    `numpy_seed = seed + mpi_rank + 4096 * extra`. This can work for either the
    global numpy.random context or for new np.random.Generator.


    Parameters
    ----------
    seed : int
        Base seed to set. If seed is :obj:`None`, re-seed randomly.
    extra : int, optional
        An extra part of the seed, which should be changed for calculations
        using the same seed, but that want different random sequences.
    gen: :class: `Generator`
        A RandomGen bit_generator whose internal seed state we are going to
        influence.

    Yields
    ------
    If we are setting the numpy.random context, nothing is yielded.

    :class: `Generator`
        If we are setting the RandomGen bit_generator, it will be returned.
    """
    import warnings
    from caput import mpiutil

    warnings.warn(
        "This routine has fatal flaws. Try using `RandomTask` instead",
        category=DeprecationWarning,
    )

    # Just choose a random number per process as the seed if nothing was set.
    if seed is None:
        seed = np.random.randint(2**30)

    # Construct the new process specific seed
    new_seed = seed + mpiutil.rank + 4096 * extra
    np.random.seed(new_seed)

    # we will be setting the numpy.random context
    if gen is None:
        # Copy the old state for restoration later.
        old_state = np.random.get_state()

        # Enter the context block, and reset the state on exit.
        try:
            yield
        finally:
            np.random.set_state(old_state)

    # we will be setting the randomgen context
    else:
        # Copy the old state for restoration later.
        old_state = gen.state

        # Enter the context block, and reset the state on exit.
        try:
            yield gen
        finally:
            gen.state = old_state


class RandomTask(task.MPILoggedTask):
    """A base class for MPI tasks that needs to generate random numbers.

    Attributes
    ----------
    seed : int, optional
        Set the seed for use in the task. If not set, a random seed is generated and
        broadcast to all ranks. The seed being used is logged, to repeat a previous
        run, simply set this as the seed parameter.
    """

    seed = config.Property(proptype=int, default=None)

    _rng = None

    @property
    def rng(self):
        """A random number generator for this task.

        .. warning::
            Initialising the RNG is a collective operation if the seed is not set,
            and so all ranks must participate in the first access of this property.

        Returns
        -------
        rng : np.random.Generator
            A deterministically seeded random number generator suitable for use in
            MPI jobs.
        """
        if self._rng is None:
            # Generate a new base seed for all MPI ranks
            if self.seed is None:
                # Use seed sequence to generate a random seed
                seed = np.random.SeedSequence().entropy
                seed = self.comm.bcast(seed, root=0)
            else:
                seed = self.seed

            self.log.info("Using random seed: %i", seed)

            # Construct the new MPI-process and task specific seed. This mixes an
            # integer checksum of the class name with the MPI-rank to generate a new
            # hash.
            # NOTE: the slightly odd (rank + 1) is to ensure that even rank=0 mixes in
            # the class seed
            cls_name = "%s.%s" % (self.__module__, self.__class__.__name__)
            cls_seed = zlib.adler32(cls_name.encode())
            new_seed = seed + (self.comm.rank + 1) * cls_seed

            self._rng = np.random.Generator(_default_bitgen(new_seed))

        return self._rng

"""
====================================================================
Instrumental Noise Simulations (:mod:`~ch_pipeline.synthesis.noise`)
====================================================================

.. currentmodule:: ch_pipeline.synthesis.noise

Add the effects of instrumental noise into the simulation.

This is separated out into two tasks. The first, :class:`ReceiverTemperature`
adds in the effects of instrumental noise bias into the data. The second,
:class:`SampleNoise`, takes a timestream which is assumed to be the expected (or
average) value and returns an observed time stream.

Tasks
=====

.. autosummary::
    :toctree: generated/

    ReceiverTemperature
    SampleNoise
"""

import contextlib

import numpy as np

from caput import config

from ..core import task


class ReceiverTemperature(task.SingleTask):
    """Add a basic receiver temperature term into the data.

    This class adds in an uncorrelated, frequency and time independent receiver
    noise temperature to the data. As it is uncorrelated this will only affect
    the auto-correlations. Note this only adds in the offset to the visibility,
    to add the corresponding random fluctuations to subsequently use the
    :class:`SampleNoise` task.

    Attributes
    ----------
    recv_temp : float
        The receiver temperature in Kelvin.
    """
    recv_temp = config.Property(proptype=float, default=0.0)

    def process(self, data):

        # Iterate over the products to find the auto-correlations and add the noise into them
        for pi, prod in enumerate(data.index_map['prod']):

            # Great an auto!
            if prod[0] == prod[1]:
                data.vis[:, pi] += self.recv_temp

        return data


class SampleNoise(task.SingleTask):
    """Add properly distributed noise to a visibility dataset.

    This task draws properly (complex Wishart) distributed samples from an input
    visibility dataset which is assumed to represent the expectation.

    See http://link.springer.com/article/10.1007%2Fs10440-010-9599-x for a
    discussion of the Bartlett decomposition for complex Wishart distributed
    quantities.

    Attributes
    ----------
    sample_frac : float
        Multiplies the number of samples in each measurement. For instance this
        could be a duty cycle if the correlator was not keeping up, or could be
        larger than one if multiple measurements were combined.
    seed : int
        Random seed for the noise generation.
    """

    sample_frac = config.Property(proptype=float, default=1.0)
    seed = config.Property(proptype=int, default=None)

    def process(self, data_exp):
        """Generate a noisy dataset.

        Parameters
        ----------
        data_exp : :class:`containers.SiderealStream` or :class:`andata.CorrData`
            The expected (i.e. noiseless) visibility dataset. Must be the full
            triangle. Make sure you have added an instrumental noise bias if you
            want instrumental noise.

        Returns
        -------
        data_samp : same as :param:`data_exp`
            The sampled (i.e. noisy) visibility dataset.
        """

        from ..util import _fast_tools

        data_exp.redistribute('freq')

        nfeed = len(data_exp.index_map['input'])

        # Get a reference to the base MPIArray. Attempting to do this in the
        # loop fails if not all ranks enter the loop (as there is an implied MPI
        # Barrier)
        vis_data = data_exp.vis[:]

        with mpi_random_seed(self.seed):

            # Iterate over frequencies
            for lfi, fi in vis_data.enumerate(0):

                # Get the time and frequency intervals
                dt = data_exp.time[1] - data_exp.time[0]
                df = data_exp.index_map['freq']['width'][fi] * 1e6

                # Calculate the number of samples
                nsamp = int(self.sample_frac * dt * df)

                # Iterate over time
                for lti, ti in vis_data.enumerate(2):

                    # Unpack visibilites into full matrix
                    vis_utv = vis_data[lfi, :, lti].view(np.ndarray).copy()
                    vis_mat = np.zeros((nfeed, nfeed), dtype=vis_utv.dtype)
                    _fast_tools._unpack_product_array_fast(vis_utv, vis_mat, np.arange(nfeed), nfeed)

                    vis_samp = draw_complex_wishart(vis_mat, nsamp) / nsamp

                    vis_data[lfi, :, lti] = vis_samp[np.triu_indices(nfeed)]

        return data_exp


def standard_complex_wishart(m, n):
    """Draw a standard Wishart matrix.

    Parameters
    ----------
    m : integer
        Number of variables (i.e. size of matrix).
    n : integer
        Number of measurements the covariance matrix is estimated from.

    Returns
    -------
    B : np.ndarray[m, m]
    """

    from scipy.stats import gamma

    # Fill in normal variables in the lower triangle
    T = np.zeros((m, m), dtype=np.complex128)
    T[np.tril_indices(m, k=-1)] = (np.random.standard_normal(m * (m - 1) / 2) +
                                   1.0J * np.random.standard_normal(m * (m - 1) / 2)) / 2**0.5

    # Gamma variables on the diagonal
    for i in range(m):
        T[i, i] = gamma.rvs(n - i)**0.5

    # Return the square to get the Wishart matrix
    return np.dot(T, T.T.conj())


def draw_complex_wishart(C, n):
    """Draw a complex Wishart matrix.

    Parameters
    ----------
    C_exp : np.ndarray[:, :]
        Expected covaraince matrix.

    n : integer
        Number of measurements the covariance matrix is estimated from.

    Returns
    -------
    C_samp : np.ndarray
        Sample covariance matrix.
    """

    import scipy.linalg as la

    # Find Cholesky of C
    L = la.cholesky(C, lower=True)

    # Generate a standard Wishart
    A = standard_complex_wishart(C.shape[0], n)

    # Transform to get the Wishart variable
    return np.dot(L, np.dot(A, L.T.conj()))


@contextlib.contextmanager
def mpi_random_seed(seed, extra=0):
    """Use a specific random seed for the context, and return to the original state on exit.

    This is designed to work for MPI computations, incrementing the actual seed
    of each process by the MPI rank. Overall each process gets the numpy seed:
    `numpy_seed = seed + mpi_rank + 4096 * extra`.

    Parameters
    ----------
    seed : int
        Base seed to set. If seed is :obj:`None`, re-seed randomly.
    extra : int, optional
        An extra part of the seed, which should be changed for calculations
        using the same seed, but that want different random sequences.
    """

    import numpy as np
    from caput import mpiutil

    # Copy the old state for restoration later.
    old_state = np.random.get_state()

    # Just choose a random number per process as the seed if nothing was set.
    if seed is None:
        seed = np.random.randint(2**30)

    # Construct the new process specific seed
    new_seed = seed + mpiutil.rank + 4096 * extra
    np.random.seed(new_seed)

    # Enter the context block, and reset the state on exit.
    try:
        yield
    finally:
        np.random.set_state(old_state)

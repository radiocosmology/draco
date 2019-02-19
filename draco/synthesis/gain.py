"""Tasks for generating random gain fluctuations in the data.
At the moment there is only a very simple task which draws random Gaussian
distributed gain fluctuations.

Tasks
=====

.. autosummary::
    :toctree:

    RandomGains
"""

import numpy as np

from caput import config, mpiarray

from ..core import containers, task


class BaseGains(task.SingleTask):
    """Rudimentary class to generate gain timestreams.
    The gains are drawn for times which match up to an input time stream file

    Attributes
    ----------
    amp: bool
        Generate gain amplitude fluctuations. Default is True.
    phase: bool
        Generate gain phase fluctuations. Default is True.
    """
    amp = config.Property(default=True, proptype=bool)
    phase = config.Property(default=True, proptype=bool)

    _prev_time = None

    def process(self, data):
        """Generate a gain timestream for the inputs and times in `data`.

        Parameters
        ----------
        data : :class:`containers.TimeStream`
            Generate a timestream for this dataset.

        Returns
        -------
        gain : :class:`containers.GainData`
        """
        data.redistribute('prod')

        time = data.time

        gain_data = containers.GainData(time=time, axes_from=data)
        gain_data.redistribute('input')

        ninput = gain_data.gain.local_shape[1]
        freq = data.index_map['freq']['centre'][:]

        gain_amp = 1.0
        gain_phase = 0.0

        if self.amp:
            gain_amp = self._generate_amp(time, freq, ninput)

        if self.phase:
            gain_phase = self._generate_phase(time, freq, ninput)

        # Combine into an overall gain fluctuation
        gain_comb = gain_amp * np.exp(1.0J * gain_phase)

        # Copy the gain entries into the output container
        gain_comb = mpiarray.MPIArray.wrap(gain_comb, axis=1)
        gain_data.gain[:] = gain_comb

        # Keep a reference to time around for the next round
        self._prev_time = time

        return gain_data

    def _corr_func(self, zeta, amp):
        """This generates the correlation function

        Parameters
        ----------
        zeta: float
            Correlation length
        amp : float
            Amplitude (given as standard deviation) of fluctuations.
        """
        def _cf(x):
            dij = x[:, np.newaxis] - x[np.newaxis, :]
            return amp**2 * np.exp(-0.5 * (dij / zeta)**2)

        return _cf

    def _generate_amp(self, time, freq, ninput):
        """Generate phase gain errors.

        This implementation is blank. Must be overriden.

        Parameters:
        -----------
        time : np.ndarray
            Generate amplitude fluctuations for this time period.
        freq : np.ndarray
             Frequencies from data for which to generate gain fluctuations.
        ninput : float
            Number of inputs to generate fluctuations for.
        """
        pass

    def _generate_phase(self,  time, freq, ninput):
        """Generate phase gain errors.

        This implementation is blank. Must be overriden.

        Parameters:
        -----------
        time : np.ndarray
               Generate phase fluctuations for this time period.
        freq : np.ndarray
             Frequencies from data for which to generate gain fluctuations.
        ninput : float
            Number of inputs to generate fluctuations for.
        """
        pass


class RandomGains(BaseGains):
    r"""Generate random gains.

    Notes
    -----
    The Random Gains class generates random fluctuations in gain amplitude and
    phase.

    Attributes
    ----------
    corr_length_amp, corr_length_phase : float
        Correlation length for amplitude and phase fluctuations in seconds.
    sigma_amp, sigma_phase : float
        Size of fluctuations for amplitude (fractional), and phase (radians).

    Notes
    -----
    This task generates gain time streams which are Gaussian distributed with
    covariance

    .. math::
        C_{ij} = \sigma^2 \exp{\left(-\frac{1}{2 \xi^2}(t_i - t_j)^2\right)}

    As the time stream is generated in separate pieces, to ensure that there is
    consistency between them each gain time stream is drawn as a constrained
    realisation against the previous file.
    """
    corr_length_amp = config.Property(default=3600.0, proptype=float)
    corr_length_phase = config.Property(default=3600.0, proptype=float)

    sigma_amp = config.Property(default=0.02, proptype=float)
    sigma_phase = config.Property(default=0.1, proptype=float)

    _prev_amp = None
    _prev_phase = None

    def _generate_amp(self, time, freq, ninput):

        # Generate the correlation function
        cf_amp = self._corr_func(self.corr_length_amp, self.sigma_amp)
        num_realisations = len(freq) * ninput
        ntime = len(time)

        # Generate amplitude fluctuations
        gain_amp = generate_fluctuations(time, cf_amp, self._prev_time,
                                         self._prev_amp, num_realisations)

        # Save amplitude fluctuations to instannce
        self._prev_amp = gain_amp

        gain_amp = gain_amp.reshape((len(freq), ninput, ntime))

        gain_amp = 1.0 + gain_amp

        return gain_amp

    def _generate_phase(self, time, freq, ninput):

        # Generate the correlation function
        cf_phase = self._corr_func(self.corr_length_phase, self.sigma_phase)
        num_realisations = len(freq) * ninput
        ntime = len(time)

        # Generate phase fluctuations
        gain_phase_fluc = generate_fluctuations(time, cf_phase, self._prev_time,
                                                self._prev_phase, num_realisations)

        # Save phase fluctuations to instannce
        self._prev_phase = gain_phase_fluc
        # Reshape to correct size
        gain_phase_fluc = gain_phase_fluc.reshape((len(freq), ninput, ntime))

        return gain_phase_fluc


def generate_fluctuations(x, corrfunc, n, prev_x, prev_fluc):
    """Generate correlated random streams.

    Generates a Gaussian field from the given correlation function and (potentially)
    correlated with prior data.

    Parameters
    ----------
    x : np.ndarray[npoints]
        Coordinates of samples in the new stream.
    corrfunc : function
        See documentation of `gaussian_realisation`.
    prev_x : np.ndarray[npoints]
        Coordinates of previous samples. Ignored if `prev_fluc` is None.
    prev_fluc : np.ndarray[npoints]
        Values of previous samples. If `None` the stream is initialised.
    n : int
        Number of realisations to generate.

    Returns
    -------
    y : np.ndarray[n, npoints]
        Realisations of the stream.
    """
    nx = len(x)

    if prev_fluc is None:
        fluctuations = gaussian_realisation(x, corrfunc, n).reshape(n, nx)

    else:
        fluctuations = constrained_gaussian_realisation(
            x, corrfunc, n, prev_x, prev_fluc).reshape(n, nx)

    return fluctuations


def gaussian_realisation(x, corrfunc, n, rcond=1e-12):
    """Generate a Gaussian random field.

    Parameters
    ----------
    x : np.ndarray[npoints] or np.ndarray[npoints, ndim]
        Co-ordinates of points to generate.
    corrfunc : function(x) -> covariance matrix
        Function that take (vectorized) co-ordinates and returns their
        covariance functions.
    n : integer
        Number of realisations to generate.
    rcond : float, optional
        Ignore eigenmodes smaller than `rcond` times the largest eigenvalue.

    Returns
    -------
    y : np.ndarray[n, npoints]
        Realisations of the gaussian field.
    """
    return _realisation(corrfunc(x), n, rcond)


def _realisation(C, n, rcond):
    """Create a realisation of the given covariance matrix. Regularise by
    throwing away small eigenvalues.
    """

    import scipy.linalg as la

    # Find the eigendecomposition, truncate small modes, and use this to
    # construct a matrix projecting from the non-singular space
    evals, evecs = la.eigh(C)
    num = np.sum(evals > rcond * evals[-1])
    R = evecs[:, -num:] * evals[np.newaxis, -num:]**0.5

    # Generate independent gaussian variables
    w = np.random.standard_normal((n, num))

    # Apply projection to get random field
    return np.dot(w, R.T)


def constrained_gaussian_realisation(x, corrfunc, n, x2, y2, rcond=1e-12):
    """Generate a constrained Gaussian random field.

    Given a correlation function generate a Gaussian random field that is
    consistent with an existing set of values :param:`y2` located at
    co-ordinates :param:`x2`.

    Parameters
    ----------
    x : np.ndarray[npoints] or np.ndarray[npoints, ndim]
        Co-ordinates of points to generate.
    corrfunc : function(x) -> covariance matrix
        Function that take (vectorized) co-ordinates and returns their
        covariance functions.
    n : integer
        Number of realisations to generate.
    x2 : np.ndarray[npoints] or np.ndarray[npoints, ndim]
        Co-ordinates of existing points.
    y2 : np.ndarray[npoints] or np.ndarray[n, npoints]
        Existing values of the random field.
    rcond : float, optional
        Ignore eigenmodes smaller than `rcond` times the largest eigenvalue.

    Returns
    -------
    y : np.ndarray[n, npoints]
        Realisations of the gaussian field.
    """
    import scipy.linalg as la

    if (y2.ndim >= 2) and (n != y2.shape[0]):
        raise ValueError('Array y2 of existing data has the wrong shape.')

    # Calculate the covariance matrix for the full dataset
    xc = np.concatenate([x, x2])
    M = corrfunc(xc)

    # Select out the different blocks
    l = len(x)
    A = M[:l, :l]
    B = M[:l, l:]
    C = M[l:, l:]

    # This method tends to be unstable when there are singular modes in the
    # covariance matrix (i.e. modes with zero variance). We can remove these by
    # projecting onto the non-singular modes.

    # Find the eigendecomposition and construct projection matrices onto the
    # non-singular space
    evals_A, evecs_A = la.eigh(A)
    evals_C, evecs_C = la.eigh(C)

    num_A = np.sum(evals_A > rcond * evals_A.max())
    num_C = np.sum(evals_C > rcond * evals_C.max())

    R_A = evecs_A[:, -num_A:]
    R_C = evecs_C[:, -num_C:]

    # Construct the covariance blocks in the reduced basis
    A_r = np.diag(evals_A[-num_A:])
    B_r = np.dot(R_A.T, np.dot(B, R_C))
    Ci_r = np.diag(1.0 / evals_C[-num_C:])

    # Project the existing data into the new basis
    y2_r = np.dot(y2, R_C)

    # Calculate the mean of the new variables
    z_r = np.dot(y2_r, np.dot(Ci_r, B_r.T))

    # Generate fluctuations for the new variables (in the reduced basis)
    Ap_r = A_r - np.dot(B_r, np.dot(Ci_r, B_r.T))
    y_r = _realisation(Ap_r, n, rcond)

    # Project into the original basis for A
    y = np.dot(z_r + y_r, R_A.T)

    return y

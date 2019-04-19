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

from caput import config, mpiarray, pipeline
from ch_util import ephemeris

from ..core import containers, task, io

SOLAR_SEC_PER_DAY = 86400.


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

        self.ninput_local = gain_data.gain.local_shape[1]
        self.ninput_global = gain_data.gain.global_shape[1]
        freq = data.index_map['freq']['centre'][:]

        gain = self._generate_gain(time, freq)

        # Copy the gain entries into the output container
        gain = mpiarray.MPIArray.wrap(gain, axis=1)
        gain_data.gain[:] = gain

        # Keep a reference to time around for the next round
        self._prev_time = time

        return gain_data

    def _generate_gain(self, time, freq):

        self._gain_hook(time)

        gain_amp = 1.0
        gain_phase = 0.0

        if self.amp:
            gain_amp = self._generate_amp(time, freq)

        if self.phase:
            gain_phase = self._generate_phase(time, freq)

        # Combine into an overall gain fluctuation
        gain_comb = gain_amp * np.exp(1.0J * gain_phase)

        return gain_comb

    def _gain_hook(self, time):
        """This hook function is intended to be able to generate stuff for
        both amplitude and phase. Can be implemented in child classes but must not.

        Parameters:
        -----------
        time : np.ndarray
            Generate whatever you need for both amplitude and phase for time
            period given.
        """
        pass

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

    def _generate_amp(self, time, freq):
        """Generate phase gain errors.

        This implementation is blank. Must be overriden.

        Parameters:
        -----------
        time : np.ndarray
            Generate amplitude fluctuations for this time period.
        freq : np.ndarray
            Frequencies from data for which to generate gain fluctuations.
            """
        raise NotImplementedError

    def _generate_phase(self,  time, freq):
        """Generate phase gain errors.

        This implementation is blank. Must be overriden.

        Parameters:
        -----------
        time : np.ndarray
           Generate phase fluctuations for this time period.
        freq : np.ndarray
            Frequencies from data for which to generate gain fluctuations.
        """
        raise NotImplementedError


class SiderealGains(BaseGains):
    """Task for simulating sidereal gains

    Attributes
    ----------
    start_time, end_time : float or datetime
        Start and end times of the timestream to simulate. Needs to be either a
        `float` (UNIX time) or a `datetime` objects in UTC.
    approx_integration_time : float
        The integration time in solar seconds you are aiming for. Based on this
        information the task will calculate the number of samples per file
        within a sidereal day.
    """

    start_time = config.utc_time()
    end_time = config.utc_time()
    approx_integration_time = config.Property(proptype=float, default=60.)

    def setup(self, bt, sstream):
        """Set up the sidereal gain errors task

        Parameters
        ----------
        bt : BeamTransfer
        sstream : SiderealStream
        """
        self.observer = io.get_telescope(bt)
        self.lsd_start = self.observer.unix_to_lsd(self.start_time)
        self.lsd_end = self.observer.unix_to_lsd(self.end_time)

        sid_sec_per_day = SOLAR_SEC_PER_DAY *  ephemeris.SIDEREAL_S
        self.samples_per_file = int(sid_sec_per_day / self.approx_integration_time)
        self.integration_time = sid_sec_per_day / self.samples_per_file

        self.log.info("Sidereal period requested for this simulation",
                       int(self.lsd_start), int(self.lsd_end))

        # Initialize the current lsd time
        self._current_lsd = None
        self.sstream = sstream

    def process(self):
        """Generate a gain timestream for the inputs and times in `data`.

        Parameters
        ----------
        data : :class:`containers.SiderealStream`
            Generate complex gains for this sidereal stream.

        Returns
        -------
        gain : :class:`containers.SiderealGainData`
        """
        # If current_lsd is None then this is the first time we've run
        if self._current_lsd is None:
            # Check if lsd is an integer, if not add an lsd
            if isinstance(self.lsd_start, int):
                self._current_lsd = int(self.lsd_start)
            else:
                self._current_lsd = int(self.lsd_start + 1)

        # Check if we have reached the end of the requested time
        if self._current_lsd >= self.lsd_end:
            raise pipeline.PipelineStopIteration

        # Convert the current lsd day to unix time
        unix_start = self.observer.lsd_to_unix(self._current_lsd)
        unix_end = self.observer.lsd_to_unix(self._current_lsd + 1)

        # Distribute the sidereal data and create a time array
        data = self.sstream
        data.redistribute('prod')
        freq = data.index_map['freq']['centre'][:]
        ra = np.linspace(0.0, 360.0, self.samples_per_file, endpoint=False)
        time = np.linspace(unix_start, unix_end, self.samples_per_file)

        # Make a sidereal gain data container
        gain_data = containers.SiderealGainData(axes_from=data, ra=ra)
        gain_data.redistribute('input')

        self.ninput_local = gain_data.gain.local_shape[1]
        self.ninput_global = gain_data.gain.global_shape[1]

        gain = self._generate_gain(time, freq)

        # Copy the gain entries into the output container
        gain = mpiarray.MPIArray.wrap(gain, axis=1)
        gain_data.gain[:] = gain
        gain_data.attrs['lsd'] = self._current_lsd
        gain_data.attrs['tag'] = 'lsd_%i' % self._current_lsd
        gain_data.attrs['int_time'] = self.integration_time

        self._current_lsd += 1

        # Keep a reference to time around for the next round
        self._prev_time = time

        return gain_data


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

    def _generate_amp(self, time, freq):

        # Generate the correlation function
        cf_amp = self._corr_func(self.corr_length_amp, self.sigma_amp)
        ninput = self.ninput_local
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

    def _generate_phase(self, time, freq):

        # Generate the correlation function
        cf_phase = self._corr_func(self.corr_length_phase, self.sigma_phase)
        ninput = self.input_local
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


class GainStacker(task.SingleTask):
    r"""Take sidereal gain data, make products and stack them up.

    Attributes
    ----------
    only_gains : bool
        Whether to return only the stacked gains or the stacked gains
        mulitplied with the vis. Default: False.

    Notes
    -----
    This task generates products of gain time streams for every day and stacks
    them up.

    Gain stack

    .. math::
        G_{ij} = \sum_{a} g_{i}(t)^{a} g_j(t)^{*a}
    """

    only_gains = config.Property(default=False, proptype=bool)

    gain_stack = None
    lsd_list = None

    def setup(self, stream):
        self.stream = stream

    def process(self, gain):
        """Make sidereal gain products and stack them up.

        Parameters
        ----------
        gain : containers.SiderealGainData or containers.GainData
            Individual sidereal or time ordered gain data

        Returns
        -------
        gain_stack : containers.TimeStream or containers.SiderealStream
            Stacked products of gains.
        """

        stream = self.stream
        stream.redistribute('freq')
        gain.redistribute('freq')
        prod = stream.index_map['prod']

        if 'lsd' in gain.attrs:
            input_lsd = gain.attrs['lsd']
        else:
            input_lsd = -1

        input_lsd = _ensure_list(input_lsd)

        # If gain_stack is None create an MPIArray to hold the product expanded
        # gain data and redistribute over all freq
        if self.gain_stack is None:

            self.gain_stack = containers.empty_like(stream)
            self.gain_stack.redistribute('freq')

            for pi, (fi, fj) in enumerate(prod):
                self.gain_stack.vis[:, pi, :] = gain.gain[:, fi, :] * np.conjugate(gain.gain[:, fj, :])

            # Creating a counter to increment during stacking
            # self.counter = np.ones(self.gain_stack.vis.local_shape)
            self.gain_stack.weight[:] = np.ones(self.gain_stack.vis.local_shape)

            self.lsd_list = input_lsd

            self.log.info("Starting gain stack with LSD:%i", input_lsd)

            return

        # Keep gains around for next round, save current lsd to list, log
        self.log.info("Adding LSD:%i to gain stack", gain.attrs['lsd'])

        # Calculate the gain products
        for pi, (fi, fj) in enumerate(prod):
            self.gain_stack.vis[:, pi] += (gain.gain[:, fi] * np.conjugate(gain.gain[:, fj]))

        # self.counter += np.ones(self.gain_stack.vis.local_shape)
        self.gain_stack.weight[:] += np.ones(self.gain_stack.vis.local_shape)

        self.lsd_list += input_lsd

    def process_finish(self):
        """Multiply summed gain with sidereal stream.

        Returns
        -------
        data : containers.SiderealStream or containers.TimeStream
            Stack of sidereal data with gain applied.
        """
        # If requested, or shapes of visibilties and gain stack don't match
        # just return stack
        if (self.stream.vis[:].shape[-1] != self.gain_stack.vis[:].shape[-1]) or self.only_gains:
            self.log.info("Saving only gain stack - either by request or shapes of visibilites and gain stack do not match")

            self.gain_stack.vis[:] = (self.gain_stack.vis[:] / self.gain_stack.weight[:])

            return self.gain_stack

        data = containers.empty_like(self.stream)
        data.redistribute('freq')

        self.gain_stack.vis[:] = (self.gain_stack.vis[:] / self.gain_stack.weight[:])
        data.vis[:] = self.stream.vis[:] * self.gain_stack.vis[:]

        # for ii in range(0, data.vis.shape[1], 128):
        #    self.gain_stack.vis[:, ii:ii + 128] = (self.gain_stack.vis[:, ii:ii + 128] / self.counter[:, ii:ii + 128])
        #    data.vis[:, ii:ii + 128] = self.stream.vis[:, ii:ii + 128] * self.gain_stack.vis[:, ii:ii + 128]

        data.attrs['tag'] = 'gain_stack'
        # data.attrs['counter'] = self.counter

        return data


def _ensure_list(x):

    if hasattr(x, '__iter__'):
        y = [xx for xx in x]
    else:
        y = [x]

    return y


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

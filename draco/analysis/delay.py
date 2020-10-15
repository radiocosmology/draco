"""Delay space spectrum estimation and filtering.
"""

import numpy as np
import scipy.linalg as la
import scipy.stats as st

try:
    # For backwards compatibility
    from randomgen import RandomGenerator
except ImportError:
    from randomgen import Generator as RandomGenerator


from caput import mpiarray, config
from cora.util import units

from ..core import containers, task, io


class DelayFilter(task.SingleTask):
    """Remove delays less than a given threshold.

    This is performed by projecting the data onto the null space that is orthogonal
    to any mode at low delays.

    Attributes
    ----------
    delay_cut : float
        Delay value to filter at in seconds.
    za_cut : float
        Sine of the maximum zenith angle included in baseline-dependent delay
        filtering. Default is 1 which corresponds to the horizon (ie: filters out all
        zenith angles). Setting to zero turns off baseline dependent cut.
    extra_cut : float
        Increase the delay threshold beyond the baseline dependent term.
    update_weight : bool
        Updates the weight dataset to reflect frequency channels that had too low
        weight to be included by the filter.
    weight_tol : float
        Maximum weight kept in the masked data, as a fraction of the largest weight
        in the original dataset.
    telescope_orientation : one of ('NS', 'EW', 'none')
        Determines if the baseline-dependent delay cut is based on the north-south
        component, the east-west component or the full baseline length. For
        cylindrical telescopes oriented in the NS direction (like CHIME) use 'NS'.
        The default is 'NS'.
    window : bool
        Apply the window function to the data when applying the filter.

    Notes
    -----
    The delay cut applied is `max(za_cut * baseline / c + extra_cut, delay_cut)`.
    """

    delay_cut = config.Property(proptype=float, default=0.1)
    za_cut = config.Property(proptype=float, default=1.0)
    extra_cut = config.Property(proptype=float, default=0.0)
    update_weight = config.Property(proptype=bool, default=True)
    weight_tol = config.Property(proptype=float, default=1e-4)
    telescope_orientation = config.enum(["NS", "EW", "none"], default="NS")
    window = config.Property(proptype=bool, default=False)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, ss):
        """Filter out delays from a SiderealStream or TimeStream.

        Parameters
        ----------
        ss : containers.SiderealStream
            Data to filter.

        Returns
        -------
        ss_filt : containers.SiderealStream
            Filtered dataset.
        """
        tel = self.telescope

        ss.redistribute(["input", "prod", "stack"])

        freq = ss.freq[:]
        bandwidth = np.ptp(freq)

        ssv = ss.vis[:].view(np.ndarray)
        ssw = ss.weight[:].view(np.ndarray)

        ubase, uinv = np.unique(
            tel.baselines[:, 0] + 1.0j * tel.baselines[:, 1], return_inverse=True
        )
        ubase = ubase.view(np.float64).reshape(-1, 2)

        for lbi, bi in ss.vis[:].enumerate(axis=1):

            # Select the baseline length to use
            baseline = ubase[uinv[bi]]
            if self.telescope_orientation == "NS":
                baseline = abs(baseline[1])  # Y baseline
            elif self.telescope_orientation == "EW":
                baseline = abs(baseline[0])  # X baseline
            else:
                baseline = np.linalg.norm(baseline)  # Norm
            # In micro seconds
            baseline_delay_cut = self.za_cut * baseline / units.c * 1e6 + self.extra_cut
            delay_cut = np.amax([baseline_delay_cut, self.delay_cut])

            # Calculate the number of samples needed to construct the delay null space.
            # `4 * tau_max * bandwidth` is the amount recommended in the DAYENU paper
            # and seems to work well here
            number_cut = int(4.0 * bandwidth * delay_cut + 0.5)

            weight_mask = np.median(ssw[:, lbi], axis=1)
            weight_mask = (weight_mask > (self.weight_tol * weight_mask.max())).astype(
                np.float64
            )
            NF = null_delay_filter(
                freq, delay_cut, weight_mask, num_delay=number_cut, window=self.window
            )

            ssv[:, lbi] = np.dot(NF, ssv[:, lbi])

            if self.update_weight:
                ssw[:, lbi] *= weight_mask[:, np.newaxis]

        return ss


class DelaySpectrumEstimator(task.SingleTask):
    """Calculate the delay spectrum of a Sidereal/TimeStream for instrumental Stokes I.

    The spectrum is calculated by Gibbs sampling. However, at the moment only
    the final sample is used to calculate the spectrum.

    Attributes
    ----------
    nsamp : int, optional
        The number of Gibbs samples to draw.
    freq_zero : float, optional
        The physical frequency (in MHz) of the *zero* channel. That is the DC
        channel coming out of the F-engine. If not specified, use the first
        frequency channel of the stream.
    freq_spacing : float, optional
        The spacing between the underlying channels (in MHz). This is conjugate
        to the length of a frame of time samples that is transformed. If not
        set, then use the smallest gap found between channels in the dataset.
    nfreq : int, optional
        The number of frequency channels in the full set produced by the
        F-engine. If not set, assume the last included frequency is the last of
        the full set (or is the penultimate if `skip_nyquist` is set).
    skip_nyquist : bool, optional
        Whether the Nyquist frequency is included in the data. This is `True` by
        default to align with the output of CASPER PFBs.
    """

    nsamp = config.Property(proptype=int, default=20)
    freq_zero = config.Property(proptype=float, default=None)
    freq_spacing = config.Property(proptype=float, default=None)
    nfreq = config.Property(proptype=int, default=None)
    skip_nyquist = config.Property(proptype=bool, default=True)

    def setup(self, telescope):
        """Set the telescope needed to generate Stokes I.

        Parameters
        ----------
        telescope : TransitTelescope
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, ss):
        """Estimate the delay spectrum.

        Parameters
        ----------
        ss : SiderealStream or TimeStream

        Returns
        -------
        dspec : DelaySpectrum
        """

        tel = self.telescope

        ss.redistribute("freq")

        # Construct the Stokes I vis
        vis_I, vis_weight, baselines = stokes_I(ss, tel)

        # ==== Figure out the frequency structure and delay values ====
        if self.freq_zero is None:
            self.freq_zero = ss.freq[0]

        if self.freq_spacing is None:
            self.freq_spacing = np.abs(np.diff(ss.freq[:])).min()

        channel_ind = (np.abs(ss.freq[:] - self.freq_zero) / self.freq_spacing).astype(
            np.int
        )

        if self.nfreq is None:
            self.nfreq = channel_ind[-1] + 1

            if self.skip_nyquist:
                self.nfreq += 1

        # Assume each transformed frame was an even number of samples long
        ndelay = 2 * (self.nfreq - 1)
        delays = np.fft.fftshift(np.fft.fftfreq(ndelay, d=self.freq_spacing))  # in us

        # Initialise the spectrum container
        delay_spec = containers.DelaySpectrum(baseline=baselines, delay=delays)
        delay_spec.redistribute("baseline")
        delay_spec.spectrum[:] = 0.0

        initial_S = np.ones_like(delays) * 1e1

        # Iterate over all baselines and use the Gibbs sampler to estimate the spectrum
        for lbi, bi in delay_spec.spectrum[:].enumerate(axis=0):

            self.log.debug("Delay transforming baseline %i/%i", bi, len(baselines))

            # Get the local selections
            data = vis_I[lbi].view(np.ndarray).T
            weight = vis_weight[lbi].view(np.ndarray)

            # Mask out data with completely zero'd weights and generate time
            # averaged weights
            weight_cut = (
                1e-4 * weight.mean()
            )  # Use approx threshold to ignore small weights
            data = data * (weight.T > weight_cut)
            weight = np.mean(weight, axis=1)

            if (data == 0.0).all():
                continue

            spec = delay_spectrum_gibbs(
                data, ndelay, weight, initial_S, fsel=channel_ind, niter=self.nsamp
            )

            # Take an average over the last half of the delay spectrum samples
            # (presuming that removes the burn-in)
            spec_av = np.median(spec[-(self.nsamp // 2) :], axis=0)
            delay_spec.spectrum[bi] = np.fft.fftshift(spec_av)

        return delay_spec


def stokes_I(sstream, tel):
    """Extract instrumental Stokes I from a time/sidereal stream.

    Parameters
    ----------
    sstream : containers.SiderealStream, container.TimeStream
        Stream of correlation data.
    tel : TransitTelescope
        Instance describing the telescope.

    Returns
    -------
    vis_I : mpiarray.MPIArray[nbase, nfreq, ntime]
        The instrumental Stokes I visibilities, distributed over baselines.
    vis_weight : mpiarray.MPIArray[nbase, nfreq, ntime]
        The weights for each visibility, distributed over baselines.
    baselines : np.ndarray[nbase, 2]
    """

    # ==== Unpack into Stokes I
    ubase, uinv, ucount = np.unique(
        tel.baselines[:, 0] + 1.0j * tel.baselines[:, 1],
        return_inverse=True,
        return_counts=True,
    )
    ubase = ubase.view(np.float64).reshape(-1, 2)
    nbase = ubase.shape[0]

    vis_shape = (nbase, sstream.vis.local_shape[0], sstream.vis.local_shape[2])
    vis_I = np.zeros(vis_shape, dtype=sstream.vis.dtype)
    vis_weight = np.zeros(vis_shape, dtype=sstream.weight.dtype)

    # Iterate over products to construct the Stokes I vis
    # TODO: this should be updated when driftscan gains a concept of polarisation
    ssv = sstream.vis[:]
    ssw = sstream.weight[:]
    for ii, ui in enumerate(uinv):

        # Skip if not all polarisations were included
        if ucount[ui] < 4:
            continue

        fi, fj = tel.uniquepairs[ii]
        bi, bj = tel.beamclass[fi], tel.beamclass[fj]

        upi = tel.feedmap[fi, fj]

        if upi == -1:
            continue

        if bi == bj:
            vis_I[ui] += ssv[:, ii]
            vis_weight[ui] += ssw[:, ii]

    vis_I = mpiarray.MPIArray.wrap(vis_I, axis=1, comm=sstream.comm)
    vis_I = vis_I.redistribute(axis=0)
    vis_weight = mpiarray.MPIArray.wrap(
        vis_weight, axis=1, comm=sstream.comm
    ).redistribute(axis=0)

    return vis_I, vis_weight, ubase


def window_generalised(x, window="nuttall"):
    """A generalised high-order window at arbitrary locations.

    Parameters
    ----------
    x : np.ndarray[n]
        Location to evaluate at. Must be in the range 0 to 1.
    window : one of {'nuttall', 'blackman_nuttall', 'blackman_harris'}
        Type of window function to return.

    Returns
    -------
    w : np.ndarray[n]
        Window function.
    """

    a_table = {
        "nuttall": np.array([0.355768, -0.487396, 0.144232, -0.012604]),
        "blackman_nuttall": np.array([0.3635819, -0.4891775, 0.1365995, -0.0106411]),
        "blackman_harris": np.array([0.35875, -0.48829, 0.14128, -0.01168]),
    }

    a = a_table[window]

    t = 2 * np.pi * np.arange(4)[:, np.newaxis] * x[np.newaxis, :]

    w = (a[:, np.newaxis] * np.cos(t)).sum(axis=0)

    return w


def fourier_matrix_r2c(N, fsel=None):
    """Generate a Fourier matrix to represent a real to complex FFT.

    Parameters
    ----------
    N : integer
        Length of timestream that we are transforming to. Must be even.
    fsel : array_like, optional
        Indexes of the frequency channels to include in the transformation
        matrix. By default, assume all channels.

    Returns
    -------
    F : np.ndarray
        An array performing the Fourier transform from a real time series to
        frequencies packed as alternating real and imaginary elements,
    """

    if fsel is None:
        fa = np.arange(N // 2 + 1)
    else:
        fa = np.array(fsel)

    fa = fa[:, np.newaxis]
    ta = np.arange(N)[np.newaxis, :]

    Fr = np.zeros((2 * fa.shape[0], N), dtype=np.float64)

    Fr[0::2] = np.cos(2 * np.pi * ta * fa / N)
    Fr[1::2] = -np.sin(2 * np.pi * ta * fa / N)

    return Fr


def fourier_matrix_c2r(N, fsel=None):
    """Generate a Fourier matrix to represent a complex to real FFT.

    Parameters
    ----------
    N : integer
        Length of timestream that we are transforming to. Must be even.
    fsel : array_like, optional
        Indexes of the frequency channels to include in the transformation
        matrix. By default, assume all channels.

    Returns
    -------
    F : np.ndarray
        An array performing the Fourier transform from frequencies packed as
        alternating real and imaginary elements, to the real time series.
    """

    if fsel is None:
        fa = np.arange(N // 2 + 1)
    else:
        fa = np.array(fsel)

    fa = fa[np.newaxis, :]

    mul = np.where((fa == 0) | (fa == N // 2), 1.0, 2.0) / N

    ta = np.arange(N)[:, np.newaxis]

    Fr = np.zeros((N, 2 * fa.shape[1]), dtype=np.float64)

    Fr[:, 0::2] = np.cos(2 * np.pi * ta * fa / N) * mul
    Fr[:, 1::2] = -np.sin(2 * np.pi * ta * fa / N) * mul

    return Fr


# RNG used for delay estimation
_delay_rng = RandomGenerator()


def delay_spectrum_gibbs(data, N, Ni, initial_S, window=True, fsel=None, niter=20):
    """Estimate the delay spectrum by Gibbs sampling.

    This routine estimates the spectrum at the `N` delay samples conjugate to
    the frequency spectrum of ``N/2 + 1`` channels. A subset of these channels
    can be specified using the `fsel` argument.

    Parameters
    ----------
    data : np.ndarray[:, freq]
        Data to estimate the delay spectrum of.
    N : int
        The length of the output delay spectrum. There are assumed to `N/2 + 1`
        total frequency channels.
    Ni : np.ndarray[freq]
        Inverse noise variance.
    initial_S : np.ndarray[delay]
        The initial delay spectrum guess.
    window : bool, optional
        Apply a Nuttall apodisation function. Default is True.
    fsel : np.ndarray[freq], optional
        Indices of channels that we have data at. By default assume all channels.
    niter : int, optional
        Number of Gibbs samples to generate.

    Returns
    -------
    spec : list
        List of spectrum samples.
    """

    # Get reference to RNG
    # TODO: do something smarter here
    # TODO: multithread RNG generation
    rng = _delay_rng

    spec = []

    total_freq = N // 2 + 1

    if fsel is None:
        fsel = np.arange(total_freq)

    # Construct the Fourier matrix
    F = fourier_matrix_r2c(N, fsel)

    # Construct a view of the data with alternating real and imaginary parts
    data = data.astype(np.complex128, order="C").view(np.float64).T.copy()

    # Window the frequency data
    if window:

        # Construct the window function
        x = fsel * 1.0 / total_freq
        w = window_generalised(x, window="nuttall")
        w = np.repeat(w, 2)

        # Apply to the projection matrix and the data
        F *= w[:, np.newaxis]
        data *= w[:, np.newaxis]

    is_real_freq = (fsel == 0) | (fsel == N // 2)

    # Construct the Noise inverse array for the real and imaginary parts (taking
    # into account that the zero and Nyquist frequencies are strictly real)
    Ni_r = np.zeros(2 * Ni.shape[0])
    Ni_r[0::2] = np.where(is_real_freq, Ni, Ni / 2 ** 0.5)
    Ni_r[1::2] = np.where(is_real_freq, 0.0, Ni / 2 ** 0.5)

    # Create the Hermitian conjugate weighted by the noise (this is used multiple times)
    FTNih = F.T * Ni_r[np.newaxis, :] ** 0.5
    FTNiF = np.dot(FTNih, FTNih.T)

    # Pre-whiten the data to save doing it repeatedly
    data = data * Ni_r[:, np.newaxis] ** 0.5

    # Set the initial starting points
    S_samp = initial_S

    def _draw_signal_sample(S):
        # Draw a random sample of the signal assuming a Gaussian model with a
        # given delay spectrum shape. Do this using the perturbed Wiener filter
        # approach

        # TODO: we can probably change the order that the Wiener filter is
        # evaluated for some computational saving as nfreq < ndelay

        # Construct the Wiener covariance
        Si = 1.0 / S
        Ci = np.diag(Si) + FTNiF

        # Draw random vectors that form the perturbations
        w1 = rng.standard_normal((N, data.shape[1]))
        w2 = rng.standard_normal(data.shape)

        # Construct the random signal sample by forming a perturbed vector and
        # then doing a matrix solve
        y = np.dot(FTNih, data + w2) + Si[:, np.newaxis] ** 0.5 * w1

        return la.solve(Ci, y, sym_pos=True)

    def _draw_ps_sample(d):
        # Draw a random power spectrum sample assuming from the signal assuming
        # the signal is Gaussian and we have a flat prior on the power spectrum.
        # This means drawing from a inverse chi^2.

        S_hat = d.var(axis=1)

        df = d.shape[1]
        chi2 = rng.chisquare(df, size=d.shape[0])

        S_samp = S_hat * df / chi2

        return S_samp

    # Perform the Gibbs sampling iteration for a given number of loops and
    # return the power spectrum output of them.
    for ii in range(niter):

        d_samp = _draw_signal_sample(S_samp)
        S_samp = _draw_ps_sample(d_samp)

        spec.append(S_samp)

    return spec


def null_delay_filter(freq, max_delay, mask, num_delay=200, tol=1e-8, window=True):
    """Take frequency data and null out any delays below some value.

    Parameters
    ----------
    freq : np.ndarray[freq]
        Frequencies we have data at.
    max_delay : float
        Maximum delay to keep.
    mask : np.ndarray[freq]
        Frequencies to mask out.
    num_delay : int, optional
        Number of delay values to use.
    tol : float, optional
        Cut off value for singular values.
    window : bool, optional
        Apply a window function to the data while filtering.

    Returns
    -------
    filter : np.ndarray[freq, freq]
        The filter as a 2D matrix.
    """

    # Construct the window function
    x = (freq - freq.min()) / freq.ptp()
    w = window_generalised(x, window="nuttall")

    delay = np.linspace(-max_delay, max_delay, num_delay)

    # Construct the Fourier matrix
    F = (mask * w)[:, np.newaxis] * np.exp(
        2.0j * np.pi * delay[np.newaxis, :] * freq[:, np.newaxis]
    )

    # Use an SVD to figure out the set of significant modes spanning the delays
    # we are wanting to get rid of.
    u, sig, vh = la.svd(F)
    nmodes = np.sum(sig > tol * sig.max())
    p = u[:, :nmodes]

    # Construct a projection matrix for the filter
    proj = np.identity(len(freq)) - np.dot(p, p.T.conj())

    if window:
        proj = (mask * w)[np.newaxis, :] * proj

    return proj

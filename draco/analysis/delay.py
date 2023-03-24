"""Delay space spectrum estimation and filtering."""

import typing

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import scipy.linalg as la

from caput import mpiarray, config
from cora.util import units

from ..core import containers, task, io
from ..util import random


class DelayFilter(task.SingleTask):
    """Remove delays less than a given threshold.

    This is performed by projecting the data onto the null space that is orthogonal
    to any mode at low delays.

    Note that for this task to work best the zero entries in the weights dataset
    should factorize in frequency-time for each baseline. A mostly optimal masking
    can be generated using the `draco.analysis.flagging.MaskFreq` task.

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
    weight_tol = config.Property(proptype=float, default=1e-4)
    telescope_orientation = config.enum(["NS", "EW", "none"], default="NS")
    window = config.Property(proptype=bool, default=False)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
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

        ia, ib = structured_to_unstructured(ss.prodstack, dtype=np.int16).T
        baselines = tel.feedpositions[ia] - tel.feedpositions[ib]

        for lbi, bi in ss.vis[:].enumerate(axis=1):
            # Select the baseline length to use
            baseline = baselines[bi]
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

            # Flag frequencies and times with zero weight. This works much better if the
            # incoming weight can be factorized
            f_samp = (ssw[:, lbi] > 0.0).sum(axis=1)
            f_mask = (f_samp == f_samp.max()).astype(np.float64)

            t_samp = (ssw[:, lbi] > 0.0).sum(axis=0)
            t_mask = (t_samp == t_samp.max()).astype(np.float64)

            try:
                NF = null_delay_filter(
                    freq,
                    delay_cut,
                    f_mask,
                    num_delay=number_cut,
                    window=self.window,
                )
            except la.LinAlgError as e:
                raise RuntimeError(
                    f"Failed to converge while processing baseline {bi}"
                ) from e

            ssv[:, lbi] = np.dot(NF, ssv[:, lbi])
            ssw[:, lbi] *= f_mask[:, np.newaxis] * t_mask[np.newaxis, :]

        return ss


# A specific subclass of a FreqContainer
FreqContainerType = typing.TypeVar("FreqContainerType", bound=containers.FreqContainer)


class DelayFilterBase(task.SingleTask):
    """Remove delays less than a given threshold.

    This is performed by projecting the data onto the null space that is orthogonal
    to any mode at low delays.

    Note that for this task to work best the zero entries in the weights dataset
    should factorize in frequency-time for each baseline. A mostly optimal masking
    can be generated using the `draco.analysis.flagging.MaskFreq` task.

    Attributes
    ----------
    delay_cut : float
        Delay value to filter at in seconds.
    window : bool
        Apply the window function to the data when applying the filter.
    axis : str
        The main axis to iterate over. The delay cut can be varied for each element
        of this axis. If not set, a suitable default is picked for the container
        type.
    dataset : str
        Apply the delay filter to this dataset.  If not set, a suitable default
        is picked for the container type.

    Notes
    -----
    The delay cut applied is `max(za_cut * baseline / c + extra_cut, delay_cut)`.
    """

    delay_cut = config.Property(proptype=float, default=0.1)
    window = config.Property(proptype=bool, default=False)
    axis = config.Property(proptype=str, default=None)
    dataset = config.Property(proptype=str, default=None)

    def setup(self, telescope: io.TelescopeConvertible):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

    def _delay_cut(self, ss: FreqContainerType, axis: str, ind: int) -> float:
        """Return the delay cut to use for this entry in microseconds.

        Parameters
        ----------
        ss
            The container we are processing.
        axis
            The axis we are looping over.
        ind : int
            The (global) index along that axis.

        Returns
        -------
        float
            The delay cut in microseconds.
        """
        return self.delay_cut

    def process(self, ss: FreqContainerType) -> FreqContainerType:
        """Filter out delays from a SiderealStream or TimeStream.

        Parameters
        ----------
        ss
            Data to filter.

        Returns
        -------
        ss_filt
            Filtered dataset.
        """
        if not isinstance(ss, containers.FreqContainer):
            raise TypeError(
                f"Can only process FreqContainer instances. Got {type(ss)}."
            )

        _default_axis = {
            containers.SiderealStream: "stack",
            containers.HybridVisMModes: "m",
            containers.RingMap: "el",
            containers.GridBeam: "theta",
        }

        _default_dataset = {
            containers.SiderealStream: "vis",
            containers.HybridVisMModes: "vis",
            containers.RingMap: "map",
            containers.GridBeam: "beam",
        }

        axis = self.axis

        if self.axis is None:
            for cls, ax in _default_axis.items():
                if isinstance(ss, cls):
                    axis = ax
                    break
            else:
                raise ValueError(f"No default axis know for {type(ss)} container.")

        dset = self.dataset

        if self.dataset is None:
            for cls, dataset in _default_dataset.items():
                if isinstance(ss, cls):
                    dset = dataset
                    break
            else:
                raise ValueError(f"No default dataset know for {type(ss)} container.")

        ss.redistribute(axis)

        freq = ss.freq[:]
        bandwidth = np.ptp(freq)

        # Get views of the relevant datasets, but make sure that the weights have the
        # same number of axes as the visibilities (inserting length-1 axes as needed)
        ssv = ss.datasets[dset][:].view(np.ndarray)
        ssw = match_axes(ss.datasets[dset], ss.weight).view(np.ndarray)

        dist_axis_pos = list(ss.datasets[dset].attrs["axis"]).index(axis)
        freq_axis_pos = list(ss.datasets[dset].attrs["axis"]).index("freq")

        # Once we have selected elements of dist_axis the location of freq_axis_pos may
        # be one lower
        sel_freq_axis_pos = (
            freq_axis_pos if freq_axis_pos < dist_axis_pos else freq_axis_pos - 1
        )

        for lbi, bi in ss.datasets[dset][:].enumerate(axis=dist_axis_pos):
            # Extract the part of the array that we are processing, and
            # transpose/reshape to make a 2D array with frequency as axis=0
            vis_local = _take_view(ssv, lbi, dist_axis_pos)
            vis_2D = _move_front(vis_local, sel_freq_axis_pos, vis_local.shape)

            weight_local = _take_view(ssw, lbi, dist_axis_pos)
            weight_2D = _move_front(weight_local, sel_freq_axis_pos, weight_local.shape)

            # In micro seconds
            delay_cut = self._delay_cut(ss, axis, bi)

            # Calculate the number of samples needed to construct the delay null space.
            # `4 * tau_max * bandwidth` is the amount recommended in the DAYENU paper
            # and seems to work well here
            number_cut = int(4.0 * bandwidth * delay_cut + 0.5)

            # Flag frequencies and times (or all other axes) with zero weight. This
            # works much better if the incoming weight can be factorized
            f_samp = (weight_2D > 0.0).sum(axis=1)
            f_mask = (f_samp == f_samp.max()).astype(np.float64)

            t_samp = (weight_2D > 0.0).sum(axis=0)
            t_mask = (t_samp == t_samp.max()).astype(np.float64)

            # This has occasionally failed to converge, catch this and output enough
            # info to debug after the fact
            try:
                NF = null_delay_filter(
                    freq,
                    delay_cut,
                    f_mask,
                    num_delay=number_cut,
                    window=self.window,
                )
            except la.LinAlgError as e:
                raise RuntimeError(
                    f"Failed to converge while processing baseline {bi}"
                ) from e

            vis_local[:] = _inv_move_front(
                np.dot(NF, vis_2D), sel_freq_axis_pos, vis_local.shape
            )
            weight_local[:] *= _inv_move_front(
                f_mask[:, np.newaxis] * t_mask[np.newaxis, :],
                sel_freq_axis_pos,
                weight_local.shape,
            )

        return ss


class DelaySpectrumEstimator(task.SingleTask, random.RandomTask):
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
    apply_window : bool, optional
        Whether to apply apodisation to frequency axis. Default: True.
    window : one of {'nuttall', 'blackman_nuttall', 'blackman_harris'}, optional
        Apodisation to perform on frequency axis. Default: 'nuttall'.
    complex_timedomain : bool, optional
        Whether to assume the original time samples that were channelized into a
        frequency spectrum were purely real (False) or complex (True). If True,
        `freq_zero`, `nfreq`, and `skip_nyquist` are ignored. Default: False.
    initial_amplitude : float, optional
        The Gibbs sampler will be initialized with a flat power spectrum with
        this amplitude. Default: 10.
    """

    nsamp = config.Property(proptype=int, default=20)
    freq_zero = config.Property(proptype=float, default=None)
    freq_spacing = config.Property(proptype=float, default=None)
    nfreq = config.Property(proptype=int, default=None)
    skip_nyquist = config.Property(proptype=bool, default=True)
    apply_window = config.Property(proptype=bool, default=True)
    window = config.enum(
        ["nuttall", "blackman_nuttall", "blackman_harris"], default="nuttall"
    )
    complex_timedomain = config.Property(proptype=bool, default=False)
    initial_amplitude = config.Property(proptype=float, default=10.0)

    def setup(self, telescope):
        """Set the telescope needed to generate Stokes I.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, ss):
        """Estimate the delay spectrum.

        Parameters
        ----------
        ss : SiderealStream or TimeStream
            Data container with visibilities to process

        Returns
        -------
        dspec : DelaySpectrum
            Delay spectrum of the input container Stokes I
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

        if self.complex_timedomain:
            self.nfreq = len(ss.freq)
            channel_ind = np.arange(self.nfreq)
            ndelay = self.nfreq
        else:
            channel_ind = (
                np.abs(ss.freq[:] - self.freq_zero) / self.freq_spacing
            ).astype(np.int64)
            if self.nfreq is None:
                self.nfreq = channel_ind[-1] + 1

                if self.skip_nyquist:
                    self.nfreq += 1

            # Assume each transformed frame was an even number of samples long
            ndelay = 2 * (self.nfreq - 1)

        # Compute delays corresponding to output delay power spectrum
        delays = np.fft.fftshift(np.fft.fftfreq(ndelay, d=self.freq_spacing))  # in us

        # Initialise the spectrum container
        delay_spec = containers.DelaySpectrum(
            baseline=baselines, delay=delays, attrs_from=ss
        )
        delay_spec.redistribute("baseline")
        delay_spec.spectrum[:] = 0.0

        initial_S = np.ones_like(delays) * self.initial_amplitude

        # Initialize the random number generator we'll use
        rng = self.rng

        # Iterate over all baselines and use the Gibbs sampler to estimate the spectrum
        for lbi, bi in delay_spec.spectrum[:].enumerate(axis=0):
            self.log.debug("Delay transforming baseline %i/%i", bi, len(baselines))

            # Get the local selections
            data = vis_I.local_array[lbi].T
            weight = vis_weight.local_array[lbi]

            # Mask out data with completely zero'd weights and generate time
            # averaged weights
            weight_cut = (
                1e-4 * weight.mean()
            )  # Use approx threshold to ignore small weights
            data = data * (weight.T > weight_cut)
            weight = np.mean(weight, axis=1)

            if (data == 0.0).all():
                continue

            # If there are no non-zero weighted entries skip
            non_zero = weight > 0
            if not non_zero.any():
                continue

            # Remove any frequency channel which is entirely zero, this is just to
            # reduce the computational cost, it should make no difference to the result
            data = data[:, non_zero]
            weight = weight[non_zero]
            non_zero_channel = channel_ind[non_zero]

            spec = delay_spectrum_gibbs(
                data,
                ndelay,
                weight,
                initial_S,
                window=self.window if self.apply_window else None,
                fsel=non_zero_channel,
                niter=self.nsamp,
                rng=rng,
                complex_timedomain=self.complex_timedomain,
            )

            # Take an average over the last half of the delay spectrum samples
            # (presuming that removes the burn-in)
            spec_av = np.median(spec[-(self.nsamp // 2) :], axis=0)
            delay_spec.spectrum[bi] = np.fft.fftshift(spec_av)

        return delay_spec


class DelaySpectrumEstimatorBase(task.SingleTask, random.RandomTask):
    """Calculate the delay spectrum of any container with a frequency axis.

    The spectrum is calculated by Gibbs sampling. The spectrum returned is the median
    of the final half of the samples calculated.

    The delay spectrum output is indexed by a `baseline` axis. This axis is the
    composite axis of all the axes in the container except the frequency axis or the
    `average_axis`. These constituent axes are included in the index map, and their
    order is given by the `baseline_axes` attribute.

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
    dataset : str
        Calculate the delay spectrum of this dataset (e.g., "vis", "map", "beam").
    average_axis : str
        Name of the axis to take the average over.
    apply_window : bool, optional
        Whether to apply apodisation to frequency axis. Default: True.
    window : one of {'nuttall', 'blackman_nuttall', 'blackman_harris', optional
        Apodisation to perform on frequency axis. Default: 'nuttall'.
    complex_timedomain : bool, optional
        Whether to assume the original time samples that were channelized into a
        frequency spectrum were purely real (False) or complex (True). If True,
        `freq_zero`, `nfreq`, and `skip_nyquist` are ignored. Default: False.
    initial_amplitude : float, optional
        The Gibbs sampler will be initialized with a flat power spectrum with
        this amplitude. Default: 10.
    """

    nsamp = config.Property(proptype=int, default=20)
    freq_zero = config.Property(proptype=float, default=None)
    freq_spacing = config.Property(proptype=float, default=None)
    nfreq = config.Property(proptype=int, default=None)
    skip_nyquist = config.Property(proptype=bool, default=True)
    apply_window = config.Property(proptype=bool, default=True)
    window = config.enum(
        ["nuttall", "blackman_nuttall", "blackman_harris"], default="nuttall"
    )
    dataset = config.Property(proptype=str, default="vis")
    average_axis = config.Property(proptype=str)
    complex_timedomain = config.Property(proptype=bool, default=False)
    initial_amplitude = config.Property(proptype=float, default=10.0)

    def setup(self, telescope: io.TelescopeConvertible):
        """Set the telescope needed to generate Stokes I.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, ss: FreqContainerType) -> containers.DelaySpectrum:
        """Estimate the delay spectrum.

        Parameters
        ----------
        ss
            Data to transform. Must have a frequency axis and one other axis to
            average over.

        Returns
        -------
        dspec : DelaySpectrum
        """
        ss.redistribute("freq")

        if self.dataset not in ss.datasets:
            raise ValueError(
                f"Specified dataset to delay transform ({self.dataset}) not in "
                f"container of type {type(ss)}."
            )

        if (
            self.average_axis not in ss.axes
            or self.average_axis not in ss.datasets[self.dataset].attrs["axis"]
        ):
            raise ValueError(
                f"Specified axis to average over ({self.average_axis}) not in "
                f"container of type {type(ss)}."
            )

        # ==== Figure out the frequency structure and delay values ====
        if self.freq_zero is None:
            self.freq_zero = ss.freq[0]

        if self.freq_spacing is None:
            self.freq_spacing = np.abs(np.diff(ss.freq[:])).min()

        if self.complex_timedomain:
            self.nfreq = len(ss.freq)
            channel_ind = np.arange(self.nfreq)
            ndelay = self.nfreq
        else:
            channel_ind = (
                np.abs(ss.freq[:] - self.freq_zero) / self.freq_spacing
            ).astype(np.int64)
            if self.nfreq is None:
                self.nfreq = channel_ind[-1] + 1

                if self.skip_nyquist:
                    self.nfreq += 1

            # Assume each transformed frame was an even number of samples long
            ndelay = 2 * (self.nfreq - 1)

        # Compute delays corresponding to output delay power spectrum
        delays = np.fft.fftshift(np.fft.fftfreq(ndelay, d=self.freq_spacing))  # in us

        # Find the relevant axis positions
        data_axes = ss.datasets[self.dataset].attrs["axis"]
        freq_axis_pos = list(data_axes).index("freq")
        average_axis_pos = list(data_axes).index(self.average_axis)

        # Create a view of the dataset with the relevant axes at the back,
        # and all other axes compressed
        data_view = np.moveaxis(
            ss.datasets[self.dataset][:].local_array,
            [average_axis_pos, freq_axis_pos],
            [-2, -1],
        )
        data_view = data_view.reshape(-1, data_view.shape[-2], data_view.shape[-1])
        data_view = mpiarray.MPIArray.wrap(data_view, axis=2, comm=ss.comm)
        nbase = int(np.prod(data_view.shape[:-2]))
        data_view = data_view.redistribute(axis=0)

        # ... do the same for the weights, but we also need to make the weights full
        # size
        weight_full = np.zeros(
            ss.datasets[self.dataset][:].shape, dtype=ss.weight.dtype
        )
        weight_full[:] = match_axes(ss.datasets[self.dataset], ss.weight)
        weight_view = np.moveaxis(
            weight_full, [average_axis_pos, freq_axis_pos], [-2, -1]
        )
        weight_view = weight_view.reshape(
            -1, weight_view.shape[-2], weight_view.shape[-1]
        )
        weight_view = mpiarray.MPIArray.wrap(weight_view, axis=2, comm=ss.comm)
        weight_view = weight_view.redistribute(axis=0)

        # Use the "baselines" axis to generically represent all the other axes

        # Initialise the spectrum container
        delay_spec = containers.DelaySpectrum(
            baseline=nbase, delay=delays, attrs_from=ss
        )
        delay_spec.redistribute("baseline")
        delay_spec.spectrum[:] = 0.0
        bl_axes = [da for da in data_axes if da not in [self.average_axis, "freq"]]

        # Copy the index maps for all the flattened axes into the output container, and
        # write out their order into an attribute so we can reconstruct this easily
        # when loading in the spectrum
        for ax in bl_axes:
            delay_spec.create_index_map(ax, ss.index_map[ax])
        delay_spec.attrs["baseline_axes"] = bl_axes

        initial_S = np.ones_like(delays) * self.initial_amplitude

        # Initialize the random number generator we'll use
        rng = self.rng

        # Iterate over all baselines and use the Gibbs sampler to estimate the spectrum
        for lbi, bi in delay_spec.spectrum[:].enumerate(axis=0):
            self.log.debug(f"Delay transforming baseline {bi}/{nbase}")

            # Get the local selections
            data = data_view.local_array[lbi]
            weight = weight_view.local_array[lbi]

            # Mask out data with completely zero'd weights and generate time
            # averaged weights
            weight_cut = (
                1e-4 * weight.mean()
            )  # Use approx threshold to ignore small weights
            data = data * (weight > weight_cut)
            weight = np.mean(weight, axis=0)

            if (data == 0.0).all():
                continue

            # If there are no non-zero weighted entries skip
            non_zero = weight > 0
            if not non_zero.any():
                continue

            # Remove any frequency channel which is entirely zero, this is just to
            # reduce the computational cost, it should make no difference to the result
            data = data[:, non_zero]
            weight = weight[non_zero]
            non_zero_channel = channel_ind[non_zero]

            spec = delay_spectrum_gibbs(
                data,
                ndelay,
                weight,
                initial_S,
                window=self.window if self.apply_window else None,
                fsel=non_zero_channel,
                niter=self.nsamp,
                rng=rng,
                complex_timedomain=self.complex_timedomain,
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
    ubase : np.ndarray[nbase, 2]
        Baseline vectors corresponding to output.
    """
    # Construct a complex number representing each baseline (used for determining
    # unique baselines).
    # NOTE: due to floating point precision, some baselines don't get matched as having
    # the same lengths. To get around this, round all separations to 0.1 mm precision
    bl_round = np.around(tel.baselines[:, 0] + 1.0j * tel.baselines[:, 1], 4)

    # ==== Unpack into Stokes I
    ubase, uinv, ucount = np.unique(bl_round, return_inverse=True, return_counts=True)
    ubase = ubase.astype(np.complex128, copy=False).view(np.float64).reshape(-1, 2)
    nbase = ubase.shape[0]

    vis_shape = (nbase, sstream.vis.global_shape[0], sstream.vis.global_shape[2])
    vis_I = mpiarray.zeros(vis_shape, dtype=sstream.vis.dtype, axis=1)
    vis_weight = mpiarray.zeros(vis_shape, dtype=sstream.weight.dtype, axis=1)

    # Iterate over products to construct the Stokes I vis
    # TODO: this should be updated when driftscan gains a concept of polarisation
    ssv = sstream.vis[:]
    ssw = sstream.weight[:]

    # Cache beamclass as it's regenerated every call
    beamclass = tel.beamclass[:]
    for ii, ui in enumerate(uinv):
        # Skip if not all polarisations were included
        if ucount[ui] < 4:
            continue

        fi, fj = tel.uniquepairs[ii]
        bi, bj = beamclass[fi], beamclass[fj]

        upi = tel.feedmap[fi, fj]

        if upi == -1:
            continue

        if bi == bj:
            vis_I[ui] += ssv[:, ii]
            vis_weight[ui] += ssw[:, ii]

    vis_I = vis_I.redistribute(axis=0)
    vis_weight = vis_weight.redistribute(axis=0)

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
    Fr : np.ndarray
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
    Fr : np.ndarray
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


def fourier_matrix_c2c(N, fsel=None):
    """Generate a Fourier matrix to represent a complex to complex FFT.

    These Fourier conventions match `numpy.fft.fft()`.

    Parameters
    ----------
    N : integer
        Length of timestream that we are transforming to.
    fsel : array_like, optional
        Indices of the frequency channels to include in the transformation
        matrix. By default, assume all channels.

    Returns
    -------
    F : np.ndarray
        An array performing the Fourier transform from a complex time series to
        frequencies, with both input and output packed as alternating real and
        imaginary elements.
    """
    if fsel is None:
        fa = np.arange(N)
    else:
        fa = np.array(fsel)

    fa = fa[:, np.newaxis]
    ta = np.arange(N)[np.newaxis, :]

    F = np.zeros((2 * fa.shape[0], 2 * N), dtype=np.float64)

    arg = 2 * np.pi * ta * fa / N
    F[0::2, 0::2] = np.cos(arg)
    F[0::2, 1::2] = np.sin(arg)
    F[1::2, 0::2] = -np.sin(arg)
    F[1::2, 1::2] = np.cos(arg)

    return F


def _complex_to_alternating_real(array):
    """View complex numbers as an array with alternating real and imaginary components.

    Parameters
    ----------
    array : array_like
        Input array of complex numbers.

    Returns
    -------
    out : array_like
        Output array of alternating real and imaginary components. These components are
        expanded along the last axis, such that if `array` has `N` complex elements in
        its last axis, `out` will have `2N` real elements.
    """
    return array.astype(np.complex128, order="C").view(np.float64)


def _alternating_real_to_complex(array):
    """View real numbers as complex, interpreted as alternating real and imag. components.

    Parameters
    ----------
    array : array_like
        Input array of real numbers. Last axis must have even number of elements.

    Returns
    -------
    out : array_like
        Output array of complex numbers, derived from compressing the last axis (if
        `array` has `N` real elements in the last axis, `out` will have `N/2` complex
        elements).
    """
    return array.astype(np.float64, order="C").view(np.complex128)


def delay_spectrum_gibbs(
    data,
    N,
    Ni,
    initial_S,
    window="nuttall",
    fsel=None,
    niter=20,
    rng=None,
    complex_timedomain=False,
):
    """Estimate the delay power spectrum by Gibbs sampling.

    This routine estimates the spectrum at the `N` delay samples conjugate to
    an input frequency spectrum with ``N/2 + 1`` channels (if the delay spectrum is
    assumed real) or `N` channels (if the delay spectrum is assumed complex).
    A subset of these channels can be specified using the `fsel` argument.

    Parameters
    ----------
    data : np.ndarray[:, freq]
        Data to estimate the delay spectrum of.
    N : int
        The length of the output delay spectrum. There are assumed to be `N/2 + 1`
        total frequency channels if assuming a real delay spectrum, or `N` channels
        for a complex delay spectrum.
    Ni : np.ndarray[freq]
        Inverse noise variance.
    initial_S : np.ndarray[delay]
        The initial delay power spectrum guess.
    window : one of {'nuttall', 'blackman_nuttall', 'blackman_harris', None}, optional
        Apply an apodisation function. Default: 'nuttall'.
    fsel : np.ndarray[freq], optional
        Indices of channels that we have data at. By default assume all channels.
    niter : int, optional
        Number of Gibbs samples to generate.
    rng : np.random.Generator, optional
        A generator to use to produce the random samples.
    complex_timedomain : bool, optional
        If True, assume input data arose from a complex timestream. If False, assume
        input data arose from a real timestream, such that the first and last frequency
        channels have purely real values. Default: False.

    Returns
    -------
    spec : list
        List of spectrum samples.
    """
    # Get reference to RNG
    if rng is None:
        rng = random.default_rng()

    spec = []

    total_freq = N if complex_timedomain else N // 2 + 1

    if fsel is None:
        fsel = np.arange(total_freq)

    # Construct the Fourier matrix
    F = (
        fourier_matrix_c2c(N, fsel)
        if complex_timedomain
        else fourier_matrix_r2c(N, fsel)
    )

    # Construct a view of the data with alternating real and imaginary parts
    data = _complex_to_alternating_real(data).T.copy()

    # Window the frequency data
    if window is not None:
        # Construct the window function
        x = fsel * 1.0 / total_freq
        w = window_generalised(x, window=window)
        w = np.repeat(w, 2)

        # Apply to the projection matrix and the data
        F *= w[:, np.newaxis]
        data *= w[:, np.newaxis]

    if complex_timedomain:
        is_real_freq = np.zeros_like(fsel).astype(bool)
    else:
        is_real_freq = (fsel == 0) | (fsel == N // 2)

    # Construct the Noise inverse array for the real and imaginary parts of the
    # frequency spectrum (taking into account that the zero and Nyquist frequencies are
    # strictly real if the delay spectrum is assumed to be real)
    Ni_r = np.zeros(2 * Ni.shape[0])
    Ni_r[0::2] = np.where(is_real_freq, Ni, Ni / 2**0.5)
    Ni_r[1::2] = np.where(is_real_freq, 0.0, Ni / 2**0.5)

    # Create the transpose of the Fourier matrix weighted by the noise
    # (this is used multiple times)
    FTNih = F.T * Ni_r[np.newaxis, :] ** 0.5
    FTNiF = np.dot(FTNih, FTNih.T)

    # Pre-whiten the data to save doing it repeatedly
    data = data * Ni_r[:, np.newaxis] ** 0.5

    # Set the initial guess for the delay power spectrum.
    S_samp = initial_S

    def _draw_signal_sample_f(S):
        # Draw a random sample of the signal (delay spectrum) assuming a Gaussian model
        # with a given delay power spectrum `S`. Do this using the perturbed Wiener
        # filter approach

        # This method is fastest if the number of frequencies is larger than the number
        # of delays we are solving for. Typically this isn't true, so we probably want
        # `_draw_signal_sample_t`

        # Construct the Wiener covariance
        if complex_timedomain:
            # If delay spectrum is complex, extend S to correspond to the individual
            # real and imaginary components of the delay spectrum, each of which have
            # power spectrum equal to 0.5 times the power spectrum of the complex
            # delay spectrum, if the statistics are circularly symmetric
            S = 0.5 * np.repeat(S, 2)
        Si = 1.0 / S
        Ci = np.diag(Si) + FTNiF

        # Draw random vectors that form the perturbations
        if complex_timedomain:
            # If delay spectrum is complex, draw for real and imaginary components
            # separately
            w1 = rng.standard_normal((2 * N, data.shape[1]))
        else:
            w1 = rng.standard_normal((N, data.shape[1]))
        w2 = rng.standard_normal(data.shape)

        # Construct the random signal sample by forming a perturbed vector and
        # then doing a matrix solve
        y = np.dot(FTNih, data + w2) + Si[:, np.newaxis] ** 0.5 * w1

        return la.solve(Ci, y, sym_pos=True)

    def _draw_signal_sample_t(S):
        # This method is fastest if the number of delays is larger than the number of
        # frequencies. This is usually the regime we are in.

        # Construct various dependent matrices
        if complex_timedomain:
            # If delay spectrum is complex, extend S to correspond to the individual
            # real and imaginary components of the delay spectrum, each of which have
            # power spectrum equal to 0.5 times the power spectrum of the complex
            # delay spectrum, if the statistics are circularly symmetric
            S = 0.5 * np.repeat(S, 2)
        Sh = S**0.5
        Rt = Sh[:, np.newaxis] * FTNih
        R = Rt.T.conj()

        # Draw random vectors that form the perturbations
        if complex_timedomain:
            # If delay spectrum is complex, draw for real and imaginary components
            # separately
            w1 = rng.standard_normal((2 * N, data.shape[1]))
        else:
            w1 = rng.standard_normal((N, data.shape[1]))
        w2 = rng.standard_normal(data.shape)

        # Perform the solve step (rather than explicitly using the inverse)
        y = data + w2 - np.dot(R, w1)
        Ci = np.identity(len(Ni_r)) + np.dot(R, Rt)
        x = la.solve(Ci, y, sym_pos=True)

        s = Sh[:, np.newaxis] * (np.dot(Rt, x) + w1)
        return s

    def _draw_ps_sample(d):
        # Draw a random delay power spectrum sample assuming the signal is Gaussian and
        # we have a flat prior on the power spectrum.
        # This means drawing from a inverse chi^2.

        if complex_timedomain:
            # If delay spectrum is complex, combine real and imaginary components
            # stored in d, such that variance below is variance of complex spectrum
            d = d[0::2] + 1.0j * d[1::2]
        S_hat = d.var(axis=1)

        df = d.shape[1]
        chi2 = rng.chisquare(df, size=d.shape[0])

        S_samp = S_hat * df / chi2

        return S_samp

    # Select the method to use for the signal sample based on how many frequencies
    # versus delays there are
    _draw_signal_sample = (
        _draw_signal_sample_f if (len(fsel) > 0.25 * N) else _draw_signal_sample_t
    )

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
    F = mask[:, np.newaxis] * np.exp(
        2.0j * np.pi * delay[np.newaxis, :] * freq[:, np.newaxis]
    )

    if window:
        F *= w[:, np.newaxis]

    # Use an SVD to figure out the set of significant modes spanning the delays
    # we are wanting to get rid of.
    # NOTE: we've experienced some convergence failures in here which ultimately seem
    # to be the fault of MKL (see https://github.com/scipy/scipy/issues/10032 and links
    # therein). This seems to be limited to the `gesdd` LAPACK routine, so we can get
    # around it by switching to `gesvd`.
    u, sig, vh = la.svd(F, lapack_driver="gesvd")
    nmodes = np.sum(sig > tol * sig.max())
    p = u[:, :nmodes]

    # Construct a projection matrix for the filter
    proj = np.identity(len(freq)) - np.dot(p, p.T.conj())
    proj *= mask[np.newaxis, :]

    if window:
        proj *= w[np.newaxis, :]

    return proj


def match_axes(dset1, dset2):
    """Make sure that dset2 has the same set of axes as dset1.

    Sometimes the weights are missing axes (usually where the entries would all be
    the same), we need to map these into one another and expand the weights to the
    same size as the visibilities. This assumes that the vis/weight axes are in the
    same order when present

    Parameters
    ----------
    dset1
        The dataset with more axes.
    dset2
        The dataset with a subset of axes. For the moment these are assumed to be in
        the same order.

    Returns
    -------
    dset2_view
        A view of dset2 with length-1 axes inserted to match the axes missing from
        dset1.
    """
    axes1 = dset1.attrs["axis"]
    axes2 = dset2.attrs["axis"]
    bcast_slice = tuple(slice(None) if ax in axes2 else np.newaxis for ax in axes1)

    return dset2[:][bcast_slice]


def _move_front(arr: np.ndarray, axis: int, shape: tuple) -> np.ndarray:
    # Move the specified axis to the front and flatten to give a 2D array
    new_arr = np.moveaxis(arr, axis, 0)
    return new_arr.reshape(shape[axis], -1)


def _inv_move_front(arr: np.ndarray, axis: int, shape: tuple) -> np.ndarray:
    # Move the first axis back to it's original position and return the original shape,
    # i.e. reverse the above operation
    rshape = (shape[axis],) + shape[:axis] + shape[(axis + 1) :]
    new_arr = arr.reshape(rshape)
    new_arr = np.moveaxis(new_arr, 0, axis)
    return new_arr.reshape(shape)


def _take_view(arr: np.ndarray, ind: int, axis: int) -> np.ndarray:
    # Like np.take but returns a view (instead of a copy), but only supports a scalar
    # index
    sl = (slice(None),) * axis
    return arr[sl + (ind,)]

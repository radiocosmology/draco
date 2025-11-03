"""Miscellaneous transformations to do on data.

This includes grouping frequencies and products to performing the m-mode transform.
"""

from typing import overload

import numpy as np
import scipy.linalg as la
from caput import config, fftw, mpiarray, pipeline
from caput.tools import invert_no_zero
from numpy.lib.recfunctions import structured_to_unstructured

from ..core import containers, io, task
from ..util import regrid, tools


class FrequencyRebin(task.SingleTask):
    """Rebin neighbouring frequency channels.

    Parameters
    ----------
    channel_bin : int
        Number of channels to in together.
    """

    channel_bin = config.Property(proptype=int, default=1)

    def process(self, ss):
        """Take the input dataset and rebin the frequencies.

        Parameters
        ----------
        ss : containers.SiderealStream or containers.TimeStream
            Input data to rebin. Can also be an `andata.CorrData` instance,
            however the output will be a `containers.TimeStream` instance.

        Returns
        -------
        sb : containers.SiderealStream or containers.TimeStream
            Rebinned data. Type should match the input.
        """
        if "freq" not in ss.index_map:
            raise RuntimeError("Data does not have a frequency axis.")

        if len(ss.freq) % self.channel_bin != 0:
            raise RuntimeError("Binning must exactly divide the number of channels.")

        # Get all frequencies onto same node
        ss.redistribute(["time", "ra"])

        # Calculate the new frequency centres and widths
        fc = ss.index_map["freq"]["centre"].reshape(-1, self.channel_bin).mean(axis=-1)
        fw = ss.index_map["freq"]["width"].reshape(-1, self.channel_bin).sum(axis=-1)

        freq_map = np.empty(fc.shape[0], dtype=ss.index_map["freq"].dtype)
        freq_map["centre"] = fc
        freq_map["width"] = fw

        # Create new container for rebinned stream
        sb = containers.empty_like(ss, freq=freq_map)

        # Get all frequencies onto same node
        sb.redistribute(["time", "ra"])

        # Rebin the arrays, do this with a loop to save memory
        for fi in range(len(ss.freq)):
            # Calculate rebinned index
            ri = fi // self.channel_bin

            sb.vis[ri] += ss.vis[fi] * ss.weight[fi]

            if "gain" in ss.datasets:
                sb.gain[ri] += (
                    ss.gain[fi] / self.channel_bin
                )  # Don't do weighted average for the moment

            sb.weight[ri] += ss.weight[fi]

            # If we are on the final sub-channel then divide the arrays through
            if (fi + 1) % self.channel_bin == 0:
                sb.vis[ri] *= tools.invert_no_zero(sb.weight[ri])

        sb.redistribute("freq")

        return sb


class TelescopeStreamMixIn:
    """A mixin providing functionality for creating telescope-defined sidereal streams.

    This mixin is designed to be used with pipeline tasks that require certain
    index maps in order to create SiderealStream containers compatible with the
    baseline configuration provided in a telescope instance.
    """

    def setup(self, tel):
        """Set up the telescope instance and precompute index maps.

        Parameters
        ----------
        tel : TransitTelescope
            The telescope instance to use to compute the prod, stack,
            and reverse_stack index maps.
        """
        self.telescope = io.get_telescope(tel)

        # Precalculate the stack properties
        self.bt_stack = np.array(
            [
                (
                    (tools.cmap(upp[0], upp[1], self.telescope.nfeed), 0)
                    if upp[0] <= upp[1]
                    else (tools.cmap(upp[1], upp[0], self.telescope.nfeed), 1)
                )
                for upp in self.telescope.uniquepairs
            ],
            dtype=[("prod", "<u4"), ("conjugate", "u1")],
        )

        # Construct the equivalent prod and stack index_map for the telescope instance
        triu = np.triu_indices(self.telescope.nfeed)
        dt_prod = np.dtype([("input_a", "<u2"), ("input_b", "<u2")])
        self.bt_prod = np.array(triu).astype("<u2").T.copy().view(dt_prod).reshape(-1)

        # Construct the equivalent reverse_map stack for the telescope instance.
        # Note that we identify invalid products here using an index that is the
        # size of the stack axis.
        feedmask = self.telescope.feedmask[triu]

        self.bt_rev = np.empty(
            feedmask.size, dtype=[("stack", "<u4"), ("conjugate", "u1")]
        )
        self.bt_rev["stack"] = np.where(
            feedmask, self.telescope.feedmap[triu], self.telescope.npairs
        )
        self.bt_rev["conjugate"] = np.where(feedmask, self.telescope.feedconj[triu], 0)


class CollateProducts(TelescopeStreamMixIn, task.SingleTask):
    """Extract and order the correlation products for map-making.

    The task will take a sidereal task and format the products that are needed
    or the map-making. It uses a BeamTransfer instance to figure out what these
    products are, and how they should be ordered. It similarly selects only the
    required frequencies.

    It is important to note that while the input
    :class:`~containers.SiderealStream` can contain more feeds and frequencies
    than are contained in the BeamTransfers, the converse is not true. That is,
    all the frequencies and feeds that are in the BeamTransfers must be found in
    the timestream object.

    Parameters
    ----------
    weight : string ('natural', 'uniform', or 'inverse_variance')
        How to weight the redundant baselines when stacking:
            'natural' - each baseline weighted by its redundancy (default)
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute
    """

    weight = config.enum(["natural", "uniform", "inverse_variance"], default="natural")

    @overload
    def process(self, ss: containers.SiderealStream) -> containers.SiderealStream: ...

    @overload
    def process(self, ss: containers.TimeStream) -> containers.TimeStream: ...

    def process(self, ss):
        """Select and reorder the products.

        Parameters
        ----------
        ss
            Data with products

        Returns
        -------
        sp
            Dataset containing only the required products.
        """
        # For each input in the file, find the corresponding index in the telescope instance
        input_ind = tools.find_inputs(
            self.telescope.input_index, ss.input, require_match=False
        )

        # Figure out the reverse mapping (i.e., for each input in the telescope instance,
        # find the corresponding index in file)
        rev_input_ind = tools.find_inputs(
            ss.input, self.telescope.input_index, require_match=True
        )

        # Figure out mapping between the frequencies
        freq_ind = tools.find_keys(
            ss.freq[:], self.telescope.frequencies, require_match=True
        )

        bt_freq = ss.index_map["freq"][freq_ind]
        # Determine the input product map and conjugation.
        # If the input timestream is already stacked, then attempt to redefine
        # its representative products so that they contain only feeds that exist
        # and are not masked in the telescope instance.
        if ss.is_stacked:
            stack_new, stack_flag = tools.redefine_stack_index_map(
                self.telescope, ss.input, ss.prod, ss.stack, ss.reverse_map["stack"]
            )

            if not np.all(stack_flag):
                self.log.warning(
                    f"There are {np.sum(~stack_flag):0.0f} stacked baselines "
                    "that are masked in the telescope instance."
                )

            ss_prod = ss.prod[stack_new["prod"]]
            ss_conj = stack_new["conjugate"]

        else:
            ss_prod = ss.prod
            ss_conj = np.zeros(ss_prod.size, dtype=bool)

        # Create output container
        sp = ss.__class__(
            freq=bt_freq,
            input=self.telescope.input_index,
            prod=self.bt_prod,
            stack=self.bt_stack,
            reverse_map_stack=self.bt_rev,
            copy_from=ss,
            distributed=True,
            comm=ss.comm,
        )

        # Check if frequencies are already ordered
        no_redistribute = freq_ind == list(range(len(ss.freq[:])))

        # If frequencies are mapped across ranks, we have to redistribute so all
        # frequencies and products are on each rank
        raxis = "freq" if no_redistribute else ["ra", "time"]
        self.log.debug(f"Distributing across '{raxis}' axis")
        ss.redistribute(raxis)
        sp.redistribute(raxis)

        # Initialize datasets in output container
        sp.vis[:] = 0.0
        sp.weight[:] = 0.0
        sp.input_flags[:] = ss.input_flags[rev_input_ind, :]

        # Infer number of products that went into each stack
        if self.weight != "inverse_variance":
            ssi = ss.input_flags[:]
            ssp = ss.index_map["prod"][:]
            sss = ss.reverse_map["stack"]["stack"][:]
            nstack = ss.vis.shape[1]

            nprod_in_stack = tools.calculate_redundancy(ssi, ssp, sss, nstack)

            if self.weight == "uniform":
                nprod_in_stack = (nprod_in_stack > 0).astype(np.float32)

        # Create counter to increment during the stacking.
        # This will be used to normalize at the end.
        counter = np.zeros_like(sp.weight[:])

        # Dereference the global slices, there's a hidden MPI call in the [:] operation.
        spv = sp.vis[:]
        ssv = ss.vis[:]
        spw = sp.weight[:]
        ssw = ss.weight[:]

        # Get the local frequency and time slice/mapping
        freq_ind = slice(None) if no_redistribute else freq_ind
        time_ind = slice(None) if no_redistribute else ssv.local_bounds

        # Iterate over products (stacked) in the sidereal stream
        for ss_pi, ((ii, ij), conj) in enumerate(zip(ss_prod, ss_conj)):
            # Map the feed indices into ones for the Telescope class
            bi, bj = input_ind[ii], input_ind[ij]

            # If either feed is not in the telescope class, skip it.
            if bi is None or bj is None:
                continue

            sp_pi = self.telescope.feedmap[bi, bj]
            feedconj = self.telescope.feedconj[bi, bj]

            # Skip if product index is not valid
            if sp_pi < 0:
                continue

            # Generate weight
            if self.weight == "inverse_variance":
                wss = ssw.local_array[freq_ind, ss_pi]

            else:
                wss = (ssw.local_array[freq_ind, ss_pi] > 0.0).astype(np.float32)
                wss[:] *= nprod_in_stack[np.newaxis, ss_pi, time_ind]

            # Accumulate visibilities, conjugating if required
            if feedconj == conj:
                spv.local_array[:, sp_pi] += wss * ssv.local_array[freq_ind, ss_pi]
            else:
                spv.local_array[:, sp_pi] += (
                    wss * ssv.local_array[freq_ind, ss_pi].conj()
                )

            # Accumulate variances in quadrature.  Save in the weight dataset.
            spw.local_array[:, sp_pi] += wss**2 * tools.invert_no_zero(
                ssw.local_array[freq_ind, ss_pi]
            )

            # Increment counter
            counter.local_array[:, sp_pi] += wss

        # Divide through by counter to get properly weighted visibility average
        sp.vis[:] *= tools.invert_no_zero(counter)
        sp.weight[:] = counter**2 * tools.invert_no_zero(sp.weight[:])

        # Copy over any additional datasets that need to be frequency filtered
        containers.copy_datasets_filter(
            ss, sp, "freq", freq_ind, ["input", "prod", "stack"]
        )

        # Switch back to frequency distribution. This will have minimal
        # cost if we are already distributed in frequency
        ss.redistribute("freq")
        sp.redistribute("freq")

        return sp


class SelectFreq(task.SingleTask):
    """Select a subset of frequencies from a container.

    Attributes
    ----------
    freq_physical : list
        List of physical frequencies in MHz.
        Given first priority.
    channel_range : list
        Range of frequency channel indices, either
        [start, stop, step], [start, stop], or [stop]
        is acceptable.  Given second priority.
    channel_index : list
        List of frequency channel indices.
        Given third priority.
    freq_physical_range : list
        Range of physical frequencies to include given as (low_freq, high_freq).
        Given fourth priority.
    """

    freq_physical = config.Property(proptype=list, default=[])
    freq_physical_range = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    def process(self, data):
        """Selet a subset of the frequencies.

        Parameters
        ----------
        data : containers.ContainerBase
            A data container with a frequency axis.

        Returns
        -------
        newdata : containers.ContainerBase
            New container with trimmed frequencies.
        """
        # Set up frequency selection.
        freq_map = data.index_map["freq"]

        # Construct the frequency channel selection
        if self.freq_physical:
            newindex = sorted(
                {
                    np.argmin(np.abs(freq_map["centre"] - freq))
                    for freq in self.freq_physical
                }
            )

        elif self.channel_range and (len(self.channel_range) <= 3):
            newindex = slice(*self.channel_range)

        elif self.channel_index:
            newindex = self.channel_index

        elif self.freq_physical_range:
            low, high = sorted(self.freq_physical_range)
            newindex = np.where(
                (freq_map["centre"] >= low) & (freq_map["centre"] < high)
            )[0]

        else:
            raise ValueError(
                "Must specify either freq_physical, channel_range, or channel_index."
            )

        freq_map = freq_map[newindex]

        # Destribute input container over ra or time.
        data.redistribute(["ra", "time", "pixel"])

        # Create new container with subset of frequencies.
        newdata = containers.empty_like(data, freq=freq_map)

        # Make sure all datasets are initialised
        for name in data.datasets.keys():
            if name not in newdata.datasets:
                newdata.add_dataset(name)

        # Redistribute new container over ra or time.
        newdata.redistribute(["ra", "time", "pixel"])

        # Copy over datasets. If the dataset has a frequency axis,
        # then we only copy over the subset.
        if isinstance(data, containers.ContainerBase):
            containers.copy_datasets_filter(
                data, newdata, "freq", newindex, copy_without_selection=True
            )
        else:
            newdata.vis[:] = data.vis[newindex]
            newdata.weight[:] = data.weight[newindex]
            newdata.gain[:] = data.gain[newindex]

            newdata.input_flags[:] = data.input_flags[:]

        # Switch back to frequency distribution
        data.redistribute("freq")
        newdata.redistribute("freq")

        return newdata


class GenerateSubBands(SelectFreq):
    """Generate multiple sub-bands from an input container.

    Attributes
    ----------
    sub_band_spec : dict
        Dictionary of the format {"band_a": {"channel_range": [0, 64]}, ...}
        where each entry is a separate sub-band with the key providing the tag
        that will be used to describe the sub-band and the value providing a
        dictionary that can contain any of the config properties used by the
        SelectFreq task to downselect along the frequency axis to obtain the
        sub-band from the input container.
    """

    sub_band_spec = config.Property(proptype=dict)

    def setup(self, data):
        """Cache the data product that will be sub-divided along the frequency axis.

        Parameters
        ----------
        data : container
            Any container with a freq axis.
        """
        self.default_parameters = {
            key: val.default
            for key, val in SelectFreq.__dict__.items()
            if isinstance(val, config.Property)
        }

        self.data = data
        self.base_tag = self.data.attrs.get("tag", None)

        self.sub_bands = list(self.sub_band_spec.keys())[::-1]

    def process(self):
        """Select the next sub-band from the container that was provided on setup.

        Returns
        -------
        sub : container
            Same type of container as was provided on setup,
            downselected along the frequency axis.
        """
        if len(self.sub_bands) == 0:
            raise pipeline.PipelineStopIteration

        tag = self.sub_bands.pop()
        self._set_freq_selection(**self.sub_band_spec[tag])

        if self.base_tag is not None:
            self.data.attrs["tag"] = "_".join([self.base_tag, tag])
        else:
            self.data.attrs["tag"] = tag

        return super().process(self.data)

    def _set_freq_selection(self, **kwargs):
        """Set properties of the SelectFreq base class to choose the next sub-band."""
        for key, default in self.default_parameters.items():
            value = kwargs[key] if key in kwargs else default
            setattr(self, key, value)


class ElevationDependentHybridVisWeight(task.SingleTask):
    """Add elevation dependence to hybrid visibility weights."""

    def process(self, data: containers.HybridVisStream):
        """Remove the weights dataset and broadcast to elevation weights.

        Parameters
        ----------
        data
            Hybrid visibilities with elevation-independent weights.

        Returns
        -------
        data
            Input container with different weights dataset
        """
        data.redistribute("freq")

        # if elevation-dependent weights alread exist, this
        # should be a no-op and just pass the dataset along
        if "elevation_vis_weight" in data:
            self.log.debug("Container already has the required dataset.")
        else:
            weights = data["vis_weight"][:].local_array
            # Remove the reference to the vis_weight dataset
            del data["vis_weight"]
            # Add the new elevation-dependent weight dataset
            data.add_dataset("elevation_vis_weight")
            # Write the weights into the new dataset, broadcasting over
            # the elevation axis
            data.weight[:].local_array[:] = weights[..., np.newaxis, :]

        return data


class MModeTransform(task.SingleTask):
    """Transform a sidereal stream to m-modes.

    Currently ignores any noise weighting.

    The maximum m used in the container is derived from the number of
    time samples, or if a manager is supplied `telescope.mmax` is used.

    Attributes
    ----------
    remove_integration_window : bool
        Deconvolve the effect of the finite width of the RA integration (presuming it
        was a rectangular integration window). This is applied to both the visibilities
        and the weights.
    use_fftw : bool
        If True, then use fftW to do the Fourier Transform, else use
        numpy fft. Default is True.
    """

    remove_integration_window = config.Property(proptype=bool, default=False)
    use_fftw = config.Property(proptype=bool, default=True)

    def setup(self, manager: io.TelescopeConvertible | None = None):
        """Set the telescope instance if a manager object is given.

        This is used to set the `mmax` used in the transform.

        Parameters
        ----------
        manager : manager.ProductManager, optional
            The telescope/manager used to set the `mmax`. If not set, `mmax`
            is derived from the timestream.
        """
        if manager is not None:
            self.telescope = io.get_telescope(manager)
        else:
            self.telescope = None

    def process(self, sstream: containers.SiderealContainer) -> containers.MContainer:
        """Perform the m-mode transform.

        Parameters
        ----------
        sstream : containers.SiderealStream or containers.HybridVisStream
            The input sidereal stream.

        Returns
        -------
        mmodes : containers.MModes
        """
        contmap = {
            containers.SiderealStream: containers.MModes,
            containers.HybridVisStream: containers.HybridVisMModes,
        }
        # Get the output container type
        out_cont = contmap[sstream.__class__]

        sstream.redistribute("freq")

        svis = sstream.vis[:].local_array
        sweight = sstream.weight[:].local_array

        # Sum the noise variance over time samples, this will become the noise
        # variance for the m-modes
        nra = sstream.weight.shape[-1]
        weight_sum = nra**2 * tools.invert_no_zero(
            tools.invert_no_zero(sweight).sum(axis=-1)
        )

        if self.telescope is not None:
            mmax = self.telescope.mmax
        else:
            mmax = svis.shape[-1] // 2

        # Create the container to store the modes in
        ma = out_cont(
            mmax=mmax,
            oddra=bool(nra % 2),
            axes_from=sstream,
            attrs_from=sstream,
            comm=sstream.comm,
        )
        ma.redistribute("freq")
        mvis = ma.vis[:].local_array
        mweight = ma.weight[:].local_array

        # Generate the m-mode transform directly into the output container
        # NOTE: Need to zero fill as not every element gets set within _make_marray
        mvis[:] = 0.0
        _make_marray(svis, mvis, mmax=None, dtype=None, use_fftw=self.use_fftw)

        # Assign the weights into the container
        mweight[:] = weight_sum[np.newaxis, np.newaxis, :, :]

        # Divide out the m-mode sinc-suppression caused by the rectangular integration window
        if self.remove_integration_window:
            m = np.arange(mmax + 1)
            w = np.sinc(m / nra)
            inv_w = tools.invert_no_zero(w)

            sl_vis = (slice(None),) + (np.newaxis,) * (len(mvis.shape) - 1)
            mvis[:] *= inv_w[sl_vis]

            sl_weight = (slice(None),) + (np.newaxis,) * (len(mweight.shape) - 1)
            mweight[:] *= w[sl_weight] ** 2

        return ma


def _make_marray(ts, mmodes=None, mmax=None, dtype=None, use_fftw=True):
    """Make an m-mode array from a sidereal stream.

    This will loop over the first axis of `ts` to avoid needing a lot of memory for
    intermediate arrays.

    It can also write the m-mode output directly into a passed `mmodes` array.
    """
    if dtype is None:
        dtype = np.complex64

    if mmodes is None and mmax is None:
        raise ValueError("One of `mmodes` or `mmax` must be set.")

    if mmodes is not None and mmax is not None:
        raise ValueError("If mmodes is set, mmax must be None.")

    if mmodes is not None and mmodes.shape[2:] != ts.shape[:-1]:
        raise ValueError(
            "ts and mmodes have incompatible shapes: "
            f"{mmodes.shape[2:]} != {ts.shape[:-1]}"
        )

    if mmodes is None:
        mmodes = np.zeros((mmax + 1, 2, *ts.shape[:-1]), dtype=dtype)

    if mmax is None:
        mmax = mmodes.shape[0] - 1

    # Total number of modes
    N = ts.shape[-1]
    # Calculate the max m to use for both positive and negative m. This is a little
    # tricky to get correct as we need to account for the number of negative
    # frequencies produced by the FFT
    mlim = min(N // 2, mmax)
    mlim_neg = N // 2 - 1 + N % 2 if mmax >= N // 2 else mmax

    # Do the transform and move the M axis to the front. There's
    # a bug in `pyfftw` which causes an error if `ts.ndim - len(axes) >= 2`,
    # so we have to flatten the other axes to get around that. This is
    # still faster than `numpy` or `scipy` ffts.
    shp = ts.shape
    if use_fftw:
        m_fft = fftw.fft(ts.reshape(-1, shp[-1]), axes=-1).reshape(shp)
    else:
        m_fft = np.fft.fft(ts.reshape(-1, shp[-1]), axis=-1).reshape(shp)

    m_fft = np.moveaxis(m_fft, -1, 0)

    # Write the positive and negative m's
    npos = mlim + 1
    nneg = mlim_neg + 1

    # Applying fft normalisation here is quite a bit
    # faster than applying it directly to `m_fft`. It's
    # not entirely clear why.
    norm = tools.invert_no_zero(shp[-1])
    mmodes[:npos, 0] = m_fft[:npos] * norm
    # Take the conjugate of the negative modes
    mmodes[1:nneg, 1] = m_fft[-1:-nneg:-1].conj() * norm

    return mmodes


class MModeInverseTransform(task.SingleTask):
    """Transform m-modes to sidereal stream.

    Currently ignores any noise weighting.

    .. warning::
        Using `apply_integration_window` will modify the input mmodes.

    Attributes
    ----------
    nra : int
        Number of RA bins in the output. Note that if the number of samples does not
        Nyquist sample the maximum m, information may be lost. If not set, then try to
        get from an `original_nra` attribute on the incoming MModes, otherwise determine
        an appropriate number of RA bins from the mmax.
    apply_integration_window : bool
        Apply the effect of the finite width of the RA integration (presuming a
        rectangular integration window). This is applied to both the visibilities and
        the weights. If this is true, as a side effect the input data will be modified
        in place.
    """

    nra = config.Property(proptype=int, default=None)
    apply_integration_window = config.Property(proptype=bool, default=False)

    def process(self, mmodes: containers.MContainer) -> containers.SiderealContainer:
        """Perform the m-mode inverse transform.

        Parameters
        ----------
        mmodes : containers.MModes
            The input m-modes.

        Returns
        -------
        sstream : containers.SiderealStream
            The output sidereal stream.
        """
        # NOTE: If n_time is smaller than Nyquist sampling the m-mode axis then
        # the m-modes get clipped. If it is larger, they get zero padded. This
        # is NOT passed directly as parameter 'n' to `numpy.fft.ifft`, as this
        # would give unwanted behaviour (https://github.com/numpy/numpy/pull/7593).

        # Ensure m-modes are distributed in frequency
        mmodes.redistribute("freq")

        # Use the nra property if set otherwise use the natural nra from the incoming
        # container
        nra_cont = 2 * mmodes.mmax + (1 if mmodes.oddra else 0)
        nra = self.nra if self.nra is not None else nra_cont

        # Apply the m-mode sinc-suppression caused by the rectangular integration window
        if self.apply_integration_window:
            m = np.arange(mmodes.mmax + 1)
            w = np.sinc(m / nra)
            inv_w = tools.invert_no_zero(w)

            sl_vis = (slice(None),) + (np.newaxis,) * (len(mmodes.vis.shape) - 1)
            mmodes.vis[:] *= w[sl_vis]

            sl_weight = (slice(None),) + (np.newaxis,) * (len(mmodes.weight.shape) - 1)
            mmodes.weight[:] *= inv_w[sl_weight] ** 2

        # Re-construct array of S-streams
        ssarray = _make_ssarray(mmodes.vis[:], n=nra)
        nra = ssarray.shape[-1]  # Get the actual nra used
        ssarray = mpiarray.MPIArray.wrap(ssarray[:], axis=0, comm=mmodes.comm)

        # Construct container and set visibility data
        sstream = containers.SiderealStream(
            ra=nra,
            axes_from=mmodes,
            attrs_from=mmodes,
            distributed=True,
            comm=mmodes.comm,
        )
        sstream.redistribute("freq")

        # Assign the visibilities and weights into the container
        sstream.vis[:] = ssarray
        # There is no way to recover time information for the weights.
        # Just assign the time average to each baseline and frequency.
        sstream.weight[:] = mmodes.weight[0, 0, :, :][:, :, np.newaxis] / nra

        return sstream


class SiderealMModeResample(task.group_tasks(MModeTransform, MModeInverseTransform)):
    """Resample a sidereal stream by FFT.

    This performs a forward and inverse m-mode transform to resample a sidereal stream.

    Attributes
    ----------
    nra : int
        The number of RA bins for the output stream.
    remove_integration_window, apply_integration_window : bool
        Remove the integration window from the incoming data, and/or apply it to the
        output sidereal stream.
    """

    pass


def _make_ssarray(mmodes, n=None):
    # Construct an array of sidereal time streams from m-modes
    marray = _unpack_marray(mmodes, n=n)
    return np.fft.ifft(marray * marray.shape[-1], axis=-1)


def _unpack_marray(mmodes, n=None):
    # Unpack m-modes into the correct format for an FFT
    # (i.e. from [m, +/-, freq, baseline] to [freq, baseline, time-FFT])

    shape = mmodes.shape[2:]
    mmax_plus = mmodes.shape[0] - 1
    if (mmodes[mmax_plus, 1, ...].flatten() == 0).all():
        mmax_minus = mmax_plus - 1
    else:
        mmax_minus = mmax_plus

    if n is None:
        ntimes = mmax_plus + mmax_minus + 1
    else:
        ntimes = n
        mmax_plus = np.amin((ntimes // 2, mmax_plus))
        mmax_minus = np.amin(((ntimes - 1) // 2, mmax_minus))

    # Create array to contain mmodes
    marray = np.zeros((*shape, ntimes), dtype=np.complex128)
    # Add the DC bin
    marray[..., 0] = mmodes[0, 0]
    # Add all m-modes up to mmax_minus
    for mi in range(1, mmax_minus + 1):
        marray[..., mi] = mmodes[mi, 0]
        marray[..., -mi] = mmodes[mi, 1].conj()

    if mmax_plus != mmax_minus:
        # In case of even number of samples. Add the Nyquist frequency.
        marray[..., mmax_plus] = mmodes[mmax_plus, 0]

    return marray


class LanczosRegridder(task.SingleTask):
    """Interpolate the time-like axis of a dataset onto a regular grid.

    Uses a maximum-likelihood inverse of a Lanczos interpolation to do the
    regridding. This gives a reasonably local regridding, that is pretty well
    behaved in m-space.

    Attributes
    ----------
    samples : int
        Number of samples to interpolate onto.
    start: float
        Start of the interpolated samples.
    end: float
        End of the interpolated samples.
    kernel_width : int
        Width of the interpolation kernel. NOTE: This was formally called
        `lanczos_width`.
    epsilon: float
        Numerical regulariser used in kernel inversion. Default is 1.0e-3.
    mask_zero_weight: bool
        Mask the output noise weights at frequencies where the weights were
        zero for all time samples.
    """

    samples = config.Property(proptype=int, default=1024)
    start = config.Property(proptype=float)
    end = config.Property(proptype=float)
    kernel_width = config.Property(proptype=int, default=5)
    epsilon = config.Property(proptype=float, default=1e-3)
    mask_zero_weight = config.Property(proptype=bool, default=False)

    def setup(self, observer):
        """Set the local observers position.

        Parameters
        ----------
        observer : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        self.observer = io.get_telescope(observer)

    def process(self, data):
        """Regrid visibility data in the time direction.

        Parameters
        ----------
        data : containers.TODContainer
            Time-ordered data.

        Returns
        -------
        new_data : containers.TODContainer
            The regularly gridded interpolated timestream.
        """
        # Redistribute if needed
        data.redistribute("freq")

        # View of data
        weight = data.weight[:].view(np.ndarray)
        vis_data = data.vis[:].view(np.ndarray)

        # Get input time grid
        timelike_axis = data.vis.attrs["axis"][-1]
        times = data.index_map[timelike_axis][:]

        # check bounds
        if self.start is None:
            self.start = times[0]
        if self.end is None:
            self.end = times[-1]
        if self.start < times[0] or self.end > times[-1]:
            msg = "Start or end points for regridder fall outside bounds of input data."
            self.log.error(msg)
            raise RuntimeError(msg)

        # perform regridding
        new_grid, new_vis, ni = self._regrid(vis_data, weight, times)

        # Wrap to produce MPIArray
        new_vis = mpiarray.MPIArray.wrap(new_vis, axis=data.vis.distributed_axis)
        ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # Create new container for output
        cont_type = data.__class__
        new_data = cont_type(axes_from=data, **{timelike_axis: new_grid})
        new_data.redistribute("freq")
        new_data.vis[:] = new_vis
        new_data.weight[:] = ni

        return new_data

    def _regrid(self, vis_data, weight, times):
        # Create a regular grid, padded at either end to supress interpolation issues
        pad = 5 * self.kernel_width
        interp_grid = (
            np.arange(-pad, self.samples + pad, dtype=np.float64) / self.samples
        )
        # scale to specified range
        interp_grid = interp_grid * (self.end - self.start) + self.start

        # Construct regridding matrix for reverse problem
        lzf = regrid.lanczos_forward_matrix(
            interp_grid, times, self.kernel_width
        ).T.copy()

        # Reshape data
        vr = vis_data.reshape(-1, vis_data.shape[-1])
        nr = weight.reshape(-1, vis_data.shape[-1])

        # Construct a signal 'covariance'
        Si = np.ones_like(interp_grid) * self.epsilon

        # Calculate the interpolated data and a noise weight at the points in the padded grid
        sts, ni = regrid.band_wiener(lzf, nr, Si, vr, 2 * self.kernel_width - 1)

        # Throw away the padded ends
        sts = sts[:, pad:-pad].copy()
        ni = ni[:, pad:-pad].copy()
        interp_grid = interp_grid[pad:-pad].copy()

        # Reshape to the correct shape
        sts = sts.reshape((*vis_data.shape[:-1], self.samples))
        ni = ni.reshape((*vis_data.shape[:-1], self.samples))

        if self.mask_zero_weight:
            # set weights to zero where there is no data
            w_mask = weight.sum(axis=-1) != 0.0
            ni *= w_mask[..., np.newaxis]

        return interp_grid, sts, ni


# Alias for compatibility
Regridder = LanczosRegridder


class ShiftRA(task.SingleTask):
    """Add a shift to the RA axis.

    This is useful for fixing a bug in earlier revisions of CHIME processing.

    Parameters
    ----------
    delta : float
        The shift to *add* to the RA axis.
    periodic : bool, optional
        If True, wrap any time sample that is shifted to RA > 360 deg around to its
        360-degree-periodic counterpart, and likewise for any sample that is shifted
        to RA < 0 deg. This wrapping is applied to the RA index_map along with any
        dataset with an `ra` axis. Default: False.
    """

    delta = config.Property(proptype=float)
    periodic = config.Property(proptype=bool, default=False)

    def process(
        self, sscont: containers.SiderealContainer
    ) -> containers.SiderealContainer:
        """Add a shift to the input sidereal container.

        Parameters
        ----------
        sscont
            The container to shift. The input is modified in place.

        Returns
        -------
        sscont
            The shifted container.
        """
        if not isinstance(sscont, containers.SiderealContainer):
            raise TypeError(
                f"Expected a SiderealContainer, got {type(sscont)} instead."
            )

        # Shift RA coordinates by delta
        sscont.ra[:] += self.delta

        if self.periodic:
            # If shift is positive, subtract 360 deg from any sample shifted to
            # > 360 deg. Same idea if shift is negative, for samples shifted to < 0 deg
            if self.delta > 0:
                sscont.ra[sscont.ra[:] >= 360] -= 360
            else:
                sscont.ra[sscont.ra[:] < 0] += 360

            # Get indices that sort shifted RA axis in ascending order, and apply sort
            ascending_ra_idx = np.argsort(sscont.ra[:])
            sscont.ra[:] = sscont.ra[ascending_ra_idx]

            # Loop over datasets in container
            for name, dset in sscont.datasets.items():
                if "ra" in dset.attrs["axis"]:
                    # If dataset has RA axis, identify which axis it is
                    ra_axis_idx = np.where(dset.attrs["axis"] == "ra")[0][0]

                    # Make sure dataset isn't distributed in RA. If it is, redistribute
                    # along another (somewhat arbitrarily chosen) axis. (This should
                    # usually not be necessary.)
                    if dset.distributed and dset.distributed_axis == ra_axis_idx:
                        redist_axis = max(ra_axis_idx - 1, 0)
                        dset.redistribute(redist_axis)

                    # Apply RA-sorting from earlier to the appropriate axis
                    slc = [slice(None)] * len(dset.attrs["axis"])
                    slc[ra_axis_idx] = ascending_ra_idx
                    dset[:] = dset[:][tuple(slc)]

        return sscont


class SelectPol(task.SingleTask):
    """Extract a subset of Stokes parameters from beamformed data.

    Supports extraction of Stokes I, Q, U, and V from beamformed data for
    linear polarisations (XX, YY, reXY, imXY). Assumes beamformed data are
    already calibrated and normalized.

    Attributes
    ----------
    pol : list of str
            List of Stokes parameters to extract.
            Must be a subset of ['I', 'Q', 'U', 'V'].
    """

    pol = config.Property(proptype=list)

    def setup(self):
        """Check that the requested polarisations are valid."""
        self.P = {
            "I": {"XX": 1, "YY": 1},
            "Q": {"XX": 1, "YY": -1},
            "U": {"reXY": 1},
            "V": {"imXY": 1},
        }

        missing_pol = [pstr for pstr in self.pol if pstr not in self.P]
        if missing_pol:
            raise ValueError(
                f"Do not support the selection of {missing_pol}.  "
                f"Available options include {list(self.P.keys())}."
            )

        if len(set(self.pol)) != len(self.pol):
            raise ValueError("Duplicate Stokes parameters requested in `pol`.")

    def process(self, polcont):
        """Extract the requested Stokes parameters from the input container.

        Parameters
        ----------
        polcont : ContainerBase
            A container with a 'pol' axis containing linear polarisation data
            (e.g., XX, YY, reXY, imXY).

        Returns
        -------
        outcont : same type as polcont
            A new container containing only the requested Stokes parameters.
        """
        polcont.redistribute("freq")

        if "pol" not in polcont.axes:
            raise ValueError(
                f"Container of type {type(polcont)} does not have a pol axis."
            )

        input_pol = list(polcont.index_map["pol"])

        # First, make sure that we have all of the input polarisations we require
        # to construct the selected polarisations.
        required_pol = [pol for pstr in self.pol for pol in self.P[pstr]]
        missing_pol = [pol for pol in np.unique(required_pol) if pol not in input_pol]
        if len(missing_pol) > 0:
            raise ValueError(
                f"Missing the following polarisations {missing_pol}, "
                f"which are needed to construct {self.pol}."
            )

        # Identify the "data" and "weight" dataset
        data_dset_name = getattr(polcont, "_data_dset_name", None)
        weight_dset_name = getattr(polcont, "_weight_dset_name", None)

        # Create the output container
        outcont = containers.empty_like(polcont, pol=np.array(self.pol))

        for name in polcont.datasets.keys():
            if name not in outcont.datasets:
                outcont.add_dataset(name)

        outcont.redistribute("freq")

        # Create a function that generates the appropriate slices
        def make_slice(index, axis_pos):
            return (slice(None),) * axis_pos + (index,)

        # Loop over datasets
        for name, dset in polcont.datasets.items():

            out_dset = outcont.datasets[name]

            if "pol" not in dset.attrs["axis"]:
                # No polarisation axis, directly copy over dataset
                out_dset[:] = dset[:]
                continue

            # Polarisation axis present, initialize output dataset to zero
            out_dset[:] = 0

            pol_axis_pos = list(dset.attrs["axis"]).index("pol")

            # If this is the weight dataset, keep track of where it is
            # non-zero across all of the summed polarisations.
            if name == weight_dset_name:
                flag = mpiarray.ones(
                    out_dset[:].global_shape,
                    axis=out_dset[:].axis,
                    dtype=bool,
                    comm=outcont.comm,
                )

            # Loop over output polarisations
            for oo, po in enumerate(self.pol):

                oslc = make_slice(oo, pol_axis_pos)
                pol_to_sum = self.P[po]
                nsum = len(pol_to_sum)

                # Loop over the input polarisations that we need to sum
                for pi, sign in pol_to_sum.items():

                    ii = input_pol.index(pi)
                    islc = make_slice(ii, pol_axis_pos)

                    if name == data_dset_name:
                        # This is the primary data product, where we would like
                        # to account for the sign.
                        out_dset[oslc] += sign * dset[islc]

                    elif name == weight_dset_name:
                        # Keep track of what samples have non-zero weight
                        # across all input polarisations.
                        flag[oslc] &= dset[islc] > 0.0

                        # Invert weights so that we properly propagate the variance.
                        out_dset[oslc] += tools.invert_no_zero(dset[islc])

                    elif np.issubdtype(out_dset.dtype, np.bool_):
                        # All cases of boolean datasets with a polarisation axis
                        # are masks, where a True value indicates a masked sample.
                        # In this case, we will combine the masks, so that if any
                        # of the input polarisations are masked then the output
                        # polarisation is also masked.
                        out_dset[oslc] |= dset[islc]

                    else:
                        # For all other datasets, we will simply take the average
                        out_dset[oslc] += dset[islc]

                # Normalize the output based on how many polarisations were summed
                if name == weight_dset_name:
                    out_dset[oslc] = (
                        flag[oslc] * nsum**2 * tools.invert_no_zero(out_dset[oslc])
                    )

                elif np.issubdtype(out_dset.dtype, np.integer):
                    out_dset[oslc] //= nsum

                elif np.issubdtype(out_dset.dtype, np.bool_):
                    pass

                elif "freq_cov" in name:
                    out_dset[oslc] /= nsum**2

                else:
                    out_dset[oslc] /= nsum

        return outcont


class PolWeightedAverage(task.SingleTask):
    """Compute an optimally weighted pseudo-Stokes I from XX and YY polarisations.

    This computes a weighted average:
        data_I = (w_XX * d_XX + w_YY * d_YY) / (w_XX + w_YY)
        weight_I = w_XX + w_YY

    Requires the input container to be a subclass of DataWeightContainer and to
    contain both XX and YY polarisations.
    """

    def process(self, polcont):
        """Compute pseudo-Stokes I from XX and YY.

        Parameters
        ----------
        polcont : DataWeightContainer
            A container with weight dataset and a 'pol' axis containing XX and YY.

        Returns
        -------
        outcont : same type as polcont
            A container with a single polarisation axis labeled 'I'.
        """
        # Check input container
        if not hasattr(polcont, "_weight_dset_name"):
            raise TypeError(
                "Input must be a subclass of DataWeightContainer with defined weight datasets."
            )

        if "pol" not in polcont.axes:
            raise ValueError(
                f"Input container of type {type(polcont)} does not have a 'pol' axis."
            )

        input_pol = list(polcont.index_map["pol"])
        if "XX" not in input_pol or "YY" not in input_pol:
            raise ValueError("Input must contain both 'XX' and 'YY' polarisations.")

        ixx = input_pol.index("XX")
        iyy = input_pol.index("YY")

        if iyy > ixx:
            start = ixx
            stride = iyy - ixx
        else:
            start = iyy
            stride = ixx - iyy

        pol_slice = slice(start, start + stride + 1, stride)

        def make_pol_slice(axis_names):
            axis = list(axis_names).index("pol")
            slc = (slice(None),) * axis + (pol_slice,)
            return axis, slc

        # Create output container
        outcont = containers.empty_like(polcont, pol=np.array(["I"]))

        for name in polcont.datasets.keys():
            if name not in outcont.datasets:
                outcont.add_dataset(name)

        polcont.redistribute("freq")
        outcont.redistribute("freq")

        # Extract the weights dataset for the XX and YY polarisation.
        # Save their sum to the output container.
        waxis = polcont.weight.attrs["axis"]
        wpax, wslc = make_pol_slice(waxis)

        weight = polcont.weight[wslc]
        outcont.weight[:] = np.sum(weight, axis=wpax, keepdims=True)
        norm = tools.invert_no_zero(outcont.weight[:])

        # Loop over all other datasets
        for name, dset in polcont.datasets.items():

            # Already dealt with weights
            if name == polcont._weight_dset_name:
                continue

            # If the dataset does not have a pol axis, then just
            # copy it over directly and continue.
            if "pol" not in dset.attrs["axis"]:
                outcont.datasets[name][:] = dset[:]
                continue

            # Make the weights broadcastable against this dataset
            pax, dslc = make_pol_slice(dset.attrs["axis"])
            wexp = tools.broadcast_weights(waxis, dset.attrs["axis"])

            # Take the weighted average
            outcont.datasets[name][:] = (
                np.sum(weight[wexp] * dset[dslc], axis=pax, keepdims=True) * norm[wexp]
            )

        return outcont


class StokesIVis(task.SingleTask):
    """Extract instrumental Stokes I from visibilities."""

    def setup(self, telescope):
        """Set the local observers.

        Parameters
        ----------
        telescope : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, data):
        """Extract instrumental Stokes I.

        This process will reduce the length of the baseline axis.

        Parameters
        ----------
        data : containers.VisContainer
            Container with visibilities and baselines matching
            the telescope object.

        Returns
        -------
        data : containers.VisContainer
            Container with the same type as `data`, with polarised
            baselines combined into Stokes I.
        """
        data.redistribute("freq")

        # Get stokes I
        vis, weight, baselines = stokes_I(data, self.telescope)

        # Make the output container
        # TODO: the axes for this container should probably
        # be adjusted to make more sense
        out = containers.empty_like(data, stack=baselines)
        out.redistribute("freq")

        out.vis[:] = vis.redistribute(0)
        out.weight[:] = weight.redistribute(0)

        return out


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
    vis_I : mpiarray.MPIArray[nfreq, nbase, ntime]
        The instrumental Stokes I visibilities, distributed over baselines.
    vis_weight : mpiarray.MPIArray[nfreq, nbase, ntime]
        The weights for each visibility, distributed over baselines.
    ubase : np.ndarray[nbase, 2]
        Baseline vectors corresponding to output.
    """
    # Make sure the data is distributed in a reasonable way
    sstream.redistribute("freq")
    # Construct a complex number representing each baseline (used for determining
    # unique baselines).
    # Due to floating point precision, some baselines don't get matched as having
    # the same lengths. To get around this, round all separations to 0.1 mm precision
    bl_round = np.around(tel.baselines[:, 0] + 1.0j * tel.baselines[:, 1], 4)

    # Map unique baseline lengths to each polarisation pair
    ubase, uinv, ucount = np.unique(bl_round, return_inverse=True, return_counts=True)
    ubase = ubase.astype(np.complex128, copy=False).view(np.float64).reshape(-1, 2)

    # Construct the output arrays
    new_shape = (
        sstream.vis.global_shape[0],
        ubase.shape[0],
        sstream.vis.global_shape[2],
    )
    vis_I = mpiarray.zeros(new_shape, dtype=sstream.vis.dtype, axis=0)
    vis_weight = mpiarray.zeros(new_shape, dtype=sstream.weight.dtype, axis=0)

    # Find co-pol baselines (XX and YY)
    pairs = tel.uniquepairs
    pols = tel.polarisation[pairs]
    is_copol = pols[:, 0] == pols[:, 1]

    # Iterate over products to construct the Stokes I vis
    ssv = sstream.vis[:].local_array
    ssw = sstream.weight[:].local_array

    for ii, ui in enumerate(uinv):
        # Skip if not a co-pol baseline
        if not is_copol[ii]:
            continue

        # Skip if not all polarisations are included
        if ucount[ui] < 4:
            continue

        # Skip if there's a bad feed
        if tel.feedmap[(*pairs[ii],)] == -1:
            continue

        # Accumulate the visibilities and weights
        vis_I.local_array[:, ui] += ssv[:, ii]
        vis_weight.local_array[:, ui] += ssw[:, ii]

    return vis_I, vis_weight, ubase


class TransformJanskyToKelvin(task.SingleTask):
    """Task to convert from Jy to Kelvin and vice-versa.

    This integrates over the primary beams in the telescope class to derive the
    brightness temperature to flux conversion.

    Attributes
    ----------
    convert_Jy_to_K : bool
        If True, apply a Jansky to Kelvin conversion factor. If False apply a Kelvin to
        Jansky conversion.
    reference_declination : float, optional
        The declination to set the flux reference for. A source transiting at this
        declination will produce a visibility signal equal to its flux. If `None`
        (default) use the zenith.
    share : {"none", "all"}
        Which datasets should the output share with the input. Default is "all".
    nside : int
        The NSIDE to use for the primary beam area calculation. This may need to be
        increased for beams with intricate small scale structure. Default is 256.
    """

    convert_Jy_to_K = config.Property(proptype=bool, default=True)
    reference_declination = config.Property(proptype=float, default=None)
    share = config.enum(["none", "all"], default="all")

    nside = config.Property(proptype=int, default=256)

    def setup(self, telescope: io.TelescopeConvertible):
        """Set the telescope object.

        Parameters
        ----------
        telescope
            An object we can get a telescope object from. This telescope must be able to
            calculate the beams at all incoming frequencies.
        """
        self.telescope = io.get_telescope(telescope)
        self.telescope._init_trans(self.nside)

        # If not explicitly set, use the zenith as the reference declination
        if self.reference_declination is None:
            self.reference_declination = self.telescope.latitude

        self._omega_cache = {}

    def _beam_area(self, feed, freq):
        """Calculate the primary beam solid angle."""
        import healpy

        beam = self.telescope.beam(feed, freq)
        horizon = self.telescope._horizon[:, np.newaxis]
        beam_pow = np.sum(np.abs(beam) ** 2 * horizon, axis=1)

        pxarea = 4 * np.pi / beam.shape[0]
        omega = beam_pow.sum() * pxarea

        # Normalise omega by the squared magnitude of the beam at the reference position
        # NOTE: this is slightly less accurate than the previous approach of reseting
        # the internal `_angpos` property to force evaluation of the beam at the exact
        # coordinates, but is more generically applicable, and works (for instance) with
        # the CHIMEExternalBeam class.
        #
        # Also, for a reason I don't fully understand it's more accurate to use the
        # value of the pixel including the reference position, and not do an
        # interpolation using it's neighbours...
        beam_ref = beam_pow[
            healpy.ang2pix(self.nside, 0.0, self.reference_declination, lonlat=True)
        ]
        omega *= tools.invert_no_zero(beam_ref)

        return omega

    def process(self, sstream: containers.SiderealStream) -> containers.SiderealStream:
        """Apply the brightness temperature to flux conversion to the data.

        Parameters
        ----------
        sstream
            The visibilities to apply the conversion to. They are converted to/from
            brightness temperature units depending on the setting of `convert_Jy_to_K`.

        Returns
        -------
        new_sstream
            Visibilities with the conversion applied. This may be the same as the input
            container if `share == "all"`.
        """
        import scipy.constants as c

        sstream.redistribute("freq")

        # Get the local frequencies in the sidereal stream
        sfreq = sstream.vis.local_offset[0]
        efreq = sfreq + sstream.vis.local_shape[0]
        local_freq = sstream.freq[sfreq:efreq]

        # Get the indices of the incoming frequencies as far as the telescope class is
        # concerned
        local_freq_inds = []
        for freq in local_freq:
            local_freq_inds.append(np.argmin(np.abs(self.telescope.frequencies - freq)))

        # Get the feedpairs we have data for and their beamclass (usually this maps to
        # polarisation)
        feedpairs = structured_to_unstructured(sstream.prodstack)
        beamclass_pairs = self.telescope.beamclass[feedpairs]

        # Calculate all the unique beams that we need to calculate areas for
        unique_beamclass, bc_index = np.unique(beamclass_pairs, return_index=True)

        # Calculate any missing beam areas and to the cache
        for beamclass, bc_ind in zip(unique_beamclass, bc_index):
            feed_ind = feedpairs.ravel()[bc_ind]

            for freq, freq_ind in zip(local_freq, local_freq_inds):
                key = (beamclass, freq)

                if key not in self._omega_cache:
                    self._omega_cache[key] = self._beam_area(feed_ind, freq_ind)

        # Loop over all frequencies and visibilities and get the effective primary
        # beam area for each
        om_ij = np.zeros((len(local_freq), sstream.vis.shape[1]))
        for fi, freq in enumerate(local_freq):
            for bi, (bci, bcj) in enumerate(beamclass_pairs):
                om_i = self._omega_cache[(bci, freq)]
                om_j = self._omega_cache[(bcj, freq)]
                om_ij[fi, bi] = (om_i * om_j) ** 0.5

        # Calculate the Jy to K conversion
        wavelength = (c.c / (local_freq * 10**6))[:, np.newaxis, np.newaxis]
        K_to_Jy = 2 * 1e26 * c.k * om_ij[:, :, np.newaxis] / wavelength**2
        Jy_to_K = tools.invert_no_zero(K_to_Jy)

        # Get the container we will apply the conversion to (either the input, or a
        # copy)
        if self.share == "all":
            new_stream = sstream
        else:  # self.share == "none"
            new_stream = sstream.copy()

        # Apply the conversion to the data and the weights
        vis = new_stream.vis[:].local_array
        weight = new_stream.weight[:].local_array
        if self.convert_Jy_to_K:
            vis *= Jy_to_K
            weight *= K_to_Jy**2
        else:
            vis *= K_to_Jy
            weight *= Jy_to_K**2

        return new_stream


class MixData(task.SingleTask):
    """Mix together pieces of data with specified weights.

    This can generate arbitrary linear combinations of the data and weights for both
    `SiderealStream` and `RingMap` objects, and can be used for many purposes such as:
    adding together simulated timestreams, injecting signal into data, replacing weights
    in simulated data with those from real data, performing jackknifes, etc.

    All coefficients are applied naively to generate the final combinations, i.e. no
    normalisations or weighted summation is performed.

    Attributes
    ----------
    data_coeff : list
        A list of coefficients to apply to the data dataset of each input container to
        produce the final output. These are applied to either the `vis` or `map` dataset
        depending on the the type of the input container.
    weight_coeff : list
        Coefficient to be applied to each input containers weights to generate the
        output.
    tag_coeff : list
        Boolean array indicating which input containers tags should be used to generate
        the output tag.
    aux_coeff : dict
        Coefficients to be applied to auxiliary datasets in the input container to
        generate the output.  This should be a dictionary where each key is the name
        of a dataset in the input container, and the corresponding value is a list
        of coefficients used to mix.
    invert_weight : bool
        Invert the weights to convert to variance prior to mixing.  Re-invert in the
        final mixed data product to convert back to inverse variance.
    require_nonzero_weight : bool
        Set the weight to zero in the mixed data if the weight is zero in any of the
        input data.
    """

    data_coeff = config.list_type(type_=float)
    weight_coeff = config.list_type(type_=float)
    tag_coeff = config.list_type(type_=bool)
    aux_coeff = config.Property(proptype=dict)
    invert_weight = config.Property(proptype=bool, default=False)
    require_nonzero_weight = config.Property(proptype=bool, default=False)

    mixed_data = None

    def setup(self):
        """Check the lists have the same length."""
        if len(self.data_coeff) != len(self.weight_coeff):
            raise config.CaputConfigError(
                "data and weight coefficient lists must be the same length"
            )

        self._data_ind = 0
        self._tags = []
        self._wfunc = tools.invert_no_zero if self.invert_weight else lambda x: x

    def process(
        self,
        data: (
            containers.SiderealStream | containers.HybridVisStream | containers.RingMap
        ),
    ):
        """Add the input data into the mixed data output.

        Parameters
        ----------
        data
            The data to be added into the mix.
        """
        if self._data_ind >= len(self.data_coeff):
            raise RuntimeError(
                "This task cannot accept more items than there are coefficents set."
            )

        if self.mixed_data is None:
            self.mixed_data = containers.empty_like(data)

            # If requested, add auxiliary datasets
            for key in self.aux_coeff.keys():
                if key not in self.mixed_data.datasets:
                    self.mixed_data.add_dataset(key)
                self.mixed_data.datasets[key][:] = 0.0

            # Redistribute over frequency
            self.mixed_data.redistribute("freq")

            # Zero out data and weights
            self.mixed_data.data[:] = 0.0
            self.mixed_data.weight[:] = 0.0

            if self.require_nonzero_weight:
                self._flag = mpiarray.ones(
                    self.mixed_data.weight.shape,
                    axis=self.mixed_data.weight.distributed_axis,
                    comm=self.mixed_data.comm,
                    dtype=bool,
                )

        # Validate the types are the same
        if type(self.mixed_data) is not type(data):
            raise TypeError(
                f"type(data) (={type(data)}) must match "
                f"type(data_stack) (={type(self.type)}"
            )

        data.redistribute("freq")

        # Validate the shapes match
        if self.mixed_data.data.shape != data.data.shape:
            raise ValueError(
                f"Size of data ({data.data.shape}) must match "
                f"data_stack ({self.mixed_data.data.shape})"
            )

        if self.mixed_data.weight.shape != data.weight.shape:
            raise ValueError(
                f"Size of data ({data.weight.shape}) must match "
                f"data_stack ({self.mixed_data.weightshape})"
            )

        # Mix in the data and weights
        dco = self.data_coeff[self._data_ind]
        if dco != 0.0:
            self.mixed_data.data[:] += dco * data.data[:]

        wco = self.weight_coeff[self._data_ind]
        if wco != 0.0:
            self.mixed_data.weight[:] += wco * self._wfunc(data.weight[:])

            # Update the flag
            if self.require_nonzero_weight:
                self._flag &= data.weight[:] > 0.0

        # Deal with auxiliary datasets
        for key, aux_coeff in self.aux_coeff.items():
            aco = aux_coeff[self._data_ind]
            if aco != 0.0:
                self.mixed_data.datasets[key][:] += aco * data.datasets[key][:]

        # Save the tags
        if "tag" in data.attrs and (
            self.tag_coeff is None or self.tag_coeff[self._data_ind]
        ):
            self._tags.append(data.attrs["tag"])

        self._data_ind += 1

    def _make_output(
        self,
    ) -> containers.SiderealStream | containers.HybridVisStream | containers.RingMap:
        """Return the container with the mixed inputs.

        Returns
        -------
        mixed_data
            The mixed data.
        """
        if self._data_ind != len(self.data_coeff):
            raise RuntimeError(
                "Did not receive enough inputs. "
                f"Got {self._data_ind}, expected {len(self.data_coeff)}."
            )

        # Get an ephemeral reference to the mixed data and remove the task reference so
        # the object can be eventually deleted
        data = self.mixed_data
        self.mixed_data = None

        # Apply the logical AND of all flags
        if self.require_nonzero_weight:
            data.weight[:] *= self._flag.astype(data.weight.dtype)
            self._flag = None

        # Convert back to inverse variance
        data.weight[:] = self._wfunc(data.weight[:])

        # Combine the tags
        data.attrs["tag"] = "_".join(self._tags)

        return data

    def process_finish(
        self,
    ) -> containers.SiderealStream | containers.HybridVisStream | containers.RingMap:
        """Return the container with the mixed inputs.

        Returns
        -------
        mixed_data
            The mixed data.
        """
        return self._make_output()


class Jackknife(MixData):
    """Perform a jackknife of two datasets.

    This is identical to MixData but sets the default config properties
    to values appropriate for carrying out a jackknife of two datasets.
    """

    data_coeff = config.list_type(type_=float, default=[0.5, -0.5])
    weight_coeff = config.list_type(type_=float, default=[0.25, 0.25])
    tag_coeff = config.list_type(type_=bool, default=[True, True])
    invert_weight = config.Property(proptype=bool, default=True)
    require_nonzero_weight = config.Property(proptype=bool, default=True)


class MixTwoDatasets(MixData):
    """Mix two datasets in a single iteration."""

    data_coeff = config.list_type(type_=float, length=2)
    weight_coeff = config.list_type(type_=float, length=2)
    tag_coeff = config.list_type(type_=bool, length=2)

    def process(self, data1, data2):
        """Combine the two datasets into mixed data output.

        Parameters
        ----------
        data1 : containers.SiderealStream | containers.RingMap
            First dataset to mix
        data2 : containers.SiderealStream | containers.RingMap
            Second dataset to mix
        """
        # Combine the two datasets
        super().process(data1)
        super().process(data2)

        out = self._make_output()

        # Reset data counter and tags
        self._data_ind = 0
        self._tags = []

        return out

    def process_finish(self):
        """Overwrite `process_finish` to no-op."""
        return


class Downselect(io.SelectionsMixin, task.SingleTask):
    """Apply axis selections to a container.

    Apply slice or `np.take` operations across multiple axes of a container.
    The selections are applied to every dataset.

    If a dataset is distributed, there must be at least one axis not included
    in the selections.
    """

    def process(self, data: containers.ContainerBase) -> containers.ContainerBase:
        """Apply downselections to the container.

        Parameters
        ----------
        data
            Container to process

        Returns
        -------
        out
            Container of same type as the input with specific axis selections.
        """
        sel = {}

        # Parse axes with selections and reformat to use only
        # the axis name
        for k in self.selections:
            *axis, type_ = k.split("_")
            axis_name = "_".join(axis)

            ax_sel = self._sel.get(f"{axis_name}_sel")

            if type_ == "map":
                # Use index map to get the correct axis indices
                imap = list(data.index_map[axis_name])
                ax_sel = [imap.index(x) for x in ax_sel]

            if ax_sel is not None:
                sel[axis_name] = ax_sel

        # Figure out the axes for the new container and
        # Apply the downselections to each axis index_map
        output_axes = {
            ax: mpiarray._apply_sel(data.index_map[ax], ax_sel, 0)
            for ax, ax_sel in sel.items()
        }
        # Create the output container without initializing any datasets.
        out = data.__class__(
            axes_from=data, attrs_from=data, skip_datasets=True, **output_axes
        )
        containers.copy_datasets_filter(
            data, out, selection=sel, copy_without_selection=True
        )

        return out


class ReduceBase(task.SingleTask):
    """Apply a weighted reduction operation across specific axes.

    This is non-functional without overriding the `reduction` method.

    There must be at least one axis not included in the reduction.

    Attributes
    ----------
    axes : list
        Axis names to apply the reduction to
    dataset : str
        Dataset name to reduce.
    weighting : str
        Which type of weighting to use, if applicable. Options are "none",
        "masked", or "weighted"
    """

    axes = config.Property(proptype=list)
    dataset = config.Property(proptype=str)
    weighting = config.enum(["none", "masked", "weighted"], default="none")

    _op = None

    def process(self, data: containers.ContainerBase) -> containers.ContainerBase:
        """Downselect and apply the reduction operation to the data.

        Parameters
        ----------
        data
            Dataset to process.

        Returns
        -------
        out
            Dataset of same type as input with axes reduced. Any datasets
            which are not included in the reduction list will not be initialized,
            other than weights.
        """
        out = self._make_output_container(data)
        out.add_dataset(self.dataset)

        # Get the dataset
        ds = data.datasets[self.dataset]
        original_ax_id = ds.distributed_axis

        # Get the axis indices to apply the operation over
        ds_axes = list(ds.attrs["axis"])

        # Get the new axis to distribute over
        if ds_axes[original_ax_id] not in self.axes:
            new_ax_id = original_ax_id
        else:
            ax_priority = [
                x for _, x in sorted(zip(ds.shape, ds_axes)) if x not in self.axes
            ]
            if not ax_priority:
                raise ValueError(
                    "Could not find a valid axis to redistribute. At least one "
                    "axis must be omitted from filtering."
                )
            # Get the longest axis
            new_ax_id = ds_axes.index(ax_priority[-1])

        new_ax_name = ds_axes[new_ax_id]

        # Redistribute the dataset to the target axis
        ds.redistribute(new_ax_id)
        # Redistribute the output container (group) to the target axis
        # Since this is a container, distribute based on axis name
        # rather than index
        out.redistribute(new_ax_name)

        # Get the weights
        if hasattr(data, "weight"):
            # The weights should be distributed over the same axis as the array,
            # even if they don't share all the same axes
            w_axes = list(data.weight.attrs["axis"])
            new_weight_ax = w_axes.index(new_ax_name)
            weight = data.weight[:].redistribute(new_weight_ax)
            # Insert a size 1 axis for each missing axis in the weights
            wslc = [slice(None) if ax in w_axes else None for ax in ds_axes]
            weight = weight.local_array[tuple(wslc)]
        else:
            self.log.info("No weights available. Using equal weighting.")
            wslc = None
            weight = np.ones(ds.local_shape, ds.dtype)

        # Apply the reduction, ensuring that the weights have the correct dimension
        weight = np.broadcast_to(weight, ds.local_shape, subok=False)
        apply_over = tuple([ds_axes.index(ax) for ax in self.axes if ax in ds_axes])

        reduced, reduced_weight = self.reduction(
            ds[:].local_array[:], weight, apply_over
        )

        # Add the reduced data and redistribute the container back to the
        # original axis
        out[self.dataset][:] = reduced[:]

        if hasattr(out, "weight"):
            if wslc is None:
                out.weight[:] = reduced_weight
            else:
                owslc = [ws if ws is not None else 0 for ws in wslc]
                out.weight[:] = reduced_weight[tuple(owslc)]

        # Redistribute bcak to the original axis, again using the axis name
        out.redistribute(ds_axes[original_ax_id])

        return out

    def _make_output_container(
        self, data: containers.ContainerBase
    ) -> containers.ContainerBase:
        """Create the output container."""
        # For a collapsed axis, the meaning of the index map will depend on
        # the reduction being done, and can be meaningless. The first value
        # of the relevant index map is chosen as the default to provide
        # some meaning to the index map regardless of the reduction operation
        # or reduction axis involved
        output_axes = {ax: np.array([data.index_map[ax][0]]) for ax in self.axes}

        # Create the output container without initializing any datasets.
        # Add some extra metadata about which axes were reduced and which
        # datasets are meaningful
        out = data.__class__(
            axes_from=data, attrs_from=data, skip_datasets=True, **output_axes
        )
        out.attrs["reduced"] = True
        out.attrs["reduction_axes"] = np.array(self.axes)
        out.attrs["reduced_dataset"] = self.dataset
        out.attrs["reduction_op"] = self._op

        # Initialize the weight dataset
        if "weight" in data.datasets:
            out.add_dataset("weight")
        elif "vis_weight" in data.datasets:
            out.add_dataset("vis_weight")

        return out

    def reduction(
        self, arr: np.ndarray, weight: np.ndarray, axis: tuple
    ) -> tuple[np.ndarray, np.ndarray]:
        """Overwrite to implement the reductino operation."""
        raise NotImplementedError


class ReduceVar(ReduceBase):
    """Take the weighted variance of a container."""

    _op = "variance"

    def reduction(self, arr, weight, axis):
        """Apply a weighted variance."""
        if self.weighting == "none":
            v = np.var(arr, axis=axis, keepdims=True)

            return v, np.ones_like(v)

        if self.weighting == "masked":
            weight = (weight > 0).astype(weight.dtype)

        # Calculate the inverted sum of the weights. This is used
        # more than once
        ws = np.sum(weight, axis=axis, keepdims=True)
        iws = invert_no_zero(ws)
        # Get the weighted mean
        mu = np.sum(weight * arr, axis=axis, keepdims=True) * iws
        # Get the weighted variance
        v = np.sum(weight * (arr - mu) ** 2, axis=axis, keepdims=True) * iws

        return v, ws


class ReduceChisq(ReduceBase):
    """Calculate the chi-squared per degree of freedom.

    Assumes that the visibilities are uncorrelated noise
    whose inverse variance is given by the weight dataset.
    """

    _op = "chisq_per_dof"

    def reduction(self, arr, weight, axis):
        """Apply a chi-squared calculation."""
        # Get the total number of unmasked samples
        num = np.maximum(np.sum(weight > 0, axis=axis, keepdims=True) - 1, 0)

        # Calculate the inverted sum of the weights
        iws = invert_no_zero(np.sum(weight, axis=axis, keepdims=True))

        # Get the weighted mean
        mu = np.sum(weight * arr, axis=axis, keepdims=True) * iws

        # Get the chi-squared per degree of freedom
        v = np.sum(
            weight * np.abs(arr - mu) ** 2, axis=axis, keepdims=True
        ) * invert_no_zero(num)

        return v, num


class HPFTimeStream(task.SingleTask):
    """High pass filter a timestream.

    This is done by solving for a low-pass filtered version of the timestream and then
    subtracting it from the original.

    Parameters
    ----------
    tau
        Timescale in seconds to filter out fluctuations below.
    pad
        Implicitly pad the timestream with this many multiples of tau worth of zeros.
        This is used to mitigate edge effects. The default is 2.
    window
        Use a Blackman window when determining the low-pass filtered timestream. When
        applied this approximately doubles the length of the timescale, which is only
        crudely corrected for.
    prior
        This should be approximately the size of the large scale fluctuations that we
        will use as a regulariser.
    """

    tau = config.Property(proptype=float)
    pad = config.Property(proptype=float, default=2)
    window = config.Property(proptype=bool, default=True)

    prior = config.Property(proptype=float, default=1e2)

    def process(self, tstream: containers.TODContainer) -> containers.TODContainer:
        """High pass filter a time stream.

        Parameters
        ----------
        tstream
            A TOD container that also implements DataWeightContainer.

        Returns
        -------
        filtered_tstream
            The high-pass filtered time stream.
        """
        if not isinstance(tstream, containers.DataWeightContainer):
            # NOTE: no python intersection type so need to do this for now
            raise TypeError("Need a DataWeightContainers")

        if "time" != tstream.data.attrs["axis"][-1]:
            raise TypeError("'time' is not the last axis of the dataset.")

        if tstream.data.shape != tstream.weight.shape:
            raise ValueError("Data and weights must have the same shape.")

        # Distribute over the first axis
        tstream.redistribute(tstream.data.attrs["axis"][0])

        tau = 2 * self.tau if self.window else self.tau

        dt = np.diff(tstream.time)
        if not np.allclose(dt, dt[0], atol=1e-4):
            self.log.warn(
                "Samples are not regularly spaced. This might not work super well."
            )

        total_T = tstream.time[-1] - tstream.time[0] + 2 * tau

        # Calculate the nearest integer multiple of modes based on the total length and
        # the timescale
        nmodes = int(np.ceil(total_T / tau))

        # Calculate the conjugate fourier frequencies to use, we don't need to be in the
        # canonical order as we're going to calculate this exactly via matrices
        t_freq = np.arange(-nmodes, nmodes) / total_T

        F = np.exp(2.0j * np.pi * tstream.time[:, np.newaxis] * t_freq[np.newaxis, :])

        if self.window:
            F *= np.blackman(2 * nmodes)[np.newaxis, :]

        Fh = F.T.conj().copy()

        dflat = tstream.data[:].view(np.ndarray).reshape(-1, len(tstream.time))
        wflat = tstream.weight[:].view(np.ndarray).reshape(-1, len(tstream.time))

        Si = np.identity(2 * nmodes) * self.prior**-2

        for ii in range(dflat.shape[0]):
            d, w = dflat[ii], wflat[ii]

            wsum = w.sum()
            if wsum == 0:
                continue

            m = np.sum(d * w) / wsum

            # dirty = Fh @ ((d - m) * w)
            # Ci = Fh @ (w[:, np.newaxis] * F)
            d -= m
            dirty = np.dot(Fh, (d * w))
            Ci = np.dot(Fh, w[:, np.newaxis] * F)
            Ci += Si

            f_lpf = la.solve(Ci, dirty, assume_a="pos")

            # As we know the result will be real, split up the matrix multiplication to
            # guarantee this
            t_lpf = np.dot(F.real, f_lpf.real) - np.dot(F.imag, f_lpf.imag)
            d -= t_lpf

        return tstream

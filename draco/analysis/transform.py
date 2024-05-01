"""Miscellaneous transformations to do on data.

This includes grouping frequencies and products to performing the m-mode transform.
"""

from typing import Optional, Tuple, Union, overload

import numpy as np
import scipy.linalg as la
from caput import config, mpiarray
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


class CollateProducts(task.SingleTask):
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

    weight = config.Property(proptype=str, default="natural")

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
            Telescope object to use
        """
        if self.weight not in ["natural", "uniform", "inverse_variance"]:
            KeyError(f"Do not recognize weight = {self.weight!s}")

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
                    "There are %d stacked baselines that are masked "
                    "in the telescope instance." % np.sum(~stack_flag)
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
            containers.copy_datasets_filter(data, newdata, "freq", newindex)
        else:
            newdata.vis[:] = data.vis[newindex]
            newdata.weight[:] = data.weight[newindex]
            newdata.gain[:] = data.gain[newindex]

            newdata.input_flags[:] = data.input_flags[:]

        # Switch back to frequency distribution
        data.redistribute("freq")
        newdata.redistribute("freq")

        return newdata


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
    """

    remove_integration_window = config.Property(proptype=bool, default=False)

    def setup(self, manager: Optional[io.TelescopeConvertible] = None):
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

        # Get the output container and figure out at which position is it's
        # frequency axis
        out_cont = contmap[sstream.__class__]

        sstream.redistribute("freq")

        # Sum the noise variance over time samples, this will become the noise
        # variance for the m-modes
        nra = sstream.weight.shape[-1]
        weight_sum = nra**2 * tools.invert_no_zero(
            tools.invert_no_zero(sstream.weight[:]).sum(axis=-1)
        )

        if self.telescope is not None:
            mmax = self.telescope.mmax
        else:
            mmax = sstream.vis.shape[-1] // 2

        # Create the container to store the modes in
        ma = out_cont(
            mmax=mmax,
            oddra=bool(nra % 2),
            axes_from=sstream,
            attrs_from=sstream,
            comm=sstream.comm,
        )
        ma.redistribute("freq")

        # Generate the m-mode transform directly into the output container
        # NOTE: Need to zero fill as not every element gets set within _make_marray
        ma.vis[:] = 0.0
        _make_marray(sstream.vis[:].local_array, ma.vis[:].local_array)

        # Assign the weights into the container
        ma.weight[:] = weight_sum[np.newaxis, np.newaxis, :, :]

        # Divide out the m-mode sinc-suppression caused by the rectangular integration window
        if self.remove_integration_window:
            m = np.arange(mmax + 1)
            w = np.sinc(m / nra)
            inv_w = tools.invert_no_zero(w)

            sl_vis = (slice(None),) + (np.newaxis,) * (len(ma.vis.shape) - 1)
            ma.vis[:] *= inv_w[sl_vis]

            sl_weight = (slice(None),) + (np.newaxis,) * (len(ma.weight.shape) - 1)
            ma.weight[:] *= w[sl_weight] ** 2

        return ma


def _make_marray(ts, mmodes=None, mmax=None, dtype=None):
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
        mmodes = np.zeros((mmax + 1, 2) + ts.shape[:-1], dtype=dtype)

    if mmax is None:
        mmax = mmodes.shape[0] - 1

    # Total number of modes
    N = ts.shape[-1]
    # Calculate the max m to use for both positive and negative m. This is a little
    # tricky to get correct as we need to account for the number of negative
    # frequencies produced by the FFT
    mlim = min(N // 2, mmax)
    mlim_neg = N // 2 - 1 + N % 2 if mmax >= N // 2 else mmax

    for i in range(ts.shape[0]):
        m_fft = np.fft.fft(ts[i], axis=-1) / ts.shape[-1]

        # Loop and copy over positive and negative m's
        # NOTE: this is done as a loop to try and save memory
        for mi in range(mlim + 1):
            mmodes[mi, 0, i] = m_fft[..., mi]

        for mi in range(1, mlim_neg + 1):
            mmodes[mi, 1, i] = m_fft[..., -mi].conj()

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


class Regridder(task.SingleTask):
    """Interpolate time-ordered data onto a regular grid.

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
    lanczos_width : int
        Width of the Lanczos interpolation kernel.
    snr_cov: float
        Ratio of signal covariance to noise covariance (used for Wiener filter).
    mask_zero_weight: bool
        Mask the output noise weights at frequencies where the weights were
        zero for all time samples.
    """

    samples = config.Property(proptype=int, default=1024)
    start = config.Property(proptype=float)
    end = config.Property(proptype=float)
    lanczos_width = config.Property(proptype=int, default=5)
    snr_cov = config.Property(proptype=float, default=1e-8)
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
        self.observer = observer

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
        pad = 5 * self.lanczos_width
        interp_grid = (
            np.arange(-pad, self.samples + pad, dtype=np.float64) / self.samples
        )
        # scale to specified range
        interp_grid = interp_grid * (self.end - self.start) + self.start

        # Construct regridding matrix for reverse problem
        lzf = regrid.lanczos_forward_matrix(
            interp_grid, times, self.lanczos_width
        ).T.copy()

        # Reshape data
        vr = vis_data.reshape(-1, vis_data.shape[-1])
        nr = weight.reshape(-1, vis_data.shape[-1])

        # Construct a signal 'covariance'
        Si = np.ones_like(interp_grid) * self.snr_cov

        # Calculate the interpolated data and a noise weight at the points in the padded grid
        sts, ni = regrid.band_wiener(lzf, nr, Si, vr, 2 * self.lanczos_width - 1)

        # Throw away the padded ends
        sts = sts[:, pad:-pad].copy()
        ni = ni[:, pad:-pad].copy()
        interp_grid = interp_grid[pad:-pad].copy()

        # Reshape to the correct shape
        sts = sts.reshape(vis_data.shape[:-1] + (self.samples,))
        ni = ni.reshape(vis_data.shape[:-1] + (self.samples,))

        if self.mask_zero_weight:
            # set weights to zero where there is no data
            w_mask = weight.sum(axis=-1) != 0.0
            ni *= w_mask[..., np.newaxis]

        return interp_grid, sts, ni


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
    """Extract a subset of polarisations, including Stokes parameters.

    This currently only extracts Stokes I.

    Attributes
    ----------
    pol : list
        Polarisations to extract. Only Stokes I extraction is supported (i.e. `pol =
        ["I"]`).
    """

    pol = config.Property(proptype=list)

    def process(self, polcont):
        """Extract the specified polarisation from the input.

        This will combine polarisation pairs to get instrumental Stokes polarisations if
        requested.

        Parameters
        ----------
        polcont : ContainerBase
            A container with a polarisation axis.

        Returns
        -------
        selectedpolcont : same as polcont
            A new container with the selected polarisation.
        """
        polcont.redistribute("freq")

        if "pol" not in polcont.axes:
            raise ValueError(
                f"Container of type {type(polcont)} does not have a pol axis."
            )

        if len(self.pol) != 1 or self.pol[0] != "I":
            raise NotImplementedError("Only selecting stokes I is currently working.")

        outcont = containers.empty_like(polcont, pol=np.array(self.pol))
        outcont.redistribute("freq")

        # Get the locations of the XX and YY components
        XX_ind = list(polcont.index_map["pol"]).index("XX")
        YY_ind = list(polcont.index_map["pol"]).index("YY")

        for name, dset in polcont.datasets.items():
            out_dset = outcont.datasets[name]
            if "pol" not in dset.attrs["axis"]:
                out_dset[:] = dset[:]
            else:
                pol_axis_pos = list(dset.attrs["axis"]).index("pol")

                sl = tuple([slice(None)] * pol_axis_pos)
                out_dset[(*sl, 0)] = dset[(*sl, XX_ind)]
                out_dset[(*sl, 0)] += dset[(*sl, YY_ind)]

                if np.issubdtype(out_dset.dtype, np.integer):
                    out_dset[:] //= 2
                else:
                    out_dset[:] *= 0.5

        return outcont


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
        if self.convert_Jy_to_K:
            new_stream.vis[:] *= Jy_to_K
            new_stream.weight[:] *= K_to_Jy**2
        else:
            new_stream.vis[:] *= K_to_Jy
            new_stream.weight[:] *= Jy_to_K**2

        return new_stream


class MixData(task.SingleTask):
    """Mix together pieces of data with specified weights.

    This can generate arbitrary linear combinations of the data and weights for both
    `SiderealStream` and `RingMap` objects, and can be used for many purposes such as:
    adding together simulated timestreams, injecting signal into data, replacing weights
    in simulated data with those from real data, etc.

    All coefficients are applied naively to generate the final combinations, i.e. no
    normalisations or weighted summation is performed.

    Attributes
    ----------
    data_coeff : list
        A list of coefficients to apply to the data dataset of each input containter to
        produce the final output. These are applied to either the `vis` or `map` dataset
        depending on the the type of the input container.
    weight_coeff : list
        Coefficient to be applied to each input containers weights to generate the
        output.
    """

    data_coeff = config.list_type(type_=float)
    weight_coeff = config.list_type(type_=float)

    mixed_data = None

    def setup(self):
        """Check the lists have the same length."""
        if len(self.data_coeff) != len(self.weight_coeff):
            raise config.CaputConfigError(
                "data and weight coefficient lists must be the same length"
            )

        self._data_ind = 0

    def process(self, data: Union[containers.SiderealStream, containers.RingMap]):
        """Add the input data into the mixed data output.

        Parameters
        ----------
        data
            The data to be added into the mix.
        """

        def _get_dset(data):
            # Helpful routine to get the data dset depending on the type
            if isinstance(data, containers.SiderealStream):
                return data.vis

            if isinstance(data, containers.RingMap):
                return data.map

            return None

        if self._data_ind >= len(self.data_coeff):
            raise RuntimeError(
                "This task cannot accept more items than there are coefficents set."
            )

        if self.mixed_data is None:
            self.mixed_data = containers.empty_like(data)
            self.mixed_data.redistribute("freq")

            # Zero out data and weights
            _get_dset(self.mixed_data)[:] = 0.0
            self.mixed_data.weight[:] = 0.0

        # Validate the types are the same
        if type(self.mixed_data) != type(data):
            raise TypeError(
                f"type(data) (={type(data)}) must match "
                f"type(data_stack) (={type(self.type)}"
            )

        data.redistribute("freq")

        mixed_dset = _get_dset(self.mixed_data)[:]
        data_dset = _get_dset(data)[:]

        # Validate the shapes match
        if mixed_dset.shape != data_dset.shape:
            raise ValueError(
                f"Size of data ({data_dset.shape}) must match "
                f"data_stack ({mixed_dset.shape})"
            )

        # Mix in the data and weights
        mixed_dset[:] += self.data_coeff[self._data_ind] * data_dset[:]
        self.mixed_data.weight[:] += self.weight_coeff[self._data_ind] * data.weight[:]

        self._data_ind += 1

    def process_finish(self) -> Union[containers.SiderealStream, containers.RingMap]:
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

        return data


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
            Any datasets not included in the selections will not be initialized.
        """
        # Re-format selections to only use axis name
        for ax_sel in list(self._sel):
            ax = ax_sel.replace("_sel", "")
            self._sel[ax] = self._sel.pop(ax_sel)

        # Figure out the axes for the new container and
        # Apply the downselections to each axis index_map
        output_axes = {
            ax: mpiarray._apply_sel(data.index_map[ax], sel, 0)
            for ax, sel in self._sel.items()
        }
        # Create the output container without initializing any datasets.
        out = data.__class__(
            axes_from=data, attrs_from=data, skip_datasets=True, **output_axes
        )
        containers.copy_datasets_filter(data, out, selection=self._sel)

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
            weight = data.weight[:]
            # The weights should be distributed over the same axis as the array,
            # even if they don't share all the same axes
            new_weight_ax = list(data.weight.attrs["axis"]).index(new_ax_name)
            weight = weight.redistribute(new_weight_ax)
        else:
            self.log.info("No weights available. Using equal weighting.")
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
            out.weight[:] = reduced_weight[:]

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        ws = invert_no_zero(np.sum(weight, axis=axis, keepdims=True))
        # Get the weighted mean
        mu = np.sum(weight * arr, axis=axis, keepdims=True) * ws
        # Get the weighted variance
        v = np.sum(weight * (arr - mu) ** 2, axis=axis, keepdims=True) * ws

        return v, np.ones_like(v)


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

"""Miscellaneous transformations to do on data.

This includes grouping frequencies and products to performing the m-mode transform.
"""
from typing import Optional, Union

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import pandas as pd
import healpy as hp

from caput import mpiarray, config
from ch_util import ephemeris
from cora.util import hputil

from ..core import containers, task, io
from ..util import tools
from ..util import regrid


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
        """

        if self.weight not in ["natural", "uniform", "inverse_variance"]:
            KeyError("Do not recognize weight = %s" % self.weight)

        self.telescope = io.get_telescope(tel)

        # Precalculate the stack properties
        self.bt_stack = np.array(
            [
                (tools.cmap(upp[0], upp[1], self.telescope.nfeed), 0)
                if upp[0] <= upp[1]
                else (tools.cmap(upp[1], upp[0], self.telescope.nfeed), 1)
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

    def process(self, ss):
        """Select and reorder the products.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sp : SiderealStream
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
            ss_conj = np.zeros(ss_prod.size, dtype=np.bool)

        # Create output container
        if isinstance(ss, containers.SiderealStream):
            OutputContainer = containers.SiderealStream
            output_kwargs = {"ra": ss.ra[:]}
        else:
            OutputContainer = containers.TimeStream
            output_kwargs = {"time": ss.time[:]}

        sp = OutputContainer(
            freq=bt_freq,
            input=self.telescope.input_index,
            prod=self.bt_prod,
            stack=self.bt_stack,
            reverse_map_stack=self.bt_rev,
            axes_from=ss,
            attrs_from=ss,
            distributed=True,
            comm=ss.comm,
            **output_kwargs,
        )

        # Add gain dataset.
        # if 'gain' in ss.datasets:
        #     sp.add_dataset('gain')

        # Ensure all frequencies and products are on each node
        ss.redistribute(["ra", "time"])
        sp.redistribute(["ra", "time"])

        # Initialize datasets in output container
        sp.vis[:] = 0.0
        sp.weight[:] = 0.0
        sp.input_flags[:] = ss.input_flags[rev_input_ind, :]

        # The gain transfer below fails when distributed over multiple nodes,
        # have to debug.
        # if 'gain' in ss.datasets:
        #     sp.gain[:] = ss.gain[freq_ind][:, rev_input_ind, :]

        # Infer number of products that went into each stack
        if self.weight != "inverse_variance":

            ssi = ss.input_flags[:]
            ssp = ss.index_map["prod"][:]
            sss = ss.reverse_map["stack"]["stack"][:]
            nstack = ss.vis.shape[1]

            nprod_in_stack = tools.calculate_redundancy(ssi, ssp, sss, nstack)

            if self.weight == "uniform":
                nprod_in_stack = (nprod_in_stack > 0).astype(np.float32)

        # Find the local times (necessary because nprod_in_stack is not distributed)
        ntt = ss.vis.local_shape[-1]
        stt = ss.vis.local_offset[-1]
        ett = stt + ntt

        # Create counter to increment during the stacking.
        # This will be used to normalize at the end.
        counter = np.zeros_like(sp.weight[:])

        # Dereference the global slices now, there's a hidden MPI call in the [:] operation.
        spv = sp.vis[:]
        ssv = ss.vis[:]
        spw = sp.weight[:]
        ssw = ss.weight[:]

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
                wss = ssw[freq_ind, ss_pi]

            else:
                wss = (ssw[freq_ind, ss_pi] > 0.0).astype(np.float32)
                wss *= nprod_in_stack[np.newaxis, ss_pi, stt:ett]

            # Accumulate visibilities, conjugating if required
            if feedconj == conj:
                spv[:, sp_pi] += wss * ssv[freq_ind, ss_pi]
            else:
                spv[:, sp_pi] += wss * ssv[freq_ind, ss_pi].conj()

            # Accumulate variances in quadrature.  Save in the weight dataset.
            spw[:, sp_pi] += wss ** 2 * tools.invert_no_zero(ssw[freq_ind, ss_pi])

            # Increment counter
            counter[:, sp_pi] += wss

        # Divide through by counter to get properly weighted visibility average
        sp.vis[:] *= tools.invert_no_zero(counter)
        sp.weight[:] = counter ** 2 * tools.invert_no_zero(sp.weight[:])

        # Switch back to frequency distribution
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
                set(
                    [
                        np.argmin(np.abs(freq_map["centre"] - freq))
                        for freq in self.freq_physical
                    ]
                )
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
            ValueError(
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

            for name, dset in data.datasets.items():

                if "freq" in dset.attrs["axis"]:
                    slc = [slice(None)] * len(dset.shape)
                    slc[list(dset.attrs["axis"]).index("freq")] = newindex
                    newdata.datasets[name][:] = dset[slc]
                else:
                    newdata.datasets[name][:] = dset[:]

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
    """

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
        weight_sum = nra ** 2 * tools.invert_no_zero(
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
        _make_marray(sstream.vis[:], ma.vis[:])

        # Assign the weights into the container
        ma.weight[:] = weight_sum[np.newaxis, np.newaxis, :, :]

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

    Attributes
    ----------
    nra : int
        Number of RA bins in the output. Note that if the number of samples does not
        Nyquist sample the maximum m, information may be lost. If not set, then try to
        get from an `original_nra` attribute on the incoming MModes, otherwise determine
        an appropriate number of RA bins from the mmax.
    """

    nra = config.Property(proptype=int, default=None)

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
    nra
        The number of RA bins for the output stream.
    """

    nra = config.Property(proptype=int, default=None)


def _make_ssarray(mmodes, n=None):
    # Construct an array of sidereal time streams from m-modes
    marray = _unpack_marray(mmodes, n=n)
    ssarray = np.fft.ifft(marray * marray.shape[-1], axis=-1)

    return ssarray


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
    marray = np.zeros(shape + (ntimes,), dtype=np.complex128)
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
    """

    delta = config.Property(proptype=float)

    def process(
        self, sscont: containers.SiderealContainer
    ) -> containers.SiderealContainer:
        """Add a shift to the input sidereal cont.

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

        sscont.ra[:] += self.delta

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

            if "pol" not in dset.attrs["axis"]:
                outcont.datasets[name][:] = dset[:]
            else:
                pol_axis_pos = list(dset.attrs["axis"]).index("pol")

                sl = tuple([slice(None)] * pol_axis_pos)
                outcont.datasets[name][sl + (0,)] = dset[sl + (XX_ind,)]
                outcont.datasets[name][sl + (0,)] += dset[sl + (YY_ind,)]
                outcont.datasets[name][:] *= 0.5

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
        wavelength = (c.c / (local_freq * 10 ** 6))[:, np.newaxis, np.newaxis]
        K_to_Jy = 2 * 1e26 * c.k * om_ij[:, :, np.newaxis] / wavelength ** 2
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
            new_stream.weight[:] *= K_to_Jy ** 2
        else:
            new_stream.vis[:] *= K_to_Jy
            new_stream.weight[:] *= Jy_to_K ** 2

        return new_stream


class RingMapToHealpixMap(task.SingleTask):
    """Convert an input ringmap to a healpix map.

    Attributes
    ----------
    nside : int
        Nside parameter for output healpix map.
        Recommended value is <=128 to avoid too many blank healpix
        pixels for default CHIME ringmap resolution. Default: 128
    coord_frame : string, optional
        Coordinate frame to use in computing sky positions of healpix pixels.
        Must be one of 'icrs' or 'cirs'. Default: 'cirs'.
    fill_value : float, optional
        Value to fill empty healpix pixels with. Default: NaN
    median_subtract: bool, optional
        Whether to subtract the median across RA from the ringmap before
        converting to healpix. Default: False
    mult_by_weights : bool, optional
        Whether to multiply the ringmap by its (normalized) weights at
        each pol and frequency before converting to healpix. Default: False.
    use_unit_weights : bool, optional
        In mult_by_weights, use unit weights instead of weights taken from
        ringmap. (This can be used to check the impact of the ringmap weights,
        while still accounting for the ringmap's incomplete sky coverage.)
        Default: False.
    filter_weights_m0 : bool
        Filter |m|>0 components out of the weights. Default: False.
    filter_map_with_m_pixwin : bool
        After convering the ringmap to healpix, apply a filter in m space
        corresponding to a top-hat in RA, with width equal to the RA width
        of pixels in the original ringmap. This filter takes the form
        W_m = 2 / (m * dra) sin(m * dra / 2) with dra in rad. Default: False.
    map_nan_to_num : bool, optional
        Convert NaNs in output map to numbers. Default: False.
    """

    nside = config.Property(proptype=int, default=128)
    coord_frame = config.enum(["icrs", "cirs"], default="cirs")
    fill_value = config.Property(proptype=float, default=np.nan)
    median_subtract = config.Property(proptype=bool, default=False)
    mult_by_weights = config.Property(proptype=bool, default=False)
    use_unit_weights = config.Property(proptype=bool, default=False)
    filter_weights_m0 = config.Property(proptype=bool, default=False)
    filter_map_with_m_pixwin = config.Property(proptype=bool, default=False)
    linear_weights_test = config.Property(proptype=bool, default=False)
    map_nan_to_num = config.Property(proptype=bool, default=False)
    
    # Skip NaN checks, because it is likely (and expected) that output
    # map will contain some NaNs
    nan_check = False
    nan_skip = False
    nan_dump = False

    def setup(self, bt):
        """Load the telescope.

        Parameters
        ----------
        bt : ProductManager or BeamTransfer
            Beam Transfer manager.
        """
        self.beamtransfer = io.get_beamtransfer(bt)
        self.telescope = io.get_telescope(bt)

    def process(self, ringmap: containers.RingMap) -> containers.Map:
        """Convert an input ringmap to a healpix map.

        Parameters
        ----------
        ringmap: RingMap
            The ringmap to convert to healpix.

        Returns
        -------
        map_: Map
            The output healpix map.
        """

        # Calculate the epoch for the data so we can calculate the correct
        # CIRS coordinates
        if "lsd" not in ringmap.attrs:
            ringmap.attrs["lsd"] = 1950
            self.log.error("Input must have an LSD attribute to calculate the epoch.")

        # Ensure ringmap is distrubited in frequency
        ringmap.redistribute("freq")
            
        # Convert elevation coordinates in ringnmap to dec
        dec = np.degrees(np.arcsin(ringmap.el)) + self.telescope.latitude

        # Cut elevation/dec values at north celestial pole
        # el_imax = -1e20
        el_imax = np.argmin(np.abs(dec - 90.0))
        sin_za_cut = ringmap.el[:el_imax]
        dec_cut = dec[:el_imax]

        # Map angular sky coordinates in ringmap to healpix pixel coordinates
        hp_pix_coords = self._make_healpix_pixel_coords(ringmap.ra, dec_cut, ringmap.attrs["lsd"], self.nside)

        # Warn that we'll only consider beam 0
        # TODO: add support for >1 beam
        if len(ringmap.index_map["beam"]) > 1:
            self.log.warning(
                "%d beams present in ringmap, but only beam 0 will be converted to healpix" % len(ringmap.index_map["beam"])
            )

        # Make container to store healpix maps
        map_ = containers.Map(
            pixel=hp.nside2npix(self.nside),
            pol=ringmap.pol,
            axes_from=ringmap,
            attrs_from=ringmap,
            distributed=True,
            comm=ringmap.comm,
        )
        map_.add_dataset("weight")
        map_.redistribute("freq")

        # Get local sections of ringmap, weights (if needed late), and healpix map,
        # along with local offset and shape of each frequency section
        ringmap_local = ringmap.map[:]
        weight_local = ringmap.weight[:]
        map_local = map_.map[:]
        map_weight_local = map_.weight[:]
        lo = ringmap.map.local_offset[2]
        ls = ringmap.map.local_shape[2]

        # Loop over frequency
        for fi_local, fi in enumerate(ringmap.freq[lo : lo + ls]):
            self.log.debug(
                "Converting maps for freq = %g MHz" % fi
            )

            for pi in range(len(ringmap.pol)):
                # Fetch map and weights at this frequency and polarization,
                # and transpose to be packed as [el, ra]
                in_map = ringmap_local[0, pi, fi_local].T
                in_weight = weight_local[pi, fi_local].T

                # Cut sin(za) range to be < 90 deg
                in_map = in_map[:el_imax]
                in_weight = in_weight[:el_imax]
                
                # If requested, subtract median at each dec from map
                if self.median_subtract:
                    in_map = self._subtract_median(in_map)

                # Put ringmap pixel values into dataframe, and copy map values into
                # corresponding healpix pixels, averaging ringmap pixels if more than
                # one pixel center falls into a given healpix pixel. Healpix pixels
                # with no corresponding ringmap pixel get fill_value.
                # Note that requesting an nside value too low will result in many blakc
                # pixels - for default CHIME ringmap resolution, nside<= 128 is recommended.
                df = pd.DataFrame({"indices": hp_pix_coords, "data": in_map.flatten()})
                datalist = df.groupby("indices").mean()
                desired_indices = pd.Series(np.arange(hp.nside2npix(self.nside)))
                is_in_index = desired_indices.isin(datalist.index)
                desired_indices[:] = self.fill_value
                desired_indices[is_in_index] = datalist.data
                map_local[fi_local, pi] = np.array(desired_indices)
                if self.map_nan_to_num:
                    map_local[fi_local, pi] = np.nan_to_num(map_local[fi_local, pi])
                # If desired, apply pixel window function corresponding to dra from ringmap
                if self.filter_map_with_m_pixwin:
                    alm = hputil.sphtrans_sky(map_local[fi_local : fi_local+1])
                    m_for_filt = np.arange(alm.shape[-1])
                    dra_for_filt = np.deg2rad(np.median(np.abs(np.diff(ringmap.index_map["ra"]))))
                    x_for_filt = m_for_filt * dra_for_filt / 2
                    w_rm_filt = np.ones_like(x_for_filt)
                    w_rm_filt[1:] = np.sin(x_for_filt[1:]) / x_for_filt[1:]
                    alm *= w_rm_filt[np.newaxis, np.newaxis, np.newaxis, :]
                    map_local[fi_local : fi_local+1] = hputil.sphtrans_inv_sky(alm, self.nside)

                # Do same thing for weights
                if not self.use_unit_weights:
                    df = pd.DataFrame({"indices": hp_pix_coords, "data": in_weight.flatten()})
                else:
                    df = pd.DataFrame({"indices": hp_pix_coords, "data": np.ones_like(in_weight.flatten())})
                datalist = df.groupby("indices").mean()
                desired_indices = pd.Series(np.arange(hp.nside2npix(self.nside)))
                is_in_index = desired_indices.isin(datalist.index)
                desired_indices[:] = self.fill_value
                desired_indices[is_in_index] = datalist.data
                map_weight_local[fi_local, pi] = np.array(desired_indices)
                # Additionally, convert any NaNs to zeros in weights
                map_weight_local[fi_local, pi] = np.nan_to_num(map_weight_local[fi_local, pi])
                # If desired, overwrite weights with weights that are equal to the dec of each source
                # (for testing purposes)
                if self.linear_weights_test:
                    map_weight_local[fi_local, pi][map_weight_local[fi_local, pi] != 0] = hp.pix2ang(
                        self.nside, np.arange(len(map_weight_local[fi_local, pi]))[map_weight_local[fi_local, pi] != 0], lonlat=True
                    )[1]

            # If desired, filter healpix weight map to only have m=0 component
            if self.filter_weights_m0:
                alm = hputil.sphtrans_sky(map_weight_local[fi_local : fi_local+1])
                alm[:, :, :, 1:] = 0
                map_weight_local[fi_local : fi_local+1] = hputil.sphtrans_inv_sky(alm, self.nside)
                
            # If requested, multiply weights into map, normalized by mean of weights
            # (with mean taken in healpix pixelization)
            if self.mult_by_weights:
                map_local[fi_local] *= map_weight_local[fi_local] / map_weight_local[fi_local].mean(axis=1)[:, np.newaxis]
                
        return map_

    def _make_healpix_pixel_coords(self, ra, dec, csd, nside):
        """Convert ringmap pixel coordinates to healpix pixel coordinates.

        Parameters
        ----------
        ra : np.array
            1d array of RA values in ringmap
        dec : np.array
            1d array of dec values in ringmap
        csd : int
            CHIME sidereal day of ringmap
        nside : int
            Healpix nside for output map
        Returns
        -------
        hp_pix_coords : np.array
            Healpix pixel coordinates
        """

        # Make 2d arrays of RA and dec at each map pixel
        nra, ndec = ra.shape[0], dec.shape[0]
        rara = np.ones((ndec, nra)) * ra[np.newaxis, :]
        decdec = np.ones((ndec, nra)) * dec[:, np.newaxis]

        # Make array of sky coordinates at each map pixel, then delete temporary
        # RA, dec arrays
        skycoord = SkyCoord(
            rara.flatten() * u.deg,
            decdec.flatten() * u.deg,
            frame=self.coord_frame,
            obstime=Time(datetime.datetime.fromtimestamp(ephemeris.csd_to_unix(csd))),
        )
        del rara
        del decdec

        # Convert angular sky coordinates to healpix pixel coordinates for specified
        # nside
        hp_pix_coords = hp.ang2pix(
            nside=nside,
            theta=skycoord.icrs.ra.value,
            phi=skycoord.icrs.dec.value,
            lonlat=True,
        )

        return hp_pix_coords


    def _subtract_median(self, in_map, axis=1):
        """Subtract median from ringmap along specified axis.

        Parameters
        ----------
        in_map : np.array, packed as [el, ra]
            The input ringmap

        Returns
        -------
        out_map : np.array, packed as [el, ra]
            Median-subtracted ringmap
        """

        map_ramedian = np.median(in_map, axis=axis, keepdims=True)
        return in_map - map_ramedian


class MSumTable(containers.FreqContainer):
    """A container for holding the output of MapEllMSum.
    """

    _axes = ("pol",)

    _dataset_spec = {
        "msum": {
            "axes": ["freq", "pol"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    def __init__(self, ell, *args, **kwargs):

        super(MSumTable, self).__init__(*args, **kwargs)
        self.attrs["ell"] = ell

    @property
    def msum(self):
        return self.datasets["msum"]


class MapEllMSum(task.SingleTask):
    """Sum a map's spherical harmonic coefficients over m for specific ell.

    Specifically, this task computes
        1 / (4 \pi) * ( a_{\ell 0} + 2 * \sum_{m=1}^\ell Re[a_{\ell m}] )
    for a specific input ell. This sum corresponds to the angular transfer
    function associated with stacking on a source catalog. 

    This task accepts healpix maps and ringmaps: cora's healpy-based routines
    are used for harmonic transforms of the former, while a libsharp-based
    routine is used for the latter.

    Attributes
    ----------
    ell : int
        Spherical harmonic ell of interest.
    nthreads : int
        Number of OpenMP threads to use for SHTs of ringmaps (ignored for
        healpix maps). Default: 1.
    mult_by_weights : bool
        Multiply ringmap by weights (normalized by mean at each frequency)
        before taking SHT (ignored for healpix maps). Default: False.
    use_unit_weights : bool
        Replace ringmap weights by unity for pixels within the horizon,
        if multiplying map by weights before SHT. Ignored in mult_by_weights
        is False or working with a healpix map. Default: False.
    """

    ell = config.Property(proptype=int)

    nthreads = config.Property(proptype=int, default=1)
    mult_by_weights = config.Property(proptype=bool, default=False)
    use_unit_weights = config.Property(proptype=bool, default=False)

    latitude = None
    
    def setup(self, manager: Optional[io.TelescopeConvertible] = None):
        """Set the telescope that determines the observing latitude.

        Parameters
        ----------
        manager : manager.ProductManager, optional
            The telescope/manager used to set the observing latitude.
            Ignored if working with a healpix map.
        """
        if manager is not None:
            tel = io.get_telescope(manager)
            self.latitude = np.deg2rad(tel.latitude)
            
    
    def process(self, map_: Union[containers.Map, containers.RingMap]) -> MSumTable:
        """Take SHT of map and sum a_{ell m} over m for specific ell.

        Parameters
        ----------
        map_: Map or RingMap
            The input healpix map or ringmap.

        Returns
        -------
        msum: MSumTable
            The output sums of spherical harmonic coefficients.
        """

        # Create MSumTable container
        msum = MSumTable(
            ell = self.ell,
            axes_from=map_,
            attrs_from=map_,
            distributed=True,
            comm=map_.comm,
        )

        # Make sure map is distributed in frequency
        map_.redistribute("freq")
        
        # Get local sections of input map and output table
        msum_local = msum.msum[:]
        map_local = map_.map[:]
        
        if isinstance(map_, containers.Map):
        
            nside = hp.npix2nside(len(map_.index_map["pixel"]))
            if self.ell > 3 * nside - 1:
                raise ValueError("Ell cannot be larger than 3*nside-1!")   

            # Compute spherical harmonic coefficients for local map section,
            # packed as [freq, pol, ell, m]
            alm_local = hputil.sphtrans_sky(np.nan_to_num(map_local))

            # Put m sums into container
            msum_local[:] = (
                alm_local[:, :, self.ell, 0]
                + 2 * alm_local[:, :, self.ell, 1:].sum(axis=-1).real
            ) / (4 * np.pi)

        else: # containers.RingMap
            import ducc0

            if self.latitude is None:
                raise ValueError("Must specify telescope to transform a ringmap!")

            # ducc's Gauss-Legendre scheme requires a number of iso-latitude rings
            # equal to lmax+1, so we set our lmax based on the length of the el axis
            lmax = len(map_.el) - 1
            mmax = lmax

            # For the Gauss-Legendre scheme, it is recommended that each ring contain
            # >=2*lmax+1 values, so let's warn the user if this is not satisfied
            if len(map_.ra) < 2*lmax+1:
                self.log.warning(
                    "Warning: ringmap has < 2*lmax+1 RA values, which is not recommended for the ducc SHT"
                )

            # Get indices of m=0 and m>0 elements of alm array
            m0_idx = hp.Alm.getidx(lmax, self.ell, 0)
            mg0_idx = hp.Alm.getidx(lmax, self.ell, np.arange(1, self.ell+1))

            # The outputs of the ducc routines appear to have a different convention
            # than a_lm's from healpix, requiring us to flip the sign of odd-m coefficients
            mg0_signflip = (-1)**(np.arange(1, self.ell+1) % 2)

            # If telescope is not at the equator, we'll need to shift the el axis of
            # the ringmap such that the NCP corresponds to final element of the el axis.
            # We prepare for this by finding the amount to shift by
            if self.latitude != 0.:
                original_za = np.arcsin(map_.el)
                ncp_idx = np.argmin(np.abs(original_za + self.latitude - 0.5 * np.pi))
                el_idx_shift = len(original_za) - 1 - ncp_idx
            else:
                el_idx_shift = 0
            
            # If we'll need the weights, we need to find their mean over frequency and sky
            # at each pol, after shifting the el axis as above
            if self.mult_by_weights:
                weight_local = map_.weight[:]

                # If telescope is not at the equator, translate weights along el axis
                # as described above
                weight_rolled = np.roll(weight_local, el_idx_shift, axis=3)
                weight_rolled[:, :, :, :el_idx_shift] = 0
                # If using unit weights, set weights corresponding to observable part of
                # sky to unity
                if self.use_unit_weights:
                    weight_rolled[:, :, :, el_idx_shift:] = 1

                # We want to normalize the weights by their mean over the sphere, and
                # over frequency.
                # To do so, we multiply the weights by a cos(za) factor that accounts
                # for the polar-angle-dependence of the ringmap pixel solid angle,
                # and further multiply by pi/2. The latter factor can be derived
                # from the expression for the pixel-solid-angle-weighted mean
                # of the ringmap weights: \sum_pix \Omega_pix w_pix / \sum_pix Omega_pix.
                weight_local_sum = 0.5 * np.pi * np.sum(
                    np.cos(np.arcsin(map_.el))[np.newaxis, np.newaxis, np.newaxis, :] * weight_rolled,
                    axis=(2,3)
                )
                weight_global_mean = self.comm.allreduce(weight_local_sum.sum(axis=1))
                weight_global_mean /= np.prod(map_.weight.global_shape[1:])
                
                self.log.info("Multiplying ringmap by normalized weights before taking SHT")
                
            # Need to loop over pol and freq, since ducc routine requires a 3d array
            # where the zeroth axis has length 1 for a spin-0 map.
            for pi, p in enumerate(map_.pol):
                for fi in range(map_local.shape[2]):
                    # Only consider zeroth beam in ringmap, and transpose map input
                    # to be packed as [el, ra]
                    map_for_sht = np.array([map_local[0, pi, fi].view(np.ndarray).T])

                    # If telescope is not at the equator, shift el axis of map such that
                    # NCP corresponds to final element of el axis
                    map_for_sht = np.roll(map_for_sht, el_idx_shift, axis=1)
                    map_for_sht[0, :el_idx_shift] = 0
                    
                    # If desired, multiply by normalized weights
                    if self.mult_by_weights:
                        weight_for_sht = weight_local[pi, fi].view(np.ndarray).T

                        # If telescope is not at the equator, translate weights in same manner
                        # as map
                        weight_for_sht = np.roll(weight_for_sht, el_idx_shift, axis=0)
                        weight_for_sht[:el_idx_shift] = 0
                        if self.use_unit_weights:
                            weight_for_sht[el_idx_shift:] = 1

                        # Multiply weights into map, normalized by mean computed earlier
                        map_for_sht *= weight_for_sht / weight_global_mean[pi]

                    # Do SHT
                    alm_local = ducc0.sht.experimental.analysis_2d(
                        map=map_for_sht,
                        lmax=lmax,
                        mmax=mmax, 
                        spin=0,
                        geometry="GL",
                        nthreads=self.nthreads
                    )

                    # Put m sums into container
                    msum_local[fi, pi] = (
                        alm_local[0, m0_idx].real
                        + 2 * (mg0_signflip * alm_local[0, mg0_idx]).sum().real
                    ) / (4 * np.pi)
            
        return msum


class MultiplyMaps(task.SingleTask):
    """Multiply two healpix maps.

    If the map resolutions differ, the second map is up/downgraded
    to match the first. There is also the option to filter the |m|>0
    components out of the second map, and to normalize the second map
    by its mean.

    Attributes
    ----------
    map2_pol0_only : bool
        Multiply all polarizations of the first map by the unpolarized
        part of the second map. Useful if the second map is a selection
        function, for example. Default: False.
    filter_map2_m0 : bool
        Filter |m|>0 components out of the second map. Default: False.
    mean_normalize_map2 : bool
        Divide map2 by its mean at each frequency prior to m-filtering.
        Default: False.
    nan_to_num : bool
        If either input map contains NaNs, convert to numbers with numpy.
        Default: True.

    """

    map2_pol0_only = config.Property(proptype=bool, default=False)
    filter_map2_m0 = config.Property(proptype=bool, default=False)
    mean_normalize_map2 = config.Property(proptype=bool, default=False)
    nan_to_num = config.Property(proptype=bool, default=True)

    def process(self, map1: containers.Map, map2: containers.Map) -> containers.Map:
        """Multiply maps, processing map 2 if necessary.

        Parameters
        ----------
        map1, map2: Map
            Input healpix maps.

        Returns
        -------
        map_out: Map
            Output healpix map.
        """

        # Get npix and nside of maps
        npix1 = len(map1.index_map["pixel"])
        npix2 = len(map2.index_map["pixel"])
        nside1 = hp.npix2nside(npix1)
        nside2 = hp.npix2nside(npix2)

        # Create container for output map
        out_map = containers.Map(
            pixel=npix1,
            axes_from=map1,
            attrs_from=map1,
            distributed=True,
            comm=map1.comm,
        )

        # Get local sections of maps
        map1_local = map1.map[:]
        map2_local = map2.map[:]
        out_map_local = out_map.map[:]

        # Convert NaNs to numbers, if desired
        if self.nan_to_num:
            map1_local = np.nan_to_num(map1_local)
            map2_local = np.nan_to_num(map2_local)

        # If map resolutions differ, up/downgrade map2 to match map 1.
        # Note: it would probably be better to do this by forward and
        # backward SHTs, rather than at the pixel level with hp.ud_grade(),
        # but ud_grade() seems good enough in practice.
        if nside2 != nside1:
            self.log.debug("Changing map2 resolution from Nside=%d to %d" % (nside2, nside1))
            map2_local_new = np.zeros_like(map1_local)
            for pi in range(map2_local.shape[1]):
                map2_local_new[:, pi] = hp.ud_grade(map2_local[:, pi], nside1)
            map2_local = map2_local_new

        # If only using unpolarized part, replace polarized parts of map2 with that,
        # such that elementwise multiplication with map1 will produce desired result.
        if self.map2_pol0_only:
            self.log.debug("Copying unpolqrized part of map2 into polarized parts")
            for pi in range(1, map2_local.shape[1]):
                map2_local[:, pi] = map2_local[:, 0]

        # If desired, normalize map2 by mean at each frequency
        if self.mean_normalize_map2:
            self.log.debug("Dividing map2 by mean at each frequency")
            map2_local_mean = map2_local.mean(axis=2)
            print(map2_local.mean(axis=2))
            map2_local /= map2_local.mean(axis=2)[:, :, np.newaxis]

        # If desired, filter m>0 components out of map2
        if self.filter_map2_m0:
            self.log.debug("Filtering map2 to pure m=0")
            alm2 = hputil.sphtrans_sky(map2_local)
            alm2[:, :, :, 1:] = 0
            map2_local = hputil.sphtrans_inv_sky(alm2, nside1)

        # Multiply maps together
        out_map_local[:] = map1_local * map2_local

        return out_map


class ModulateMapFrequencies(task.SingleTask):
    """Modulate map or ringmap with frequency oscillations.

    This task constructs a (real) Fourier mode along the frequency
    direction with a given value of k_parallel, and multiplies
    it into the frequency direction of a map or ringmap.

    Parameters
    ----------
    kpar : float
        The k_parallel value to use, interpreted depending on
        how kpar_as_kf_mult is set.
    kpar_as_kf_mult : bool, optional
        If True, kpar is interpreted as a multiple of k_fundmanental,
        defined as 2 * pi / (chi_max - chi_min) where chi_X is
        the comoving radial distance to the lower and upper band
        edges respectively, in Mpc/h. If False, kpar is interpreted
        as k_parallel itself, in h/Mpc. Default: True
    """

    kpar = config.Property(proptype=float, default=None)
    kpar_as_kf_mult = config.Property(proptype=bool, default=True)

    def process(self, map_: containers.FreqContainer) -> containers.FreqContainer:
        """Modulate frequency direction by k_parallel mode.

        Parameters
        ----------
        map_: containers.Map or containers.Ringmap
            Input map.

        Returns
        -------
        out_map : containers.Map or containers.Ringmap
            Output map.
        """
        from draco.synthesis.stream import channel_values_from_kpar

        # Create output container with same type as input
        cont = map_.__class__
        if cont not in [containers.Map, containers.RingMap]:
            raise InputError("Input container must be Map or RingMap!")
        out_cont = cont(
            axes_from=map_,
            attrs_from=map_,
            distributed=True,
            comm=self.comm
        )

        # Construct k_parallel mode, sampled at frequency channel centers
        freq = map_.freq
        mode_vals, kpar_value = channel_values_from_kpar(
            freq, self.kpar, kpar_as_kf_mult=self.kpar_as_kf_mult
        )
        self.log.debug("Modulating map with kpar = %g h/Mpc" % kpar_value)
        # Save kpar to attribute of output container
        out_cont.attrs["kpar"] = kpar_value        
        if self.kpar_as_kf_mult:
            out_cont.attrs["kf"] = kpar_value / self.kpar
        
        if cont is containers.Map:
            # For healpix map, multiply map data by k-mode and copy to
            # new container
            map_.redistribute("pixel")
            out_cont.redistribute("pixel")
            map_local = map_.map[:]
            out_map_local = out_cont.map[:]

            out_map_local[:] = map_local * mode_vals[:, np.newaxis, np.newaxis]

        else:
            # For ringmap, multiply map data by k-mode and copy to
            # new container
            map_.redistribute("el")
            out_cont.redistribute("el")
            map_local = map_.map[:]
            out_map_local = out_cont.map[:]

            out_map_local[:] = map_local * mode_vals[
                np.newaxis, np.newaxis, :, np.newaxis, np.newaxis
            ]

            # Also copy weights, dirty beam, and rms to new container
            out_cont.weight[:] = map_.weight[:]
            if "dirty_beam" in map_.datasets:
                out_cont.add_dataset("dirty_beam")
                out_cont.redistribute("el")
                out_cont.dirty_beam[:] = map_.dirty_beam[:]
            if "rms" in map_.datasets:
                out_cont.add_dataset("rms")
                out_cont.redistribute("el")
                out_cont.rms[:] = map_.rms[:]

        # Other tasks that use the original map will typically assume
        # it is distributed in frequency, so we redistribute here
        map_.redistribute("freq")

        return out_cont

    
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
            elif isinstance(data, containers.RingMap):
                return data.map

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


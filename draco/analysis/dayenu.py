"""DAYENU delay and m-mode filtering.

See https://arxiv.org/abs/2004.11397 for a description.
"""

import time

import numpy as np
import scipy.interpolate
from caput import config
from cora.util import units
from mpi4py import MPI

from ..core import containers, io, task
from ..util import tools
from . import transform


class DayenuDelayFilter(task.SingleTask):
    """Apply a DAYENU high-pass delay filter to visibility data.

    Attributes
    ----------
    za_cut : float
        Sine of the maximum zenith angle included in
        baseline-dependent delay filtering. Default is 1.0,
        which corresponds to the horizon (ie: filters
        out all zenith angles). Setting to zero turns off
        baseline dependent cut.
    telescope_orientation : one of ('NS', 'EW', 'none')
        Determines if the baseline-dependent delay cut is based on
        the north-south component, the east-west component or the full
        baseline length. For cylindrical telescopes oriented in the
        NS direction (like CHIME) use 'NS'. The default is 'NS'.
    epsilon : float
        The stop-band rejection of the filter.  Default is 1e-12.
    tauw : float
        Delay cutoff in micro-seconds.  Default is 0.1 micro-seconds.
    single_mask : bool
        Apply a single frequency mask for all times.  Only includes
        frequencies where the weights are nonzero for all times.
        Otherwise will construct a filter for all unique single-time
        frequency masks (can be significantly slower).  Default is True.
    atten_threshold : float
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    za_cut = config.Property(proptype=float, default=1.0)
    telescope_orientation = config.enum(["NS", "EW", "none"], default="NS")
    epsilon = config.Property(proptype=float, default=1e-12)
    tauw = config.Property(proptype=float, default=0.100)
    single_mask = config.Property(proptype=bool, default=True)
    atten_threshold = config.Property(proptype=float, default=0.0)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

        self.log.info(f"Instrumental delay cut set to {self.tauw:.3f} micro-sec.")

        if self.atten_threshold > 0.0:
            self.log.info(
                "Flagging frequencies with attenuation less "
                f"than {self.atten_threshold:0.2f} of median attenuation."
            )

    def process(self, stream):
        """Filter out delays from a SiderealStream or TimeStream.

        Parameters
        ----------
        stream : SiderealStream
            Data to filter.

        Returns
        -------
        stream_filt : SiderealStream
            Filtered dataset.
        """
        # Distribute over products
        stream.redistribute(["input", "prod", "stack"])

        # Extract the required axes
        freq = stream.freq[:]

        nprod = stream.vis.local_shape[1]
        sp = stream.vis.local_offset[1]
        ep = sp + nprod

        prod = stream.prodstack[sp:ep]

        # Determine the baseline dependent cutoff
        cutoff = self._get_cut(prod)

        # Dereference the required datasets
        vis = stream.vis[:].view(np.ndarray)
        weight = stream.weight[:].view(np.ndarray)

        # Loop over products
        for bb, bcut in enumerate(cutoff):
            t0 = time.time()

            # Flag frequencies and times with zero weight
            flag = weight[:, bb, :] > 0.0

            if self.single_mask:
                flag = np.all(flag, axis=-1, keepdims=True)
                weight[:, bb] *= flag.astype(weight.dtype)

            if not np.any(flag):
                continue

            bvis = np.ascontiguousarray(vis[:, bb])
            bvar = tools.invert_no_zero(weight[:, bb])

            self.log.debug(f"Filter baseline {bb} of {nprod}. [{bcut:0.3f} micro-sec]")

            # Construct the filter
            try:
                NF, index = highpass_delay_filter(
                    freq, bcut, flag, epsilon=self.epsilon
                )

            except np.linalg.LinAlgError as exc:
                self.log.error(
                    f"Failed to converge while processing baseline {bb} "
                    f"[{bcut:0.3f} micro-sec]\n"
                    f"Percentage unmasked frequencies:  {100 * flag.mean():0.1f}\n"
                    f"Exception:  {exc}"
                )
                weight[:, bb] = 0.0
                continue

            # Apply the filter
            if self.single_mask:
                vis[:, bb] = np.matmul(NF[0], bvis)
                weight[:, bb] = tools.invert_no_zero(np.matmul(NF[0] ** 2, bvar))

                if self.atten_threshold > 0.0:
                    diag = np.diag(NF[0])
                    med_diag = np.median(diag[diag > 0.0])

                    flag_low = diag > (self.atten_threshold * med_diag)

                    weight[:, bb] *= flag_low[:, np.newaxis].astype(np.float32)

            else:
                self.log.debug(f"There are {len(index):d} unique masks/filters.")
                for ii, ind in enumerate(index):
                    vis[:, bb, ind] = np.matmul(NF[ii], bvis[:, ind])
                    weight[:, bb, ind] = tools.invert_no_zero(
                        np.matmul(NF[ii] ** 2, bvar[:, ind])
                    )

                    if self.atten_threshold > 0.0:
                        diag = np.diag(NF[ii])
                        med_diag = np.median(diag[diag > 0.0])

                        flag_low = diag > (self.atten_threshold * med_diag)

                        weight[:, bb, ind] *= flag_low[:, np.newaxis].astype(np.float32)

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        return stream

    def _get_cut(self, prod):
        baselines = (
            self.telescope.feedpositions[prod["input_a"], :]
            - self.telescope.feedpositions[prod["input_b"], :]
        )

        if self.telescope_orientation == "NS":
            baselines = abs(baselines[:, 1])  # Y baseline
        elif self.telescope_orientation == "EW":
            baselines = abs(baselines[:, 0])  # X baseline
        else:
            baselines = np.sqrt(np.sum(baselines**2, axis=-1))  # Norm

        baseline_delay_cut = 1e6 * self.za_cut * baselines / units.c

        return baseline_delay_cut + self.tauw


class DayenuDelayFilterFixedCutoff(transform.ReduceChisq):
    """Apply a DAYENU high-pass delay filter to visibility data.

    This task loops over time instead of baseline.  It can be used
    to filter timeseries that have a rapidly changing frequency mask,
    with the caveat that one must use the same delay cutoff for all baselines.

    If reduce_baseline is set to True, then this task will return a
    chi-squared-per-dof test statistic for each frequency and time
    by calculating the weighted average over baselines of the squared
    magnitude of the visibilities.

    Attributes
    ----------
    epsilon : float
        The stop-band rejection of the filter.  Default is 1e-12.
    tauw : float
        Delay cutoff in micro-seconds.  Default is 0.45 micro-seconds.
    single_mask : bool
        Apply a single frequency mask for all baselines.  Only includes
        frequencies where the weights are nonzero for all baselines.
        Otherwise will construct a filter for all unique single-time
        frequency masks (can be significantly slower).  Default is True.
    atten_threshold : float
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    reduce_baseline : bool
        Return chi-squared-per-dof by peforming a weighted average of the
        squared magnitude of the visibilities after applying the filter.
    mask_short : float
        Mask out baselines shorter than a given distance.
    """

    epsilon = config.Property(proptype=float, default=1e-12)
    tauw = config.Property(proptype=float, default=0.450)
    single_mask = config.Property(proptype=bool, default=True)
    atten_threshold = config.Property(proptype=float, default=0.0)

    reduce_baseline = config.Property(proptype=bool, default=False)
    mask_short = config.Property(proptype=float, default=None)

    dataset = "vis"
    axes = ("stack",)

    def setup(self, telescope=None):
        """Set the telescope model.

        Only required if masking short baselines.

        Parameters
        ----------
        telescope : TransitTelescope
            Telescope object containing the baseline distances
            to use for masking.
        """
        self.tel = None if telescope is None else io.get_telescope(telescope)

        if self.tel is None and self.mask_short is not None:
            raise RuntimeError(
                "Must provide telescope object at setup if masking short baselines."
            )

        if not self.reduce_baseline and self.mask_short is not None:
            self.log.warning(
                "You have requested this task mask baselines shorter "
                f"than {self.mask_short:0.1f} meters, eventhough you "
                "are not summing over the baseline axis.  Consider using "
                "task flagging.MaskBaselines instead."
            )

    def process(self, stream):
        """Filter delays below some cutoff.

        If reduce_baseline is False, then this will modify
        the container in place.  If reduce_baseline is True,
        then this will return a new container that has been
        collapsed over the `stack` axis.

        Parameters
        ----------
        stream : TimeStream or SiderealStream
            Raw visibilities.

        Returns
        -------
        stream_filt : TimeStream or SiderealStream
            Filtered visibilities.
        """
        # Distribute over time
        stream.redistribute(["ra", "time"])

        # Extract the required axes
        freq = stream.freq
        ntime = stream.vis.local_shape[2]

        # Create output container
        if self.reduce_baseline:
            out = self._make_output_container(stream)
            out.add_dataset(self.dataset)
            out.redistribute(["ra", "time"])

            # Initialize datasets to zero
            for dset in out.datasets.values():
                dset[:] = 0.0
        else:
            out = stream

        # Dereference the required datasets
        vis = stream.vis[:].local_array
        weight = stream.weight[:].local_array

        ovis = out.vis[:].local_array
        oweight = out.weight[:].local_array

        # Identify baselines that are always flagged
        temp = np.any(weight > 0.0, axis=(0, 2))
        baseline_flag = np.zeros_like(temp)
        self.comm.Allreduce(temp, baseline_flag, op=MPI.LOR)

        # If requested, mask the shortest baselines
        if self.mask_short is not None:
            baseline_flag &= (
                np.sqrt(np.sum(self.tel.baselines**2, axis=1)) >= self.mask_short
            )

            fmask = 1.0 - baseline_flag.mean()
            self.log.info(f"Masking {100 * fmask:0.1f} percent of baselines.")

        # Return if all baselines are masked
        if not np.any(baseline_flag):
            self.log.error("All baselines flagged as bad.")
            return None

        valid = np.flatnonzero(baseline_flag)

        # If we are not outputing a sum over baselines,
        # then make sure the invalid baselines have been masked.
        if not self.reduce_baseline:
            invalid = np.flatnonzero(~baseline_flag)
            oweight[:, invalid, :] = 0.0

        # Loop over time samples
        for tt in range(ntime):
            t0 = time.time()
            self.log.debug(f"Filter time {tt} of {ntime}. [{self.tauw:0.3f} microsec]")

            tweight = weight[:, valid, tt]
            flag = tweight > 0.0

            # Flag frequencies and times with zero weight
            if self.single_mask:
                flag = np.all(flag, axis=-1, keepdims=True)

            if not np.any(flag):
                oweight[:, :, tt] = 0.0
                continue

            # Construct the filter
            try:
                NF, index = highpass_delay_filter(
                    freq, self.tauw, flag, epsilon=self.epsilon
                )

            except np.linalg.LinAlgError as exc:
                self.log.error(
                    f"Failed to converge while processing time {tt}.\n"
                    f"Percentage unmasked frequencies:  {100 * flag.mean():0.1f}\n"
                    f"Exception:  {exc}"
                )
                oweight[:, :, tt] = 0.0
                continue

            # Extract data for filtering
            tvis = np.ascontiguousarray(vis[:, valid, tt])
            tvar = tools.invert_no_zero(tweight)

            tempv = np.zeros_like(tvis)
            tempw = np.zeros_like(tweight)

            # Apply the filter
            self.log.debug(f"There are {len(index)} unique masks/filters.")
            for ii, ind in enumerate(index):
                bind = slice(None) if self.single_mask else ind

                v = np.matmul(NF[ii], tvis[:, bind])
                w = tools.invert_no_zero(np.matmul(NF[ii] ** 2, tvar[:, bind]))

                if self.atten_threshold > 0.0:
                    diag = np.diag(NF[ii])
                    med_diag = np.median(diag[diag > 0.0])

                    flag_low = diag > (self.atten_threshold * med_diag)

                    w *= flag_low[:, np.newaxis].astype(weight.dtype)

                tempv[:, bind] = v
                tempw[:, bind] = w

            # If requested, apply a reduction along baseline axis
            if self.reduce_baseline:
                ovis[:, :, tt], oweight[:, :, tt] = self.reduction(tempv, tempw, 1)
            else:
                ovis[:, valid, tt] = tempv
                oweight[:, valid, tt] = tempw

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        return out


class DayenuDelayFilterHybridVis(task.SingleTask):
    """Apply a DAYENU high-pass delay filter to hybrid beamformed visibilities.

    Attributes
    ----------
    tauw : float or np.ndarray[nstopband,]
        The half width of the stop-band region in micro-seconds.
    tauc : float or np.ndarray[nstopband,]
        The centre of the stop-band region in micro-seconds.
        Defaults to 0.
    epsilon : float or np.ndarray[nstopband,]
        The stop-band rejection.  Defaults to 1e-12.
    atten_threshold : float
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    apply_filter : bool
        Apply the filter that was generated. If False, `save_filter`
        must be True.
    save_filter : bool
        Save the filter that was applied to the output container.
    """

    tauw = config.Property(proptype=np.atleast_1d, default=0.4)
    tauc = config.Property(proptype=np.atleast_1d, default=0.0)
    epsilon = config.Property(proptype=np.atleast_1d, default=1e-12)

    atten_threshold = config.Property(proptype=float, default=0.0)
    apply_filter = config.Property(proptype=bool, default=True)
    save_filter = config.Property(proptype=bool, default=False)

    def setup(self):
        """Check that `save_filter` and `apply_filter` are set."""
        if not self.apply_filter and not self.save_filter:
            raise RuntimeError(
                "At least one of `save_filter` and `apply_filter` must be True."
            )

    def process(self, stream):
        """Filter out delays from a SiderealStream or TimeStream.

        Parameters
        ----------
        stream : HybridVisStream
            Data to filter.

        Returns
        -------
        stream_filt : HybridVisStream
            Filtered dataset.
        """
        # Create a filter dataset
        if self.save_filter:
            if np.any(np.abs(self.tauc) > 0.0):
                stream.add_dataset("complex_filter")
            else:
                stream.add_dataset("filter")
            stream.filter[:] = 0.0

        if not self.apply_filter:
            self.log.debug("Filter will be generated but not applied.")

        # Distribute over products
        stream.redistribute(["ra", "time"])

        # Extract the required axes
        freq = stream.freq[:]

        npol, nfreq, new, nel, ntime = stream.vis.local_shape

        # Dereference the required datasets
        vis = stream.vis[:].local_array
        weight = stream.weight[:].local_array
        if self.save_filter:
            filt = stream.filter[:].local_array

        # Loop over products
        for tt in range(ntime):

            t0 = time.time()

            flag = weight[..., tt] > 0.0
            flag = np.all(flag, axis=0, keepdims=True)

            for xx in range(new):

                self.log.debug(f"Filter time {tt} of {ntime}, baseline {xx} of {new}.")

                flagx = flag[0, :, xx, np.newaxis]
                if not np.any(flagx):
                    continue

                # Construct the filter
                try:
                    NF, index = delay_filter(
                        freq,
                        flagx,
                        tau_width=self.tauw,
                        tau_centre=self.tauc,
                        epsilon=self.epsilon,
                    )

                except np.linalg.LinAlgError as exc:
                    self.log.error(
                        f"Failed to converge while processing time {tt}: {exc}"
                    )
                    if self.apply_filter:
                        weight[:, :, xx, tt] = 0.0
                    continue

                # Apply the filter
                for pp in range(npol):

                    # Save the filter to the container
                    if self.save_filter:
                        filt[pp, :, :, xx, tt] = NF[0]

                    if not self.apply_filter:
                        # Skip the rest
                        continue

                    # Grab datasets for this pol and ew baseline
                    tvis = np.ascontiguousarray(vis[pp, :, xx, :, tt])
                    tvar = tools.invert_no_zero(weight[pp, :, xx, tt])

                    # Apply filter
                    vis[pp, :, xx, :, tt] = np.matmul(NF[0], tvis)
                    weight[pp, :, xx, tt] = tools.invert_no_zero(
                        np.matmul(np.abs(NF[0]) ** 2, tvar)
                    )

                    # Flag frequencies with large attenuation
                    if self.atten_threshold > 0.0:
                        diag = np.abs(np.diag(NF[0]))
                        med_diag = np.median(diag[diag > 0.0])

                        flag_low = diag > (self.atten_threshold * med_diag)

                        weight[pp, :, xx, tt] *= flag_low.astype(weight.dtype)

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # Problems saving to disk when distributed over last axis
        stream.redistribute("freq")

        return stream


class ApplyDelayFilterHybridVis(task.SingleTask):
    """Apply a previously saved filter to the hybrid beamformed visibilities.

    This task takes a DAYENU filter saved in hybrid beamformed data and applies to
    another hybrid beamformed visibilities. This task is used to apply the foreground
    filter to the 21-cm simulation.

    Attributes
    ----------
    atten_threshold : float
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    copy_weight : bool
        If True, do not apply the filter to the weight dataset. Instead,
        copy it directly  from the container used to retrieve the filter.
        Default is False.
    copy_tag : bool
        If True, copy the tag from the container where the filter
        was obtained.  Default is False.
    """

    atten_threshold = config.Property(proptype=float, default=0.0)
    copy_weight = config.Property(proptype=bool, default=False)
    copy_tag = config.Property(proptype=bool, default=False)

    def process(self, hv, source):
        """Apply the DAYENU filter to a HybridVisStream.

        Parameters
        ----------
        hv: containers.HybridVisStream
            The data the filter will be applied to.
        source: containers.HybridVisStream
            The filter of HybridVisStream to be applied.

        Returns
        -------
        hv_filt: containers.HybridVisStream
            The filtered dataset.
        """
        # Distribute over products
        hv.redistribute(["ra", "time"])
        source.redistribute(["ra", "time"])

        # If requested, copy over the tag
        if self.copy_tag:
            hv.attrs["tag"] = source.attrs["tag"]

        # Validate that both hybrid beamformed visibilites match
        if not np.array_equal(source.freq, hv.freq):
            raise ValueError("Frequencies do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["el"], hv.index_map["el"]):
            raise ValueError("Elevations do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["ew"], hv.index_map["ew"]):
            raise ValueError("EW baselines do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["pol"], hv.index_map["pol"]):
            raise ValueError("Polarisations do not match for hybrid visibilities.")

        if not np.array_equal(source.ra, hv.ra):
            raise ValueError("Right Ascension do not match for hybrid visibilities.")

        npol, nfreq, new, nel, ntime = hv.vis.local_shape

        # Dereference the required datasets
        vis = hv.vis[:].local_array
        weight = hv.weight[:].local_array
        filt = source.filter[:].local_array

        # loop over products
        for tt in range(ntime):
            t0 = time.time()
            self.log.debug(f"Filter time {tt} of {ntime}.")

            for xx in range(new):

                for pp in range(npol):

                    flag = weight[pp, :, xx, tt] > 0.0

                    # Skip fully masked samples
                    if not np.any(flag):
                        continue

                    # Grab datasets for this pol and ew baseline
                    tvis = np.ascontiguousarray(vis[pp, :, xx, :, tt])

                    # Grab the filter for this pol and ew baseline
                    NF = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    # Make sure that any frequencies unmasked during filter generation
                    # are also unmasked in the data
                    valid_freq_flag = np.any(np.abs(NF) > 0.0, axis=0)

                    if not np.any(valid_freq_flag):
                        # Skip samples where filter is entirely zero
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    missing_freq = np.flatnonzero(valid_freq_flag & ~flag)
                    if missing_freq.size > 0:
                        self.log.warning(
                            "Missing the following frequencies that were "
                            "assumed valid during filter generation: "
                            f"{missing_freq}"
                        )
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    # Apply the filter
                    vis[pp, :, xx, :, tt] = np.matmul(NF, tvis)

                    # Propagate the weights through the filter
                    if not self.copy_weight:
                        tvar = tools.invert_no_zero(weight[pp, :, xx, tt])

                        weight[pp, :, xx, tt] = tools.invert_no_zero(
                            np.matmul(np.abs(NF) ** 2, tvar)
                        )

                        # Flag frequencies with large attenuation
                        if self.atten_threshold > 0.0:
                            diag = np.abs(np.diag(NF))
                            nonzero_diag_flag = diag > 0.0
                            if np.any(nonzero_diag_flag):
                                med_diag = np.median(diag[nonzero_diag_flag])
                                flag_low = diag > (self.atten_threshold * med_diag)
                                weight[pp, :, xx, tt] *= flag_low.astype(weight.dtype)

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # If requested copy over the weight dataset
        if self.copy_weight:
            weight[:] = source.weight[:].local_array

        # Problems saving to disk when distributed over last axis
        hv.redistribute("freq")

        return hv


class ApplyDelayFilterHybridVisSingleSource(ApplyDelayFilterHybridVis):
    """Apply a previously saved filter to the hybrid beamformed visibilities.

    This task differs from `ApplyDelayFilterHybridVis` in that it applies
    a _single_ filter to multiple possible datasets.
    """

    def setup(self, source):
        """Set the DAYENU filter to be applied to hybrid beamformed data.

        Parameters
        ----------
        source: containers.HybridVisStream
          The filter of HybridVisStream to be applied.
        """
        self.source = source

    def process(self, hv):
        """Apply the DAYENU filter to a HybridVisStream.

        Parameters
        ----------
        hv: containers.HybridVisStream
            The data the filter will be applied to.

        Returns
        -------
        hv_filt: containers.HybridVisStream
            The filtered dataset.
        """
        # Apply the previously saved filter to this dataset
        return super().process(hv, self.source)


class DayenuDelayFilterMap(task.SingleTask):
    """Apply a DAYENU high-pass delay filter to ringmap data.

    Attributes
    ----------
    epsilon : float
        The stop-band rejection of the filter.  Default is 1e-12.
    filename : str
        The name of an hdf5 file containing a DelayCutoff container.
        If a filename is provided, then it will be loaded during setup
        and the `cutoff` dataset will be interpolated to determine
        the cutoff of the filter based on the el coordinate of the map.
    tauw : float
        Delay cutoff in micro-seconds.  If a filename is not provided,
        then tauw will be used as the delay cutoff for all el.
        If a filename is provided, then tauw will be used as the delay
        cutoff for any el that is beyond the range of el contained in
        that file.  Default is 0.1 micro-second.
    single_mask : bool
        Apply a single frequency mask for all times.  Only includes
        frequencies where the weights are nonzero for all times.
        Otherwise will construct a filter for all unique single-time
        frequency masks (can be significantly slower).  Default is True.
    atten_threshold : float
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    epsilon = config.Property(proptype=float, default=1e-12)
    filename = config.Property(proptype=str, default=None)
    tauw = config.Property(proptype=float, default=0.100)
    single_mask = config.Property(proptype=bool, default=True)
    atten_threshold = config.Property(proptype=float, default=0.0)

    _ax_dist = "el"

    def setup(self):
        """Create the function used to determine the delay cutoff."""
        if self.filename is not None:
            fcut = containers.DelayCutoff.from_file(self.filename, distributed=False)
            kind = fcut.attrs.get("kind", "linear")

            self.log.info(
                f"Using {kind} interpolation of the delay cut in the file: "
                f"{self.filename}"
            )

            self._cut_interpolator = {}
            for pp, pol in enumerate(fcut.pol):
                self._cut_interpolator[pol] = scipy.interpolate.interp1d(
                    fcut.el,
                    fcut.cutoff[pp],
                    kind=kind,
                    bounds_error=False,
                    fill_value=self.tauw,
                )

        else:
            self._cut_interpolator = None

        if self.atten_threshold > 0.0:
            self.log.info(
                "Flagging frequencies with attenuation less "
                f"than {self.atten_threshold:0.2f} of median attenuation."
            )

    def process(self, ringmap):
        """Filter out delays from a RingMap.

        Parameters
        ----------
        ringmap : RingMap
            Data to filter.

        Returns
        -------
        ringmap_filt : RingMap
            Filtered data.
        """
        # Distribute over el
        ringmap.redistribute(self._ax_dist)

        # Extract the required axes
        axes = list(ringmap.map.attrs["axis"])
        ax_freq = axes.index("freq")
        ax_dist = axes.index(self._ax_dist)

        lshp = ringmap.map.local_shape[0:ax_freq]

        freq = ringmap.freq[:]

        nel = ringmap.map.local_shape[ax_dist]
        sel = ringmap.map.local_offset[ax_dist]
        eel = sel + nel

        els = ringmap.index_map[self._ax_dist][sel:eel]

        # Dereference the required datasets
        rm = ringmap.map[:].view(np.ndarray)
        weight = ringmap.weight[:].view(np.ndarray)

        # Loop over beam and polarisation
        for ind in np.ndindex(*lshp):
            wind = ind[1:]

            kwargs = {ax: ringmap.index_map[ax][ii] for ax, ii in zip(axes, ind)}

            for ee, el in enumerate(els):
                t0 = time.time()

                slc = (*ind, slice(None), slice(None), ee)
                wslc = slc[1:]

                # Flag frequencies and times with zero weight
                flag = weight[wslc] > 0.0

                if self.single_mask:
                    flag = np.all(flag, axis=-1, keepdims=True)
                    weight[wslc] *= flag.astype(weight.dtype)

                if not np.any(flag):
                    continue

                # Determine the delay cutoff
                ecut = self._get_cut(el, **kwargs)

                self.log.debug(
                    f"Filtering el {el:0.3f}, {ee:d} of {nel:d}. [{ecut:0.3f} micro-sec]"
                )

                erm = np.ascontiguousarray(rm[slc])
                evar = tools.invert_no_zero(weight[wslc])

                # Construct the filter
                try:
                    NF, index = highpass_delay_filter(
                        freq, ecut, flag, epsilon=self.epsilon
                    )

                except np.linalg.LinAlgError as exc:
                    self.log.error(
                        f"Failed to converge while processing el {el:0.3f} "
                        f"[{ecut:0.3f} micro-sec]\n"
                        f"Percentage unmasked frequencies:  {100 * flag.mean():0.1f}\n"
                        f"Exception:  {exc}"
                    )
                    weight[wslc] = 0.0
                    continue

                # Apply the filter
                if self.single_mask:
                    rm[slc] = np.matmul(NF[0], erm)
                    weight[wslc] = tools.invert_no_zero(np.matmul(NF[0] ** 2, evar))

                    if self.atten_threshold > 0.0:
                        diag = np.diag(NF[0])
                        med_diag = np.median(diag[diag > 0.0])

                        flag_low = diag > (self.atten_threshold * med_diag)

                        weight[wslc] *= flag_low[:, np.newaxis].astype(np.float32)

                else:
                    for ii, rr in enumerate(index):
                        rm[ind][:, rr, ee] = np.matmul(NF[ii], erm[:, rr])
                        weight[wind][:, rr, ee] = tools.invert_no_zero(
                            np.matmul(NF[ii] ** 2, evar[:, rr])
                        )

                        if self.atten_threshold > 0.0:
                            diag = np.diag(NF[ii])
                            med_diag = np.median(diag[diag > 0.0])

                            flag_low = diag > (self.atten_threshold * med_diag)

                            weight[wind][:, rr, ee] *= flag_low[:, np.newaxis].astype(
                                np.float32
                            )

                self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # Do this due to a bug in MPI IO when distributed over last axis
        ringmap.redistribute("freq")

        return ringmap

    def _get_cut(self, el, pol=None, **kwargs):
        """Return the delay cutoff in micro-seconds."""
        if self._cut_interpolator is None:
            return self.tauw

        if pol in self._cut_interpolator:
            return self._cut_interpolator[pol](el)

        # The file does not contain this polarisation (likely XY or YX).
        # Use the maximum value over the polarisations that we do have.
        return np.max([func(el) for func in self._cut_interpolator.values()])


class DayenuMFilter(task.SingleTask):
    """Apply a DAYENU bandpass m-mode filter.

    Attributes
    ----------
    dec: float
        The bandpass filter is centered on the m corresponding to the
        fringe rate of a source at the meridian at this declination.
        Default is 40 degrees.
    epsilon : float
        The stop-band rejection of the filter.  Default is 1e-10.
    fkeep_intra : float
        Width of the bandpass filter for intracylinder baselines in terms
        of the fraction of the telescope cylinder width.  Default is 0.75.
    fkeep_inter : float
        Width of the bandpass filter for intercylinder baselines in terms
        of the fraction of the telescope cylinder width.  Default is 0.75.
    """

    dec = config.Property(proptype=float, default=40.0)
    epsilon = config.Property(proptype=float, default=1e-10)
    fkeep_intra = config.Property(proptype=float, default=0.75)
    fkeep_inter = config.Property(proptype=float, default=0.75)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, stream):
        """Filter out m-modes from a SiderealStream or TimeStream.

        Parameters
        ----------
        stream : SiderealStream
            Data to filter.

        Returns
        -------
        stream_filt : SiderealStream
            Filtered dataset.
        """
        # Distribute over products
        stream.redistribute("freq")

        # Extract the required axes
        ra = np.radians(stream.ra[:])

        nfreq = stream.vis.local_shape[0]
        sf = stream.vis.local_offset[0]
        ef = sf + nfreq

        freq = stream.freq[sf:ef]

        # Calculate unique E-W baselines
        prod = stream.prodstack[:]
        baselines = (
            self.telescope.feedpositions[prod["input_a"], 0]
            - self.telescope.feedpositions[prod["input_b"], 0]
        )
        baselines = (
            np.round(baselines / self.telescope.cylinder_spacing)
            * self.telescope.cylinder_spacing
        )

        uniqb, indexb = np.unique(baselines, return_inverse=True)

        db = 0.5 * self.telescope.cylinder_spacing

        # Dereference the required datasets
        vis = stream.vis[:].view(np.ndarray)
        weight = stream.weight[:].view(np.ndarray)

        # Loop over frequencies
        for ff, nu in enumerate(freq):
            t0 = time.time()

            # The next several lines determine the mask as a function of time
            # that is used to construct the filter.
            flag = weight[ff, :, :] > 0.0

            # Identify the valid baselines, i.e., those that have nonzero weight
            # for some fraction of the time.
            gb = np.flatnonzero(np.any(flag, axis=-1))

            if gb.size == 0:
                continue

            # Mask any RA where more than 10 percent of the valid baselines are masked.
            flag = np.sum(flag[gb, :], axis=0, keepdims=True) > (0.90 * float(gb.size))

            weight[ff] *= flag.astype(weight.dtype)

            if not np.any(flag):
                continue

            self.log.debug(f"Filtering freq {ff:d} of {nfreq:d}.")

            # Construct the filters
            m_cut = np.abs(self._get_cut(nu, db))

            m_center_intra = 0.5 * (2.0 - self.fkeep_intra) * m_cut
            m_cut_intra = 0.5 * self.fkeep_intra * m_cut

            m_cut_inter = self.fkeep_inter * m_cut

            INTRA, _ = bandpass_mmode_filter(
                ra, m_center_intra, m_cut_intra, flag, epsilon=self.epsilon
            )
            INTER, _ = lowpass_mmode_filter(ra, m_cut_inter, flag, epsilon=self.epsilon)

            # Loop over E-W baselines
            for uu, ub in enumerate(uniqb):
                iub = np.flatnonzero(indexb == uu)

                visfb = np.ascontiguousarray(vis[ff, iub])

                # Construct the filter
                if np.abs(ub) < db:
                    vis[ff, iub, :] = np.matmul(INTRA, visfb[:, :, np.newaxis])[:, :, 0]

                else:
                    m_center = self._get_cut(nu, ub)
                    mixer = np.exp(-1.0j * m_center * ra)[np.newaxis, :]
                    vis_mixed = visfb * mixer

                    vis[ff, iub, :] = (
                        np.matmul(INTER, vis_mixed[:, :, np.newaxis])[:, :, 0]
                        * mixer.conj()
                    )

            self.log.debug(f"Took {time.time() - t0:0.2f} seconds.")

        return stream

    def _get_cut(self, freq, xsep):
        lmbda = units.c / (freq * 1e6)
        u = xsep / lmbda
        return instantaneous_m(
            0.0, np.radians(self.telescope.latitude), np.radians(self.dec), u, 0.0
        )


def delay_filter(freq, flag, tau_width, tau_centre=0.0, epsilon=1e-12):
    """Construct a delay filter.

    The filter will attenuate signals with delays ranging from
    [tau_centre - tau_width, tau_centre + tau_width].  If more
    than one value of tau_centre and tau_width are provided,
    then the filter will have multiple stop bands.

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequency in MHz.
    flag : np.ndarray[nfreq, ntime]
        Boolean flag that indicates what frequencies are valid
        as a function of time.
    tau_width : float or np.ndarray[nstopband,]
        The half width of the stop-band region in micro-seconds.
    tau_centre : float or np.ndarray[nstopband,]
        The centre of the stop-band region in micro-seconds.
        Defaults to 0.
    epsilon : float or np.ndarray[nstopband,]
        The stop-band rejection.  Defaults to 1e-12.

    Returns
    -------
    pinv : np.ndarray[ntime_uniq, nfreq, nfreq]
        Delay filter for each set of unique frequency flags.
    index : list of length ntime_uniq
        Maps the first axis of pinv to the original time axis.
        Apply pinv[i] to the time samples at index[i].
    """

    # Ensure consistent size for parameter values
    def _ensure_consistent(param, nstopband):
        if np.isscalar(param):
            return [param] * nstopband
        if len(param) == 1:
            return [param[0]] * nstopband
        assert len(param) == nstopband
        return param

    args = [tau_width, tau_centre, epsilon]
    nstopband = np.max([np.atleast_1d(param).size for param in args])
    args = [np.array(_ensure_consistent(param, nstopband)) for param in args]

    # Determine datatype
    dtype = np.complex128 if np.any(np.abs(args[1]) > 0.0) else np.float64

    # Make sure the flag array is properly sized
    ishp = flag.shape
    nfreq = freq.size
    assert ishp[0] == nfreq
    assert len(ishp) == 2

    # Construct the covariance matrix
    dfreq = freq[:, np.newaxis] - freq[np.newaxis, :]

    cov = np.eye(nfreq, dtype=dtype)
    for tw, tc, eps in zip(*args):

        term = np.sinc(2.0 * tw * dfreq) / eps
        if np.abs(tc) > 0.0:
            term = term * np.exp(-2.0j * np.pi * tc * dfreq)

        cov += term

    # Identify unique sets of flags versus frequency
    uflag, uindex = np.unique(flag.reshape(nfreq, -1), return_inverse=True, axis=-1)
    uflag = uflag.T
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]

    # Create a separate covariance matrix for each set of unique flags
    ucov = uflag * cov[np.newaxis, :, :]

    # Peform a pseudo inverse of the masked covariance matrices
    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    index = [np.flatnonzero(uindex == uu) for uu in range(pinv.shape[0])]

    return pinv, index


def highpass_delay_filter(freq, tau_cut, flag, epsilon=1e-12):
    """Construct a high-pass delay filter.

    The stop band will range from [-tau_cut, tau_cut].
    This function is maintained for backwards compatability,
    use delay_filter instead.

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequency in MHz.
    tau_cut : float
        The half width of the stop band in micro-seconds.
    flag : np.ndarray[nfreq, ntime]
        Boolean flag that indicates what frequencies are valid
        as a function of time.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-12.

    Returns
    -------
    pinv : np.ndarray[ntime_uniq, nfreq, nfreq]
        High pass delay filter for each set of unique frequency flags.
    index : list of length ntime_uniq
        Maps the first axis of pinv to the original time axis.
        Apply pinv[i] to the time samples at index[i].
    """
    return delay_filter(freq, flag, tau_cut, 0.0, epsilon)


def bandpass_mmode_filter(ra, m_center, m_cut, flag, epsilon=1e-10):
    """Construct a bandpass m-mode filter.

    The pass band will range from [m_center - m_cut, m_center + m_cut].

    Parameters
    ----------
    ra : np.ndarray[nra,]
        Righ ascension in radians.
    m_center : float
        The center of the pass band.
    m_cut : float
        The half width of the pass band.
    flag : np.ndarray[..., nra]
        Boolean flag that indicates valid right ascensions.
        This must be 2 or more dimensions, with the RA axis last.
        A separate filter will be constructed for each unique set of flags.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-10.

    Returns
    -------
    pinv : np.ndarray[nuniq_flag, nfreq, nfreq]
        Band pass m-mode filter for each set of unique RA flags.
    index : list of length nuniq_flag
        Maps the first axis of pinv to the original flag array.
        Apply pinv[i] to the sub-array at index[i].
    """
    ishp = flag.shape
    nra = ra.size
    assert ishp[-1] == nra

    a = np.median(np.abs(np.diff(ra))) * m_cut / np.pi
    aeps = a * epsilon

    dra = ra[:, np.newaxis] - ra[np.newaxis, :]

    cov = np.eye(nra, dtype=np.float64) / aeps
    cov += (
        2
        * a
        * (1.0 - 1.0 / aeps)
        * np.sinc(m_cut * dra / np.pi)
        * np.cos(m_center * dra)
    )

    uflag, uindex = np.unique(flag.reshape(-1, nra), return_inverse=True, axis=0)
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    index = [
        np.unravel_index(np.flatnonzero(uindex == uu), ishp[:-1])
        for uu in range(pinv.shape[0])
    ]

    return pinv, index


def lowpass_mmode_filter(ra, m_cut, flag, epsilon=1e-10):
    """Construct a low-pass m-mode filter.

    The pass band will range from [-m_cut, m_cut].

    Parameters
    ----------
    ra : np.ndarray[nra,]
        Righ ascension in radians.
    m_cut : float
        The half width of the pass band.
    flag : np.ndarray[..., nra]
        Boolean flag that indicates valid right ascensions.
        This must be 2 or more dimensions, with the RA axis last.
        A separate filter will be constructed for each unique set of flags.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-10.

    Returns
    -------
    pinv : np.ndarray[nuniq_flag, nfreq, nfreq]
        Low pass m-mode filter for each set of unique RA flags.
    index : list of length nuniq_flag
        Maps the first axis of pinv to the original flag array.
        Apply pinv[i] to the sub-array at index[i].
    """
    ishp = flag.shape
    nra = ra.size
    assert ishp[-1] == nra

    a = np.median(np.abs(np.diff(ra))) * m_cut / np.pi
    aeps = a * epsilon

    dra = ra[:, np.newaxis] - ra[np.newaxis, :]

    cov = np.eye(nra, dtype=np.float64) / aeps
    cov += a * (1.0 - 1.0 / aeps) * np.sinc(m_cut * dra / np.pi)

    uflag, uindex = np.unique(flag.reshape(-1, nra), return_inverse=True, axis=0)
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    index = [
        np.unravel_index(np.flatnonzero(uindex == uu), ishp[:-1])
        for uu in range(pinv.shape[0])
    ]

    return pinv, index


def highpass_mmode_filter(ra, m_cut, flag, epsilon=1e-10):
    """Construct a high-pass m-mode filter.

    The stop band will range from [-m_cut, m_cut].

    Parameters
    ----------
    ra : np.ndarray[nra,]
        Righ ascension in radians.
    m_cut : float
        The half width of the stop band.
    flag : np.ndarray[..., nra]
        Boolean flag that indicates valid right ascensions.
        This must be 2 or more dimensions, with the RA axis last.
        A separate filter will be constructed for each unique set of flags.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-10.

    Returns
    -------
    pinv : np.ndarray[nuniq_flag, nfreq, nfreq]
        High pass m-mode filter for each set of unique RA flags.
    index : list of length nuniq_flag
        Maps the first axis of pinv to the original flag array.
        Apply pinv[i] to the sub-array at index[i].
    """
    ishp = flag.shape
    nra = ra.size
    assert ishp[-1] == nra

    dra = ra[:, np.newaxis] - ra[np.newaxis, :]

    cov = np.eye(nra, dtype=np.float64)
    cov += np.sinc(m_cut * dra / np.pi) / epsilon

    uflag, uindex = np.unique(flag.reshape(-1, nra), return_inverse=True, axis=0)
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    index = [
        np.unravel_index(np.flatnonzero(uindex == uu), ishp[:-1])
        for uu in range(pinv.shape[0])
    ]

    return pinv, index


def instantaneous_m(ha, lat, dec, u, v, w=0.0):
    """Calculate the instantaneous fringe-rate.

    Parameters
    ----------
    ha : float
        Hour angle in radians.
    lat : float
        Latitude of the telescope in radians.
    dec : float
        Declination in radians.
    u : float
        EW baseline distance in wavelengths.
    v : float
        NS baseline distance in wavelengths.
    w : float
        Vertical baseline distance in wavelengths.

    Returns
    -------
    m : float
        The fringe-rate of the requested location on the sky
        as measured by the requested baseline.
    """
    deriv = u * (-1 * np.cos(dec) * np.cos(ha))
    deriv += v * (np.sin(lat) * np.cos(dec) * np.sin(ha))
    deriv += w * (-1 * np.cos(lat) * np.cos(dec) * np.sin(ha))

    return 2.0 * np.pi * deriv

"""Tasks for flagging out bad or unwanted data.

This includes data quality flagging on timestream data; sun excision on sidereal
data; and pre-map making flagging on m-modes.

The convention for flagging/masking is `True` for contaminated samples that should
be excluded and `False` for clean samples.
"""

import warnings
from typing import Union, overload

import numpy as np
import scipy.signal
from caput import config, mpiarray, weighted_median

from ..core import containers, io, task
from ..util import rfi, tools


class DayMask(task.SingleTask):
    """Crudely simulate a masking out of the daytime data.

    Attributes
    ----------
    start, end : float
        Start and end of masked out region.
    width : float
        Use a smooth transition of given width between the fully masked and
        unmasked data. This is interior to the region marked by start and end.
    zero_data : bool, optional
        Zero the data in addition to modifying the noise weights
        (default is True).
    remove_average : bool, optional
        Estimate and remove the mean level from each visibilty. This estimate
        does not use data from the masked region.
    """

    start = config.Property(proptype=float, default=90.0)
    end = config.Property(proptype=float, default=270.0)

    width = config.Property(proptype=float, default=60.0)

    zero_data = config.Property(proptype=bool, default=True)
    remove_average = config.Property(proptype=bool, default=True)

    def process(self, sstream):
        """Apply a day time mask.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Unmasked sidereal stack.

        Returns
        -------
        mstream : containers.SiderealStream
            Masked sidereal stream.
        """
        sstream.redistribute("freq")

        ra_shift = (sstream.ra[:] - self.start) % 360.0
        end_shift = (self.end - self.start) % 360.0

        # Crudely mask the on and off regions
        mask_bool = ra_shift > end_shift

        # Put in the transition at the start of the day
        mask = np.where(
            ra_shift < self.width,
            0.5 * (1 + np.cos(np.pi * (ra_shift / self.width))),
            mask_bool,
        )

        # Put the transition at the end of the day
        mask = np.where(
            np.logical_and(ra_shift > end_shift - self.width, ra_shift <= end_shift),
            0.5 * (1 + np.cos(np.pi * ((ra_shift - end_shift) / self.width))),
            mask,
        )

        if self.remove_average:
            # Estimate the mean level from unmasked data
            import scipy.stats

            nanvis = (
                sstream.vis[:]
                * np.where(mask_bool, 1.0, np.nan)[np.newaxis, np.newaxis, :]
            )
            average = scipy.stats.nanmedian(nanvis, axis=-1)[:, :, np.newaxis]
            sstream.vis[:] -= average

        # Apply the mask to the data
        if self.zero_data:
            sstream.vis[:] *= mask

        # Modify the noise weights
        sstream.weight[:] *= mask**2

        return sstream


class MaskMModeData(task.SingleTask):
    """Mask out mmode data ahead of map making.

    Attributes
    ----------
    auto_correlations : bool
        Exclude auto correlations if set (default=False).
    m_zero : bool
        Ignore the m=0 mode (default=False).
    positive_m : bool
        Include positive m-modes (default=True).
    negative_m : bool
        Include negative m-modes (default=True).
    mask_low_m : int, optional
        If set, mask out m's lower than this threshold.
    """

    auto_correlations = config.Property(proptype=bool, default=False)
    m_zero = config.Property(proptype=bool, default=False)
    positive_m = config.Property(proptype=bool, default=True)
    negative_m = config.Property(proptype=bool, default=True)

    mask_low_m = config.Property(proptype=int, default=None)

    def process(self, mmodes):
        """Mask out unwanted datain the m-modes.

        Parameters
        ----------
        mmodes : containers.MModes
            Mmode container to mask

        Returns
        -------
        mmodes : containers.MModes
            Same object as input with masking applied
        """
        mmodes.redistribute("freq")

        mw = mmodes.weight[:]

        # Exclude auto correlations if set
        if not self.auto_correlations:
            for pi, (fi, fj) in enumerate(mmodes.prodstack):
                if fi == fj:
                    mw[..., pi] = 0.0

        # Apply m based masks
        if not self.m_zero:
            mw[0] = 0.0

        if not self.positive_m:
            mw[1:, 0] = 0.0

        if not self.negative_m:
            mw[1:, 1] = 0.0

        if self.mask_low_m:
            mw[: self.mask_low_m] = 0.0

        return mmodes


class MaskBaselines(task.SingleTask):
    """Mask out baselines from a dataset.

    This task may produce output with shared datasets. Be warned that
    this can produce unexpected outputs if not properly taken into
    account.

    Attributes
    ----------
    mask_long_ns : float, optional
        Mask out baselines longer than a given distance in the N/S direction.
    mask_short : float, optional
        Mask out baselines shorter than a given distance.
    mask_short_ew : float, optional
        Mask out baselines shorter then a given distance in the East-West
        direction. Useful for masking out intra-cylinder baselines for
        North-South oriented cylindrical telescopes.
    mask_short_ns : float, optional
        Mask out baselines shorter then a given distance in the North-South
        direction.
    missing_threshold : float, optional
        Mask any baseline that is missing more than this fraction of samples. This is
        measured relative to other baselines.
    zero_data : bool, optional
        Zero the data in addition to modifying the noise weights
        (default is False).
    share : {"all", "none", "vis"}
        Which datasets should we share with the input. If "none" we create a
        full copy of the data, if "vis" we create a copy only of the modified
        weight dataset and the unmodified vis dataset is shared, if "all" we
        modify in place and return the input container.
    """

    mask_long_ns = config.Property(proptype=float, default=None)
    mask_short = config.Property(proptype=float, default=None)
    mask_short_ew = config.Property(proptype=float, default=None)
    mask_short_ns = config.Property(proptype=float, default=None)

    weight_threshold = config.Property(proptype=float, default=None)
    missing_threshold = config.Property(proptype=float, default=None)

    zero_data = config.Property(proptype=bool, default=False)

    share = config.enum(["none", "vis", "all"], default="all")

    def setup(self, telescope):
        """Set the telescope model.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

        if self.zero_data and self.share == "vis":
            self.log.warn(
                "Setting `zero_data = True` and `share = vis` doesn't make much sense."
            )

    def process(self, ss):
        """Apply the mask to data.

        Parameters
        ----------
        ss : SiderealStream or TimeStream
            Data to mask. Applied in place.
        """
        from mpi4py import MPI

        ss.redistribute("freq")

        baselines = self.telescope.baselines

        # The masking array. True will *retain* a sample
        mask = np.zeros_like(ss.weight[:], dtype=bool)

        if self.mask_long_ns is not None:
            long_ns_mask = np.abs(baselines[:, 1]) > self.mask_long_ns
            mask |= long_ns_mask[np.newaxis, :, np.newaxis]

        if self.mask_short is not None:
            short_mask = np.sum(baselines**2, axis=1) ** 0.5 < self.mask_short
            mask |= short_mask[np.newaxis, :, np.newaxis]

        if self.mask_short_ew is not None:
            short_ew_mask = np.abs(baselines[:, 0]) < self.mask_short_ew
            mask |= short_ew_mask[np.newaxis, :, np.newaxis]

        if self.mask_short_ns is not None:
            short_ns_mask = np.abs(baselines[:, 1]) < self.mask_short_ns
            mask |= short_ns_mask[np.newaxis, :, np.newaxis]

        if self.weight_threshold is not None:
            # Get the sum of the weights over frequencies
            weight_sum_local = ss.weight[:].local_array.sum(axis=0)
            weight_sum_tot = np.zeros_like(weight_sum_local)
            self.comm.Allreduce(weight_sum_local, weight_sum_tot, op=MPI.SUM)

            # Retain only baselines with average weights larger than the threshold
            mask |= weight_sum_tot[np.newaxis, :, :] < self.weight_threshold * len(
                ss.freq
            )

        if self.missing_threshold is not None:
            # Get the total number of samples for each baseline accumulated onto each
            # rank
            nsamp_local = (ss.weight[:].local_array > 0).sum(axis=-1).sum(axis=0)
            nsamp_tot = np.zeros_like(nsamp_local)
            self.comm.Allreduce(nsamp_local, nsamp_tot, op=MPI.SUM)

            # Mask out baselines with more that `missing_threshold` samples missing
            baseline_missing_ratio = 1 - nsamp_tot / nsamp_tot.max()
            mask |= (
                baseline_missing_ratio[np.newaxis, :, np.newaxis]
                > self.missing_threshold
            )

        if self.share == "all":
            ssc = ss
        elif self.share == "vis":
            ssc = ss.copy(shared=("vis",))
        else:  # self.share == "none"
            ssc = ss.copy()

        # Apply the mask to the weight
        np.multiply(
            ssc.weight[:].local_array, 0.0, where=mask, out=ssc.weight[:].local_array
        )

        # Apply the mask to the data
        if self.zero_data:
            np.multiply(
                ssc.vis[:].local_array, 0.0, where=mask, out=ssc.vis[:].local_array
            )

        return ssc


class FindBeamformedOutliers(task.SingleTask):
    """Identify beamformed visibilities that deviate from our expectation for noise.

    Attributes
    ----------
    nsigma : float
        Beamformed visibilities whose magnitude is greater than nsigma times
        the expected standard deviation of the noise, given by sqrt(1 / weight),
        will be masked.
    window : list of int
        If provided, the outlier mask will be extended to cover neighboring pixels.
        This list provides the number of pixels in each dimension that a single
        outlier will mask.  Only supported for RingMap containers, where the list
        should be length 2 with [nra, nel], and FormedBeamHA containers, where the list
        should be length 1 with [nha,].
    """

    nsigma = config.Property(proptype=float, default=3.0)
    window = config.Property(proptype=list, default=None)

    def process(self, data):
        """Create a mask that indicates outlier beamformed visibilities.

        Parameters
        ----------
        data : FormedBeam, FormedBeamHA, or RingMap
            Beamformed visibilities.

        Returns
        -------
        out : FormedBeamMask, FormedBeamHAMask, or RingMapMask
            Container with a boolean mask where True indicates
            outlier beamformed visibilities.
        """
        class_dict = {
            containers.FormedBeam: ("beam", containers.FormedBeamMask),
            containers.FormedBeamHA: ("beam", containers.FormedBeamHAMask),
            containers.RingMap: ("map", containers.RingMapMask),
        }

        dataset, out_cont = class_dict[data.__class__]

        # Redistribute data over frequency
        data.redistribute("freq")

        # Make sure the weight dataset has the same
        # number of dimensions as the visibility dataset.
        axes1 = data[dataset].attrs["axis"]
        axes2 = data.weight.attrs["axis"]

        bcast_slice = tuple(slice(None) if ax in axes2 else np.newaxis for ax in axes1)
        axes_collapse = tuple(ii for ii, ax in enumerate(axes1) if ax not in axes2)

        # Calculate the expected standard deviation based on weights dataset
        inv_sigma = np.sqrt(data.weight[:][bcast_slice].view(np.ndarray))

        # Standardize the beamformed visibilities
        ratio = np.abs(data[dataset][:].view(np.ndarray) * inv_sigma)

        # Mask outliers
        mask = ratio > self.nsigma

        if axes_collapse:
            mask = np.any(mask, axis=axes_collapse)

        # Apply a smoothing operation
        if self.window is not None:
            ndim_smooth = len(self.window)
            ndim_iter = mask.ndim - ndim_smooth
            shp = mask.shape[0:ndim_iter]

            msg = ", ".join(
                [
                    f"{axes2[ndim_iter + ww]} [{win}]"
                    for ww, win in enumerate(self.window)
                ]
            )
            self.log.info(f"Extending mask along: axis [num extended] = {msg}")

            kernel = np.ones(tuple(self.window), dtype=np.float32)
            th = 0.5 / kernel.size

            # Loop over the dimensions that are not being convolved
            # to prevent memory errors due to intermediate products
            # created by scipy's convolve.
            mask_extended = np.zeros_like(mask)
            for ind in np.ndindex(*shp):
                mask_extended[ind] = (
                    scipy.signal.convolve(
                        mask[ind].astype(np.float32),
                        kernel,
                        mode="same",
                        method="auto",
                    )
                    > th
                )

            mask = mask_extended

        # Save the mask to a separate container
        out = out_cont(
            axes_from=data,
            attrs_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )
        out.redistribute("freq")
        out.mask[:] = mask

        return out


class MaskBadGains(task.SingleTask):
    """Get a mask of regions with bad gain.

    Assumes that bad gains are set to 1.
    """

    threshold = config.Property(proptype=float, default=1.0)
    threshold_tol = config.Property(proptype=float, default=1e-5)

    def process(self, data):
        """Generate a time-freq mask.

        Parameters
        ----------
        data : :class:`andata.Corrdata` or :class:`container.ContainerBase` with a `gain` dataset
            Data containing the gains to be flagged. Must have a `gain` dataset.

        Returns
        -------
        mask : RFIMask container
            Time-freq mask
        """
        # Ensure data is distributed in frequency
        data.redistribute("freq")

        # Boolean mask where gains are bad across all baselines.
        mask = np.all(
            data.gain[:] <= self.threshold + self.threshold_tol, axis=1
        ).allgather()

        mask_cont = containers.RFIMask(axes_from=data)
        mask_cont.mask[:] = mask

        return mask_cont


class MaskBeamformedOutliers(task.SingleTask):
    """Mask beamformed visibilities that deviate from our expectation for noise.

    This is operating under the assumption that, after proper foreground filtering,
    the beamformed visibilities should be consistent with noise.
    """

    def process(self, data, mask):
        """Mask outlier beamformed visibilities.

        Parameters
        ----------
        data : FormedBeam, FormedBeamHA, or RingMap
            Beamformed visibilities.

        mask : FormedBeamMask, FormedBeamHAMask, or RingMapMask
            Container with a boolean mask where True indicates
            a beamformed visibility that should be ignored.

        Returns
        -------
        data : FormedBeam or RingMap
            The input container with the weight dataset set to zero
            for samples that were identified as outliers.
        """
        # Redistribute data over frequency
        data.redistribute("freq")
        mask.redistribute("freq")

        # Multiply the weights by the inverse of the mask
        flag = ~mask.mask[:].view(np.ndarray)

        data.weight[:] *= flag.astype(np.float32)

        return data


class MaskBeamformedWeights(task.SingleTask):
    """Mask beamformed visibilities with anomalously large weights before stacking.

    Attributes
    ----------
    nmed : float
        Any weight that is more than `nmed` times the median weight
        over all objects and frequencies will be set to zero.
        Default is 8.0.
    """

    nmed = config.Property(proptype=float, default=8.0)

    def process(self, data):
        """Mask large weights.

        Parameters
        ----------
        data : FormedBeam
            Beamformed visibilities.

        Returns
        -------
        data : FormedBeam
            The input container with the weight dataset set to zero
            if the weights exceed the threshold.
        """
        from caput import mpiutil

        data.redistribute("object_id")

        npol = data.pol.size
        med_weight = np.zeros(npol, dtype=np.float32)

        for pp in range(npol):
            wlocal = data.weight[:, pp]
            wglobal = np.zeros(wlocal.global_shape, dtype=wlocal.dtype)

            mpiutil.gather_local(
                wglobal, wlocal, wlocal.local_offset, root=0, comm=data.comm
            )

            if data.comm.rank == 0:
                med_weight[pp] = np.median(wglobal[wglobal > 0])
                self.log.info(
                    f"Median weight for Pol {data.pol[pp]}: {med_weight[pp]:0.2e}"
                )

        # Broadcast the median weight to all ranks
        data.comm.Bcast(med_weight, root=0)

        w = data.weight[:].view(np.ndarray)
        flag = w < (self.nmed * med_weight[np.newaxis, :, np.newaxis])

        data.weight[:] *= flag.astype(np.float32)

        return data


class RadiometerWeight(task.SingleTask):
    r"""Update vis_weight according to the radiometer equation.

    .. math::

        \text{weight}_{ij} = N_\text{samp} / V_{ii} V_{jj}

    Attributes
    ----------
    replace : bool, optional
        Replace any existing weights (default). If `False` then we multiply the
        existing weights by the radiometer values.
    """

    replace = config.Property(proptype=bool, default=True)

    def process(self, stream):
        """Change the vis weight.

        Parameters
        ----------
        stream : SiderealStream or TimeStream
            Data to be weighted. This is done in place.

        Returns
        -------
        stream : SiderealStream or TimeStream
        """
        from caput.time import STELLAR_S

        # Redistribute over the frequency direction
        stream.redistribute("freq")

        ninput = len(stream.index_map["input"])
        nprod = len(stream.index_map["prod"])

        if nprod != (ninput * (ninput + 1) // 2):
            raise RuntimeError(
                "Must have a input stream with the full correlation triangle."
            )

        freq_width = np.median(stream.index_map["freq"]["width"])

        if isinstance(stream, containers.SiderealStream):
            RA_S = 240 * STELLAR_S  # SI seconds in 1 deg of RA change
            int_time = np.median(np.abs(np.diff(stream.ra))) / RA_S
        else:
            int_time = np.median(np.abs(np.diff(stream.index_map["time"])))

        if self.replace:
            stream.weight[:] = 1.0

        # Construct and set the correct weights in place
        nsamp = 1e6 * freq_width * int_time
        autos = tools.extract_diagonal(stream.vis[:]).real
        weight_fac = nsamp**0.5 / autos
        tools.apply_gain(stream.weight[:], weight_fac, out=stream.weight[:])

        # Return timestream with updated weights
        return stream


class SanitizeWeights(task.SingleTask):
    """Flags weights outside of a valid range.

    Flags any weights above a max threshold and below a minimum threshold.
    Baseline dependent, so only some baselines may be flagged.

    Attributes
    ----------
    max_thresh : float
        largest value to keep
    min_thresh : float
        smallest value to keep
    """

    max_thresh = config.Property(proptype=np.float32, default=1e30)
    min_thresh = config.Property(proptype=np.float32, default=1e-30)

    def setup(self):
        """Validate the max and min values.

        Raises
        ------
        ValueError
            if min_thresh is larger than max_thresh
        """
        if self.min_thresh >= self.max_thresh:
            raise ValueError("Minimum threshold is larger than maximum threshold.")

    def process(self, data):
        """Mask any weights outside of the threshold range.

        Parameters
        ----------
        data : :class:`andata.CorrData` or :class:`containers.VisContainer` object
            Data containing the weights to be flagged

        Returns
        -------
        data : same object as data
            Data object with high/low weights masked in-place
        """
        # Ensure data is distributed in frequency
        data.redistribute("freq")

        weight = data.weight[:].local_array

        weight[weight > self.max_thresh] = 0.0
        weight[weight < self.min_thresh] = 0.0

        return data


class SmoothVisWeight(task.SingleTask):
    """Smooth the visibility weights with a median filter.

    This is done in-place.

    Attributes
    ----------
    kernel_size : int, optional
        Size of the kernel for the median filter in time points.
        Default is 31, corresponding to ~5 minutes window for 10s cadence data.
    mask_zeros : bool, optional
        Mask out zero-weight entries when taking the moving weighted median.
    """

    # 31 time points correspond to ~ 5min in 10s cadence
    kernel_size = config.Property(proptype=int, default=31)
    mask_zeros = config.Property(proptype=bool, default=False)

    def process(self, data: containers.TimeStream) -> containers.TimeStream:
        """Smooth the weights with a median filter.

        Parameters
        ----------
        data
            Data containing the weights to be smoothed

        Returns
        -------
        data
            Data object containing the same data as the input, but with the
            weights substituted by the smoothed ones.
        """
        # Ensure data is distributed in frequency,
        # so a frequency loop will not be too large.
        data.redistribute("freq")

        weight_local = data.weight[:].local_array

        for i in range(weight_local.shape[0]):
            # Find values equal to zero to preserve them in final weights
            zeromask = weight_local[i] == 0.0

            # moving_weighted_median wants float64-type weights
            if self.mask_zeros:
                mask = (weight_local[i] > 0.0).astype(np.float64)
            else:
                mask = np.ones_like(weight_local[i], dtype=np.float64)

            weight_local[i] = weighted_median.moving_weighted_median(
                data=weight_local[i],
                weights=mask,
                size=(1, self.kernel_size),
                method="split",
            )

            # Ensure zero values are zero
            weight_local[i][zeromask] = 0.0

        return data


class ThresholdVisWeightFrequency(task.SingleTask):
    """Create a mask to remove all weights below a per-frequency threshold.

    A single relative threshold is set for each frequency along with an absolute
    minimum weight threshold. Masking is done relative to the mean baseline.

    Parameters
    ----------
    absolute_threshold : float
        Any weights with values less than this number will be set to zero.
    relative_threshold : float
        Any weights with values less than this number times the average weight
        will be set to zero.
    """

    absolute_threshold = config.Property(proptype=float, default=1e-7)
    relative_threshold = config.Property(proptype=float, default=0.9)

    def process(self, stream):
        """Make a baseline-independent mask.

        Parameters
        ----------
        stream : `.core.container` with `weight` attribute
            Container to mask

        Returns
        -------
        out : RFIMask container
            RFIMask container with mask set
        """
        stream.redistribute("freq")

        # Make the output container depending on what 'stream' is
        if "ra" in stream.axes:
            mask_cont = containers.SiderealRFIMask(axes_from=stream, attrs_from=stream)
        elif "time" in stream.axes:
            mask_cont = containers.RFIMask(axes_from=stream, attrs_from=stream)
        else:
            raise TypeError(f"Require Timestream or SiderealStream. Got {type(stream)}")

        weight = stream.weight[:].local_array
        # Average over the baselines (stack) axis
        mean_baseline = np.mean(weight, axis=1, keepdims=True)
        # Cut out any values below fixed threhold. Return a np.ndarray
        threshold = np.where(
            mean_baseline > self.absolute_threshold, mean_baseline, np.nan
        )
        # Average across the time (ra) axis to get per-frequency thresholds,
        # ignoring any nans. np.nanmean will give a warning if an entire band is
        # nan, which we expect to happen in some cases.
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="Mean of empty slice")
            threshold = np.nanmean(threshold, axis=2, keepdims=True)
        # Create a 2D baseline-independent mask.
        mask = ~(
            mean_baseline
            > np.fmax(threshold * self.relative_threshold, self.absolute_threshold)
        )[:, 0, :]
        # Collect all parts of the mask. Method .allgather() returns a np.ndarray
        mask = mpiarray.MPIArray.wrap(mask, axis=0).allgather()
        # Log the percent of data masked
        drop_frac = np.sum(mask) / np.prod(mask.shape)
        self.log.info(
            "%0.5f%% of data is below the weight threshold" % (100.0 * (drop_frac))
        )

        mask_cont.mask[:] = mask

        return mask_cont


class ThresholdVisWeightBaseline(task.SingleTask):
    """Form a mask corresponding to weights that are below some threshold.

    The threshold is determined as `maximum(absolute_threshold,
    relative_threshold * average(weight))` and is evaluated per product/stack
    entry. The user can specify whether to use a mean or median as the average,
    but note that the mean is much more likely to be biased by anomalously
    high- or low-weight samples (both of which are present in raw CHIME data).
    The user can also specify that weights below some threshold should not be
    considered when taking the average and constructing the mask (the default
    is to only ignore zero-weight samples).

    The task outputs a BaselineMask or SiderealBaselineMask depending on the
    input container.

    Parameters
    ----------
    average_type : string, optional
        Type of average to use ("median" or "mean"). Default: "median".
    absolute_threshold : float, optional
        Any weights with values less than this number will be set to zero.
        Default: 1e-7.
    relative_threshold : float, optional
        Any weights with values less than this number times the average weight
        will be set to zero. Default: 1e-6.
    ignore_absolute_threshold : float, optional
        Any weights with values less than this number will be ignored when
        taking averages and constructing the mask. Default: 0.0.
    pols_to_flag : string, optional
        Which polarizations to flag. "copol" only flags XX and YY baselines,
        while "all" flags everything. Default: "all".
    """

    average_type = config.enum(["median", "mean"], default="median")
    absolute_threshold = config.Property(proptype=float, default=1e-7)
    relative_threshold = config.Property(proptype=float, default=1e-6)
    ignore_absolute_threshold = config.Property(proptype=float, default=0.0)
    pols_to_flag = config.enum(["all", "copol"], default="all")

    def setup(self, telescope):
        """Set the telescope model.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

    def process(
        self,
        stream,
    ) -> Union[containers.BaselineMask, containers.SiderealBaselineMask]:
        """Construct baseline-dependent mask.

        Parameters
        ----------
        stream : `.core.container` with `weight` attribute
            Input container whose weights are used to construct the mask.

        Returns
        -------
        out : `BaselineMask` or `SiderealBaselineMask`
            The output baseline-dependent mask.
        """
        from mpi4py import MPI

        # Only redistribute the weight dataset, because CorrData containers will have
        # other parallel datasets without a stack axis
        stream.weight.redistribute(axis=1)

        # Make the output container, depending on input type
        if "ra" in stream.axes:
            mask_cont = containers.SiderealBaselineMask(
                axes_from=stream, attrs_from=stream
            )
        elif "time" in stream.axes:
            mask_cont = containers.BaselineMask(axes_from=stream, attrs_from=stream)
        else:
            raise TypeError(
                f"Task requires TimeStream, SiderealStream, or CorrData. Got {type(stream)}"
            )

        # Redistribute output container along stack axis
        mask_cont.redistribute("stack")

        # Get local section of weights
        local_weight = stream.weight[:].local_array

        # For each baseline (axis=1), take average over non-ignored time/freq samples
        average_func = np.ma.median if self.average_type == "median" else np.ma.mean
        average_weight = average_func(
            np.ma.array(
                local_weight, mask=(local_weight <= self.ignore_absolute_threshold)
            ),
            axis=(0, 2),
        ).data

        # Figure out which entries to keep
        threshold = np.maximum(
            self.absolute_threshold, self.relative_threshold * average_weight
        )

        # Compute the mask, excluding samples that we want to ignore
        local_mask = (local_weight < threshold[np.newaxis, :, np.newaxis]) & (
            local_weight > self.ignore_absolute_threshold
        )

        # If only flagging co-pol baselines, make separate mask to select those,
        # and multiply into low-weight mask
        if self.pols_to_flag == "copol":
            # Get local section of stack axis
            local_stack = stream.stack[stream.weight[:].local_bounds]

            # Get product map from input stream
            prod = stream.prod[:]

            # Get polarisation of each input for each element of local stack axis
            pol_a = self.telescope.polarisation[prod[local_stack["prod"]]["input_a"]]
            pol_b = self.telescope.polarisation[prod[local_stack["prod"]]["input_b"]]

            # Make mask to select co-pol baselines
            local_pol_mask = (pol_a == pol_b)[np.newaxis, :, np.newaxis]

            # Apply pol mask to low-weight mask
            local_mask *= local_pol_mask

        # Compute the fraction of data that will be masked
        local_mask_sum = np.sum(local_mask)
        global_mask_total = np.zeros_like(local_mask_sum)
        stream.comm.Allreduce(local_mask_sum, global_mask_total, op=MPI.SUM)
        mask_frac = global_mask_total / float(np.prod(stream.weight.global_shape))

        self.log.info(
            "%0.5f%% of data is below the weight threshold" % (100.0 * mask_frac)
        )

        # Save mask to output container
        mask_cont.mask[:] = mpiarray.MPIArray.wrap(local_mask, axis=1)

        # Distribute back across frequency
        mask_cont.redistribute("freq")
        stream.redistribute("freq")

        return mask_cont


class CollapseBaselineMask(task.SingleTask):
    """Collapse a baseline-dependent mask along the baseline axis.

    The output is a frequency/time mask that is True for any freq/time sample
    for which any baseline is masked in the input mask.
    """

    def process(
        self,
        baseline_mask: Union[containers.BaselineMask, containers.SiderealBaselineMask],
    ) -> Union[containers.RFIMask, containers.SiderealRFIMask]:
        """Collapse input mask over baseline axis.

        Parameters
        ----------
        baseline_mask : `BaselineMask` or `SiderealBaselineMask`
            Input baseline-dependent mask

        Returns
        -------
        mask_cont : `RFIMask` or `SiderealRFIMask`
            Output baseline-independent mask.
        """
        # Redistribute input mask along freq axis
        baseline_mask.redistribute("freq")

        # Make container for output mask. Remember that this will not be distributed.
        if isinstance(baseline_mask, containers.BaselineMask):
            mask_cont = containers.RFIMask(
                axes_from=baseline_mask, attrs_from=baseline_mask
            )
        elif isinstance(baseline_mask, containers.SiderealBaselineMask):
            mask_cont = containers.SiderealRFIMask(
                axes_from=baseline_mask, attrs_from=baseline_mask
            )

        # Get local section of baseline-dependent mask
        local_mask = baseline_mask.mask[:].local_array

        # Collapse along stack axis
        local_mask = np.any(local_mask, axis=1)

        # Gather full mask on each rank
        full_mask = mpiarray.MPIArray.wrap(local_mask, axis=0).allgather()

        # Log the percent of freq/time samples masked
        drop_frac = np.sum(full_mask) / np.prod(full_mask.shape)
        self.log.info(
            f"After baseline collapse: {100.0 * drop_frac:.1f}%% of data"
            " is below the weight threshold"
        )

        mask_cont.mask[:] = full_mask

        return mask_cont


class RFISensitivityMask(task.SingleTask):
    """Slightly less crappy RFI masking.

    Attributes
    ----------
    mask_type : string, optional
        One of 'mad', 'sumthreshold' or 'combine'.
        Default is combine, which uses the sumthreshold everywhere
        except around the transits of the Sun, CasA and CygA where it
        applies the MAD mask to avoid masking out the transits.
    include_pol : list of strings, optional
        The list of polarisations to include. Default is to use all
        polarisations.
    remove_median : bool, optional
        Remove median accross times for each frequency?
        Recommended. Default: True.
    sir : bool, optional
        Apply scale invariant rank (SIR) operator on top of final mask?
        We find that this is advisable while we still haven't flagged
        out all the static bands properly. Default: True.
    sigma : float, optional
        The false positive rate of the flagger given as sigma value assuming
        the non-RFI samples are Gaussian.
        Used for the MAD and TV station flaggers.
    max_m : int, optional
        Maximum size of the SumThreshold window to use.
        The default (8) seems to work well with sensitivity data.
    start_threshold_sigma : float, optional
        The desired threshold for the SumThreshold algorithm at the
        final window size (determined by max m) given as a
        number of standard deviations (to be estimated from the
        sensitivity map excluding weight and static masks).
        The default (8) seems to work well with sensitivity data
        using the default max_m.
    tv_fraction : float, optional
        Number of bad samples in a digital TV channel that cause the whole
        channel to be flagged.
    tv_base_size : [int, int]
        The size of the region used to estimate the baseline for the TV channel
        detection.
    tv_mad_size : [int, int]
        The size of the region used to estimate the MAD for the TV channel detection.
    """

    mask_type = config.enum(["mad", "sumthreshold", "combine"], default="combine")
    include_pol = config.list_type(str, default=None)
    remove_median = config.Property(proptype=bool, default=True)
    sir = config.Property(proptype=bool, default=True)

    sigma = config.Property(proptype=float, default=5.0)
    max_m = config.Property(proptype=int, default=8)
    start_threshold_sigma = config.Property(proptype=float, default=8)

    tv_fraction = config.Property(proptype=float, default=0.5)
    tv_base_size = config.list_type(int, length=2, default=(11, 3))
    tv_mad_size = config.list_type(int, length=2, default=(201, 51))

    def process(self, sensitivity):
        """Derive an RFI mask from sensitivity data.

        Parameters
        ----------
        sensitivity : containers.SystemSensitivity
            Sensitivity data to derive the RFI mask from.

        Returns
        -------
        rfimask : containers.RFIMask
            RFI mask derived from sensitivity.
        """
        ## Constants
        # Convert MAD to RMS
        MAD_TO_RMS = 1.4826

        # The difference between the exponents in the usual
        # scaling of the RMS (n**0.5) and the scaling used
        # in the sumthreshold algorithm (n**log2(1.5))
        RMS_SCALING_DIFF = np.log2(1.5) - 0.5

        # Distribute over polarisation as we need all times and frequencies
        # available simultaneously
        sensitivity.redistribute("pol")

        # Divide sensitivity to get a radiometer test
        radiometer = sensitivity.measured[:] * tools.invert_no_zero(
            sensitivity.radiometer[:]
        )
        radiometer = mpiarray.MPIArray.wrap(radiometer, axis=1)

        freq = sensitivity.freq
        npol = len(sensitivity.pol)
        nfreq = len(freq)

        static_flag = ~self._static_rfi_mask_hook(freq, sensitivity.time[0])

        madmask = mpiarray.MPIArray(
            (npol, nfreq, len(sensitivity.time)), axis=0, dtype=bool
        )
        madmask[:] = False
        stmask = mpiarray.MPIArray(
            (npol, nfreq, len(sensitivity.time)), axis=0, dtype=bool
        )
        stmask[:] = False

        for li, ii in madmask.enumerate(axis=0):
            # Only process this polarisation if we should be including it,
            # otherwise skip and let it be implicitly set to False (i.e. not
            # masked)
            if self.include_pol and sensitivity.pol[ii] not in self.include_pol:
                continue

            # Initial flag on weights equal to zero.
            origflag = sensitivity.weight[:, ii] == 0.0

            # Remove median at each frequency, if asked.
            if self.remove_median:
                for ff in range(nfreq):
                    radiometer[ff, li] -= np.median(
                        radiometer[ff, li][~origflag[ff]].view(np.ndarray)
                    )

            # Combine weights with static flag
            start_flag = origflag | static_flag[:, None]

            # Obtain MAD and TV masks
            this_madmask, tvmask = self._mad_tv_mask(
                radiometer[:, li], start_flag, freq
            )

            # combine MAD and TV masks
            madmask[li] = this_madmask | tvmask

            # Add TV channels to ST start flag.
            start_flag = start_flag | tvmask

            # Determine initial threshold
            med = np.median(radiometer[:, li][~start_flag].view(np.ndarray))
            mad = np.median(abs(radiometer[:, li][~start_flag].view(np.ndarray) - med))
            threshold1 = (
                mad
                * MAD_TO_RMS
                * self.start_threshold_sigma
                * self.max_m**RMS_SCALING_DIFF
            )

            # SumThreshold mask
            stmask[li] = rfi.sumthreshold(
                radiometer[:, li],
                self.max_m,
                start_flag=start_flag,
                threshold1=threshold1,
                correct_for_missing=True,
            )

        # Perform an OR (.any) along the pol axis and reform into an MPIArray
        # along the freq axis
        madmask = mpiarray.MPIArray.wrap(madmask.redistribute(1).any(0), 0)
        stmask = mpiarray.MPIArray.wrap(stmask.redistribute(1).any(0), 0)

        # Pick which of the MAD or SumThreshold mask to use (or blend them)
        if self.mask_type == "mad":
            finalmask = madmask

        elif self.mask_type == "sumthreshold":
            finalmask = stmask

        else:
            # Combine ST and MAD masks
            madtimes = self._combine_st_mad_hook(sensitivity.time)
            finalmask = stmask
            finalmask[:, madtimes] = madmask[:, madtimes]

        # Collect all parts of the mask onto rank 1 and then broadcast to all ranks
        finalmask = mpiarray.MPIArray.wrap(finalmask, 0).allgather()

        # Apply scale invariant rank (SIR) operator, if asked for.
        if self.sir:
            finalmask = self._apply_sir(finalmask, static_flag)

        # Create container to hold mask
        rfimask = containers.RFIMask(axes_from=sensitivity)
        rfimask.mask[:] = finalmask

        return rfimask

    def _combine_st_mad_hook(self, times):
        """Override to add a custom blending mask between the SumThreshold and MAD flagged data.

        This is useful to use the MAD algorithm around bright source
        transits, where the SumThreshold begins to remove real signal.

        Parameters
        ----------
        times : np.ndarray[ntime]
            Times of the data at floating point UNIX time.

        Returns
        -------
        combine : np.ndarray[ntime]
            Mixing array as a function of time. If `True` that sample will be
            filled from the MAD, if `False` use the SumThreshold algorithm.
        """
        return np.ones_like(times, dtype=bool)

    def _static_rfi_mask_hook(self, freq, timestamp=None):
        """Override this function to apply a static RFI mask to the data.

        Parameters
        ----------
        freq : np.ndarray[nfreq]
            1D array of frequencies in the data (in MHz).
        timestamp : float or np.ndarray[ntimes]
            timestamps to use when determining the static mask for this datatset

        Returns
        -------
        mask : np.ndarray[nfreq]
            Mask array. True will include a frequency channel, False masks it out.
        """
        return np.ones_like(freq, dtype=bool)

    def _apply_sir(self, mask, baseflag, eta=0.2):
        """Expand the mask with SIR."""
        # Remove baseflag from mask and run SIR
        nobaseflag = np.copy(mask)
        nobaseflag[baseflag] = False
        nobaseflagsir = rfi.sir(nobaseflag[:, np.newaxis, :], eta=eta)[:, 0, :]

        # Make sure the original mask (including baseflag) is still masked
        return nobaseflagsir | mask

    def _mad_tv_mask(self, data, start_flag, freq):
        """Use the specific scattered TV channel flagging."""
        # Make copy of data
        data = np.copy(data)

        # Calculate the scaled deviations
        data[start_flag] = 0.0
        maddev = mad(
            data, start_flag, base_size=self.tv_base_size, mad_size=self.tv_mad_size
        )

        # Replace any NaNs (where too much data is missing) with a
        # large enough value to always be flagged
        maddev = np.where(np.isnan(maddev), 2 * self.sigma, maddev)

        # Reflag for scattered TV emission
        tvmask = tv_channels_flag(maddev, freq, sigma=self.sigma, f=self.tv_fraction)

        # Create MAD mask
        madmask = maddev > self.sigma

        # Ensure start flag is masked
        madmask = madmask | start_flag

        return madmask, tvmask


class RFIMask(task.SingleTask):
    """Crappy RFI masking.

    Attributes
    ----------
    sigma : float, optional
        The false positive rate of the flagger given as sigma value assuming
        the non-RFI samples are Gaussian.
    tv_fraction : float, optional
        Number of bad samples in a digital TV channel that cause the whole
        channel to be flagged.
    stack_ind : int
        Which stack to process to derive flags for the whole dataset.
    """

    sigma = config.Property(proptype=float, default=5.0)
    tv_fraction = config.Property(proptype=float, default=0.5)
    stack_ind = config.Property(proptype=int)

    @overload
    def process(
        self, sstream: containers.SiderealStream
    ) -> containers.SiderealRFIMask: ...

    @overload
    def process(self, sstream: containers.TimeStream) -> containers.RFIMask: ...

    def process(
        self, sstream: Union[containers.TimeStream, containers.SiderealStream]
    ) -> Union[containers.RFIMask, containers.SiderealRFIMask]:
        """Apply a day time mask.

        Parameters
        ----------
        sstream
            Unmasked sidereal or time stream visibility data.

        Returns
        -------
        mask
            The derived RFI mask.
        """
        # Select the correct mask type depending on if we have sidereal data or not
        output_type = (
            containers.SiderealRFIMask
            if "ra" in sstream.index_map
            else containers.RFIMask
        )

        sstream.redistribute(["stack", "prod"])

        ssv = sstream.vis[:]
        ssw = sstream.weight[:]

        # Figure out which rank actually has the requested index
        lstart = ssv.local_offset[1]
        lstop = lstart + ssv.local_shape[1]
        has_ind = (self.stack_ind >= lstart) and (self.stack_ind < lstop)
        has_ind_list = sstream.comm.allgather(has_ind)
        rank_with_ind = has_ind_list.index(True)
        self.log.debug(
            "Rank %i has the requested index %i", rank_with_ind, self.stack_ind
        )

        mask_cont = output_type(copy_from=sstream)
        mask = mask_cont.mask[:]

        # Get the rank with stack to create the new mask
        if sstream.comm.rank == rank_with_ind:
            # Cut out the right section
            wf = ssv.local_array[:, self.stack_ind - lstart]
            ww = ssw.local_array[:, self.stack_ind - lstart]

            # Generate an initial mask and calculate the scaled deviations
            # TODO: replace this magic threshold
            weight_cut = 1e-4 * ww.mean()  # Ignore samples with small weights
            wm = ww < weight_cut
            maddev = mad(wf, wm)

            # Replace any NaNs (where too much data is missing) with a large enough
            # value to always be flagged
            maddev = np.where(np.isnan(maddev), 2 * self.sigma, maddev)

            # Reflag for scattered TV emission
            tvmask = tv_channels_flag(
                maddev, sstream.freq, sigma=self.sigma, f=self.tv_fraction
            )

            # Construct the new mask
            mask[:] = tvmask | (maddev > self.sigma)

        # Broadcast the new flags to all ranks and then apply
        sstream.comm.Bcast(mask, root=rank_with_ind)

        self.log.info(
            "Flagging %0.2f%% of data due to RFI."
            % (100.0 * np.sum(mask) / float(mask.size))
        )

        return mask_cont


class ApplyTimeFreqMask(task.SingleTask):
    """Apply a time-frequency mask to the data.

    Typically this is used to ask out all inputs at times and
    frequencies contaminated by RFI.

    This task may produce output with shared datasets. Be warned that
    this can produce unexpected outputs if not properly taken into
    account.

    Attributes
    ----------
    share : {"all", "none", "vis", "map"}
        Which datasets should we share with the input. If "none" we create a
        full copy of the data, if "vis" or "map" we create a copy only of the modified
        weight dataset and the unmodified vis dataset is shared, if "all" we
        modify in place and return the input container.
    """

    share = config.enum(["none", "vis", "map", "all"], default="all")

    def process(self, tstream, rfimask):
        """Apply the mask by zeroing the weights.

        Parameters
        ----------
        tstream : timestream or sidereal stream
            A timestream or sidereal stream like container. For example,
            `containers.TimeStream`, `andata.CorrData` or
            `containers.SiderealStream`.
        rfimask : containers.RFIMask
            An RFI mask for the same period of time.

        Returns
        -------
        tstream : timestream or sidereal stream
            The masked timestream. Note that the masking is done in place.
        """
        if isinstance(rfimask, containers.RFIMask):
            if not hasattr(tstream, "time"):
                raise TypeError(
                    f"Expected a timestream like type. Got {type(tstream)}."
                )
            # Validate the time axes match
            if not np.array_equal(tstream.time, rfimask.time):
                raise ValueError("timestream and mask data have different time axes.")

        elif isinstance(rfimask, containers.SiderealRFIMask):
            if not hasattr(tstream, "ra"):
                raise TypeError(
                    f"Expected a sidereal stream like type. Got {type(tstream)}."
                )
            # Validate the RA axes match
            if not np.array_equal(tstream.ra, rfimask.ra):
                raise ValueError("timestream and mask data have different RA axes.")

        else:
            raise TypeError(
                f"Require a RFIMask or SiderealRFIMask. Got {type(rfimask)}."
            )

        # Validate the frequency axis
        if not np.array_equal(tstream.freq, rfimask.freq):
            raise ValueError("timestream and mask data have different freq axes.")

        # Ensure we are frequency distributed
        tstream.redistribute("freq")

        # Create a slice that broadcasts the mask to the final shape
        t_axes = tstream.weight.attrs["axis"]
        m_axes = rfimask.mask.attrs["axis"]
        bcast_slice = tuple(
            slice(None) if ax in m_axes else np.newaxis for ax in t_axes
        )

        # RFI Mask is not distributed, so we need to cut out the frequencies
        # that are local for the tstream
        ax = list(t_axes).index("freq")
        sf = tstream.weight.local_offset[ax]
        ef = sf + tstream.weight.local_shape[ax]

        if self.share == "all":
            tsc = tstream
        elif self.share == "vis":
            tsc = tstream.copy(shared=("vis",))
        elif self.share == "map":
            tsc = tstream.copy(shared=("map",))
        else:  # self.share == "none"
            tsc = tstream.copy()

        # Mask the data.
        tsc.weight[:].local_array[:] *= (~rfimask.mask[sf:ef][bcast_slice]).astype(
            np.float32
        )

        return tsc


class ApplyBaselineMask(task.SingleTask):
    """Apply a distributed mask that varies across baselines.

    No broadcasting is done, so the data and mask should have the same
    axes. This shouldn't be used for non-distributed time-freq masks.

    This task may produce output with shared datasets. Be warned that
    this can produce unexpected outputs if not properly taken into
    account.

    Attributes
    ----------
    share : {"all", "none", "vis", "map"}
        Which datasets should we share with the input. If "none" we create a
        full copy of the data, if "vis" or "map" we create a copy only of the modified
        weight dataset and the unmodified vis dataset is shared, if "all" we
        modify in place and return the input container.
    """

    share = config.enum(["none", "vis", "map", "all"], default="all")

    @overload
    def process(
        self, data: containers.TimeStream, mask: containers.BaselineMask
    ) -> containers.TimeStream: ...

    @overload
    def process(
        self, data: containers.SiderealStream, mask: containers.SiderealBaselineMask
    ) -> containers.SiderealStream: ...

    def process(self, data, mask):
        """Flag data by zeroing the weights.

        Parameters
        ----------
        data
            Data to apply mask to. Must have a stack axis
        mask
            A baseline-dependent mask

        Returns
        -------
        data
            The masked data. Masking is done in place.
        """
        if isinstance(mask, containers.BaselineMask):
            if not hasattr(data, "time"):
                raise TypeError(f"Expected a timestream-like type. Got {type(data)}.")

            if not (data.time.shape == mask.time.shape) and np.allclose(
                data.time, mask.time
            ):
                raise ValueError("timestream and mask have different time axes.")

        elif isinstance(mask, containers.SiderealBaselineMask):
            if not hasattr(data, "ra"):
                raise TypeError(
                    f"Expected a sidereal stream like type. Got {type(data)}."
                )

            if not (data.ra.shape == mask.ra.shape) and np.allclose(data.ra, mask.ra):
                raise ValueError("sidereal stream and mask have different RA axes.")

        else:
            raise TypeError(
                f"Require a BaselineMask or SiderealBaselineMask. Got {type(mask)}."
            )

        # Validate remaining axes
        if not np.array_equal(data.stack, mask.stack):
            raise ValueError("data and mask have different baseline axes.")

        if not (data.freq.shape == mask.freq.shape) and np.allclose(
            data.freq, mask.freq
        ):
            raise ValueError("data and mask have different freq axes.")

        if self.share == "all":
            tsc = data
        elif self.share == "vis":
            tsc = data.copy(shared=("vis",))
        elif self.share == "map":
            tsc = data.copy(shared=("map",))
        else:
            tsc = data.copy()

        tsc.weight[:] *= (~mask.mask[:]).astype(np.float32)

        return tsc


class MaskFreq(task.SingleTask):
    """Make a mask for certain frequencies.

    Attributes
    ----------
    bad_freq_ind : list, optional
        A list containing frequencies to flag out. Each entry can either be an
        integer giving an individual frequency index to remove, or 2-tuples giving
        start and end indices of a range to flag (as with a standard slice, the end
        is *not* included.)
    factorize : bool, optional
        Find the smallest factorizable mask of the time-frequency axis that covers all
        samples already flagged in the data.
    all_time : bool, optional
        Only include frequencies where all time samples are present.
    mask_missing_data : bool, optional
        Mask time-freq samples where some baselines (for visibily data) or
        polarisations/elevations (for ring map data) are missing.
    """

    bad_freq_ind = config.Property(proptype=list, default=None)
    factorize = config.Property(proptype=bool, default=False)
    all_time = config.Property(proptype=bool, default=False)
    mask_missing_data = config.Property(proptype=bool, default=False)

    def process(
        self, data: Union[containers.VisContainer, containers.RingMap]
    ) -> Union[containers.RFIMask, containers.SiderealRFIMask]:
        """Make the mask.

        Parameters
        ----------
        data
            The data to mask.

        Returns
        -------
        mask_cont
            Frequency mask container
        """
        data.redistribute("freq")

        maskcls = (
            containers.SiderealRFIMask
            if isinstance(data, containers.SiderealContainer)
            else containers.RFIMask
        )
        maskcont = maskcls(axes_from=data, attrs_from=data)
        mask = maskcont.mask[:]

        # Get the total number of amount of data for each freq-time. This is used to
        # create an initial mask. For visibilities find the number of baselines
        # present...
        if isinstance(data, containers.VisContainer):
            present_data = mpiarray.MPIArray.wrap(
                (data.weight[:] > 0).sum(axis=1), comm=data.weight.comm, axis=0
            )
        # ... for ringmaps find the number of polarisations/elevations present
        elif isinstance(data, containers.RingMap):
            present_data = mpiarray.MPIArray.wrap(
                (data.weight[:] > 0).sum(axis=3).sum(axis=0),
                comm=data.weight.comm,
                axis=0,
            )
        else:
            raise ValueError(
                f"Received data of type {data._class__}. "
                "Only visibility type data and ringmaps are supported."
            )

        all_present_data = present_data.allgather()
        mask[:] = all_present_data == 0

        self.log.info(f"Input data: {100.0 * mask.mean():.2f}% flagged.")

        # Create an initial mask of the freq-time space, where bad samples are
        # True. If `mask_missing_data` is set this masks any sample where the amount
        # of present data is less than the maximum, otherwise it is where all
        # data is missing
        if self.mask_missing_data:
            mask = all_present_data < all_present_data.max()
            self.log.info(
                f"Requiring all baselines: {100.0 * mask.mean():.2f}% flagged."
            )

        if self.bad_freq_ind is not None:
            nfreq = len(data.freq)
            mask |= self._bad_freq_mask(nfreq)[:, np.newaxis]
            self.log.info(f"Frequency mask: {100.0 * mask.mean():.2f}% flagged.")

        if self.all_time:
            mask |= mask.any(axis=1)[:, np.newaxis]
            self.log.info(f"All time mask: {100.0 * mask.mean():.2f}% flagged.")
        elif self.factorize:
            mask[:] = self._optimal_mask(mask)
            self.log.info(f"Factorizable mask: {100.0 * mask.mean():.2f}% flagged.")

        return maskcont

    def _bad_freq_mask(self, nfreq: int) -> np.ndarray:
        # Parse the bad frequency list to create a per frequency mask

        mask = np.zeros(nfreq, dtype=bool)

        for s in self.bad_freq_ind:
            if isinstance(s, int):
                if s < nfreq:
                    mask[s] = True
            elif isinstance(s, (tuple, list)) and len(s) == 2:
                mask[s[0] : s[1]] = True
            else:
                raise ValueError(
                    "Elements of `bad_freq_ind` must be integers or 2-tuples. "
                    f"Got {type(s)}."
                )

        return mask

    def _optimal_mask(self, mask: np.ndarray) -> np.ndarray:
        # From the freq-time input mask, create the smallest factorizable mask that
        # covers all the original masked samples

        from scipy.optimize import minimize_scalar

        def genmask(f):
            # Calculate a factorisable mask given the time masking threshold f
            time_mask = mask.mean(axis=0) > f
            freq_mask = mask[:, ~time_mask].any(axis=1)
            return time_mask[np.newaxis, :] | freq_mask[:, np.newaxis]

        def fmask(f):
            # Calculate the total area masked given f
            m = genmask(f).mean()
            self.log.info(f"Current value: {m}")
            return m

        # Solve to find a value of f that minimises the amount of data masked
        res = minimize_scalar(
            fmask, method="golden", options={"maxiter": 20, "xtol": 1e-2}
        )

        if not res.success:
            self.log.info("Optimisation did not converge, but this isn't unexpected.")

        return genmask(res.x)


class BlendStack(task.SingleTask):
    """Mix a small amount of a stack into data to regularise RFI gaps.

    This is designed to mix in a small amount of a stack into a day of data (which
    will have RFI masked gaps) to attempt to regularise operations which struggle to
    deal with time variable masks, e.g. `DelaySpectrumEstimator`.

    Attributes
    ----------
    frac : float, optional
        The relative weight to give the stack in the average. This multiplies the
        weights already in the stack, and so it should be remembered that these may
        already be significantly higher than the single day weights.
    match_median : bool, optional
        Estimate the median in the time/RA direction from the common samples and use
        this to match any quasi time-independent bias of the data (e.g. cross talk).
    subtract : bool, optional
        Rather than taking an average, instead subtract out the blending stack
        from the input data in the common samples to calculate the difference
        between them. The interpretation of `frac` is a scaling of the inverse
        variance of the stack to an inverse variance of a prior on the
        difference, e.g. a `frac = 1e-4` means that we expect the standard
        deviation of the difference between the data and the stacked data to be
        100x larger than the noise of the stacked data.
    """

    frac = config.Property(proptype=float, default=1e-4)
    match_median = config.Property(proptype=bool, default=True)
    subtract = config.Property(proptype=bool, default=False)

    def setup(self, data_stack):
        """Set the stacked data.

        Parameters
        ----------
        data_stack : VisContainer
            Data stack to blend
        """
        self.data_stack = data_stack

    def process(self, data):
        """Blend a small amount of the stack into the incoming data.

        Parameters
        ----------
        data : VisContainer
            The data to be blended into. This is modified in place.

        Returns
        -------
        data_blend : VisContainer
            The modified data. This is the same object as the input, and it has been
            modified in place.
        """
        if type(self.data_stack) != type(data):
            raise TypeError(
                f"type(data) (={type(data)}) must match"
                f"type(data_stack) (={type(self.type)}"
            )

        # Try and get both the stack and the incoming data to have the same
        # distribution
        self.data_stack.redistribute(["freq", "time", "ra"])
        data.redistribute(["freq", "time", "ra"])

        if isinstance(data, containers.SiderealStream):
            dset_stack = self.data_stack.vis[:]
            dset = data.vis[:]
        else:
            raise TypeError(
                "Only SiderealStream's are currently supported. "
                f"Got type(data) = {type(data)}"
            )

        if dset_stack.shape != dset.shape:
            raise ValueError(
                f"Size of data ({dset.shape}) must match "
                f"data_stack ({dset_stack.shape})"
            )

        weight_stack = self.data_stack.weight[:]
        weight = data.weight[:]

        # Find the median offset between the stack and the daily data
        if self.match_median:
            # Find the parts of the both the stack and the daily data that are both
            # measured
            mask = (
                ((weight[:] > 0) & (weight_stack[:] > 0))
                .astype(np.float32)
                .view(np.ndarray)
            )

            # ... get the median of the stack in this common subset
            stack_med_real = weighted_median.weighted_median(
                dset_stack.real.view(np.ndarray).copy(), mask
            )
            stack_med_imag = weighted_median.weighted_median(
                dset_stack.imag.view(np.ndarray).copy(), mask
            )

            # ... get the median of the data in the common subset
            data_med_real = weighted_median.weighted_median(
                dset.real.view(np.ndarray).copy(), mask
            )
            data_med_imag = weighted_median.weighted_median(
                dset.imag.view(np.ndarray).copy(), mask
            )

            # ... construct an offset to match the medians in the time/RA direction
            stack_offset = (
                (data_med_real - stack_med_real)
                + 1.0j * (data_med_imag - stack_med_imag)
            )[..., np.newaxis]
            stack_offset = mpiarray.MPIArray.wrap(
                stack_offset,
                axis=dset_stack.axis,
                comm=dset_stack.comm,
            )

        else:
            stack_offset = 0

        if self.subtract:
            # Subtract the base stack where data is present, otherwise give zeros

            dset -= dset_stack + stack_offset
            dset *= (weight > 0).astype(np.float32)

            # This sets the weights where weight == 0 to frac * weight_stack,
            # otherwise weight is the sum of the variances. It's a bit obscure
            # because it attempts to do the operations in place rather than
            # building many temporaries
            weight *= tools.invert_no_zero(weight + weight_stack)
            weight += (weight == 0) * self.frac
            weight *= weight_stack

        else:
            # Perform a weighted average of the data to fill in missing samples
            dset *= weight
            dset += weight_stack * self.frac * (dset_stack + stack_offset)
            weight += weight_stack * self.frac

            dset *= tools.invert_no_zero(weight)

        return data


# This is here for compatibility
ApplyRFIMask = ApplyTimeFreqMask
MaskData = MaskMModeData


def medfilt(x, mask, size, *args):
    """Apply a moving median filter to masked data.

    The application is done by iterative filling to
    overcome the fact we don't have an actual implementation
    of a nanmedian.

    Parameters
    ----------
    x : np.ndarray
        Data to filter.
    mask : np.ndarray
        Mask of data to filter out.
    size : tuple
        Size of the window in each dimension.
    args : optional
        Additional arguments to pass to the moving weighted median

    Returns
    -------
    y : np.ndarray
        The masked data. Data within the mask is undefined.
    """
    if np.iscomplexobj(x):
        return medfilt(x.real, mask, size) + 1.0j * medfilt(x.imag, mask, size)

    # Copy and do initial masking
    x = np.ascontiguousarray(x.astype(np.float64))
    w = np.ascontiguousarray((~mask).astype(np.float64))

    return weighted_median.moving_weighted_median(x, w, size, *args)


def mad(x, mask, base_size=(11, 3), mad_size=(21, 21), debug=False, sigma=True):
    """Calculate the MAD of freq-time data.

    Parameters
    ----------
    x : np.ndarray
        Data to filter.
    mask : np.ndarray
        Initial mask.
    base_size : tuple
        Size of the window to use in (freq, time) when
        estimating the baseline.
    mad_size : tuple
        Size of the window to use in (freq, time) when
        estimating the MAD.
    debug : bool, optional
        If True, return deviation and mad arrays as well
    sigma : bool, optional
        Rescale the output into units of Gaussian sigmas.

    Returns
    -------
    mad : np.ndarray
        Size of deviation at each point in MAD units. This output may contain
        NaN's for regions of missing data.
    """
    xs = medfilt(x, mask, size=base_size)
    dev = np.abs(x - xs)

    mad = medfilt(dev, mask, size=mad_size)

    if sigma:
        mad *= 1.4826  # apply the conversion from MAD->sigma

    # Suppress warnings about NaNs produced during the division
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dev / mad

    if debug:
        return r, dev, mad
    return r


def inverse_binom_cdf_prob(k, N, F):
    """Calculate the trial probability that gives the CDF.

    This gets the trial probability that gives an overall cumulative
    probability for Pr(X <= k; N, p) = F

    Parameters
    ----------
    k : int
        Maximum number of successes.
    N : int
        Total number of trials.
    F : float
        The cumulative probability for (k, N).

    Returns
    -------
    p : float
        The trial probability.
    """
    # This uses the result that we can write the cumulative probability of a
    # binomial in terms of an incomplete beta function

    import scipy.special as sp

    return sp.betaincinv(k + 1, N - k, 1 - F)


def sigma_to_p(sigma):
    """Get the probability of an excursion larger than sigma for a Gaussian."""
    import scipy.stats as ss

    return 2 * ss.norm.sf(sigma)


def p_to_sigma(p):
    """Get the sigma exceeded by the tails of a Gaussian with probability p."""
    import scipy.stats as ss

    return ss.norm.isf(p / 2)


def tv_channels_flag(x, freq, sigma=5, f=0.5, debug=False):
    """Perform a higher sensitivity flagging for the TV stations.

    This flags a whole TV station band if more than fraction f of the samples
    within a station band exceed a given threshold. The threshold is calculated
    by wanting a fixed false positive rate (as described by sigma) for fraction
    f of samples exceeding the threshold

    Parameters
    ----------
    x : np.ndarray[freq, time]
        Deviations of data in sigma units.
    freq : np.ndarray[freq]
        Frequency of samples in MHz.
    sigma : float, optional
        The probability of a false positive given as a sigma of a Gaussian.
    f : float, optional
        Fraction of bad samples within each channel before flagging the whole
        thing.
    debug : bool, optional
        Returns (mask, fraction) instead to give extra debugging info.

    Returns
    -------
    mask : np.ndarray[bool]
        Mask of the input data.
    """
    p_false = sigma_to_p(sigma)
    frac = np.ones_like(x, dtype=np.float32)

    tvstart_freq = 398
    tvwidth_freq = 6

    # Calculate the boundaries of each frequency channel
    df = np.median(np.abs(np.diff(freq)))
    freq_start = freq - 0.5 * df
    freq_end = freq + 0.5 * df

    for i in range(67):
        # Find all frequencies that lie wholly or partially within the TV channel
        fs = tvstart_freq + i * tvwidth_freq
        fe = fs + tvwidth_freq
        sel = (freq_end >= fs) & (freq_start <= fe)

        # Don't continue processing channels for which we don't have
        # frequencies in the incoming data
        if not sel.any():
            continue

        # Calculate the threshold to apply
        N = sel.sum()
        k = int(f * N)

        # This is the Gaussian threshold required for there to be at most a p_false
        # chance of more than k trials exceeding the threshold. This is the correct
        # expression, and has been double checked by numerical trials.
        t = p_to_sigma(inverse_binom_cdf_prob(k, N, 1 - p_false))

        frac[sel] = (x[sel] > t).mean(axis=0)[np.newaxis, :]

    mask = frac > f

    if debug:
        return mask, frac

    return mask


def complex_med(x, *args, **kwargs):
    """Complex median, done by applying to the real/imag parts individually.

    Parameters
    ----------
    x : np.ndarray
        Array to apply to.
    *args, **kwargs : list, dict
        Passed straight through to `np.nanmedian`

    Returns
    -------
    m : np.ndarray
        Median.
    """
    return np.nanmedian(x.real, *args, **kwargs) + 1j * np.nanmedian(
        x.imag, *args, **kwargs
    )


def destripe(x, w, axis=1):
    """Subtract the median along a specified axis.

    Parameters
    ----------
    x : np.ndarray
        Array to destripe.
    w : np.ndarray
        Mask array for points to include (True) or ignore (False).
    axis : int, optional
        Axis to apply destriping along.

    Returns
    -------
    y : np.ndarray
        Destriped array.
    """
    # Calculate the average along the axis
    stripe = complex_med(np.where(w, x, np.nan), axis=axis)
    stripe = np.nan_to_num(stripe)

    # Construct a slice to broadcast back along the axis
    bsel = [slice(None)] * x.ndim
    bsel[axis] = None
    bsel = tuple(bsel)

    return x - stripe[bsel]

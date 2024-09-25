"""Tasks for flagging out bad or unwanted data.

This includes data quality flagging on timestream data; sun excision on sidereal
data; and pre-map making flagging on m-modes.

The convention for flagging/masking is `True` for contaminated samples that should
be excluded and `False` for clean samples.
"""

import warnings
from typing import Union, overload

import numpy as np
from caput import config, mpiarray, weighted_median
from cora.util import units
from scipy.signal import convolve, firwin, oaconvolve
from skimage.filters import apply_hysteresis_threshold

from ..analysis import delay, transform
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
            nanvis = (
                sstream.vis[:]
                * np.where(mask_bool, 1.0, np.nan)[np.newaxis, np.newaxis, :]
            )
            average = np.nanmedian(nanvis, axis=-1)[:, :, np.newaxis]
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
        mask = np.zeros_like(ss.weight[:].local_array, dtype=bool)

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
                    convolve(
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


class RFIStokesIMask(transform.ReduceVar):
    """Two-stage RFI filter based on Stokes I visibilities.

    Tries to independently target transient and persistant RFI.

    Stage 1 is applied to each frequency independently. A high-pass
    filter is applied in RA to isolate transient RFI. The high-pass
    filtered visibilities are beamformed, and a MAD filter is applied
    to the resulting map. A time/RA sample is then flagged if some
    fraction of beams exceed the MAD threshold for that sample.

    Stage 2 is applied across frequencies. A low-pass filter is applied
    in RA to reduce transient sky sources. The average visibility power
    is taken over 2+ cylinder separation baselines to obtain a single
    1D array per frequency. These powers are gathered across all
    frequencies and a basic background subtraction is applied. Sumthreshold
    algorithm is then used for flagging, with a variance estimate used to
    boost the expected noise during the daytime and bright point source
    transits.

    Attributes
    ----------
    mad_base_size : list of int, optional
        Median absolute deviations base window. Default is [1, 101].
    mad_dev_size : list of int, optional
        Median absolute deviation median deviation window.
        Default is [1, 51].
    sigma_high : float, optional
        Median absolute deviations sigma threshold. Default is 8.0.
    sigma_low : float, optional
        Median absolute deviations low sigma threshold. A value above
        this threshold is masked only if it is either larger than `sigma_high`
        or it is larger than `sigma_low` AND connected to a region larger
        than `sigma_high`. Default is 2.0.
    frac_samples : float, optional
        Fraction of flagged samples in map space above which the entire
        time sample will be flagged. Default is 0.01.
    max_m : int, optional
        Maximum size of the SumThreshold window. Default is 64.
    nsigma : float, optional
        Initial threshold for SumThreshold. Default is 5.0.
    solar_var_boost : float, optional
        Variance boost during solar transit. Default is 1e4.
    bg_win_size : list, optional
        The size of the window used to estimate the background sky, provided
        as (number of frequency channels, number of time samples).
        Default is [11, 3].
    var_win_size : list, optional
        The size of the window used when estimating the variance, provided
        as (number of frequency channels, number of time samples).
        Default is [3, 31].
    lowpass_cutoff : float, optional
        Angular cutoff of the ra lowpass filter. Default is 7.5, which
        corresponds to about 30 minutes of observation time.
    """

    mad_base_size = config.list_type(int, length=2, default=[1, 101])
    mad_dev_size = config.list_type(int, length=2, default=[1, 51])
    sigma_high = config.Property(proptype=float, default=8.0)
    sigma_low = config.Property(proptype=float, default=2.0)
    frac_samples = config.Property(proptype=float, default=0.01)

    max_m = config.Property(proptype=int, default=64)
    nsigma = config.Property(proptype=float, default=5.0)
    solar_var_boost = config.Property(proptype=float, default=1e4)
    bg_win_size = config.list_type(int, length=2, default=[11, 3])
    var_win_size = config.list_type(int, length=2, default=[3, 31])
    lowpass_cutoff = config.Property(proptype=float, default=7.5)

    def setup(self, telescope):
        """Set up the baseline selections and ordering.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)
        # Set the parent class attribute to use the correct weighting
        self.weighting = "weighted"

    def process(self, stream):
        """Make a mask from the data.

        Parameters
        ----------
        stream : dcontainers.TimeStream | dcontainers.SiderealStream
            Data to use when masking. Axes should be frequency, stack,
            and time-like.

        Returns
        -------
        mask : dcontainers.RFIMask | dcontainers.SiderealRFIMask
            Time-frequency mask, where values marked `True` are flagged.
        power : dcontainers.TimeStream | dcontainers.SiderealStream
            Time-frequency power metric used in second-stage flagging.
        """
        stream.redistribute("freq")

        csd = stream.attrs.get("lsd", stream.attrs.get("csd"))

        if csd is None:
            raise ValueError("Dataset does not have a `csd` or `lsd` attribute.")

        if "time" in stream.index_map:
            times = stream.time
        elif "ra" in stream.index_map:
            times = self.telescope.lsd_to_unix(csd + stream.ra / 360.0)
        else:
            raise TypeError(
                f"Expected data with `time` or `ra` axis. Got {type(stream)}."
            )

        ra = 2 * np.pi * (self.telescope.unix_to_lsd(times) - csd)
        fsel = stream.vis[:].local_bounds
        freq = stream.freq[fsel]

        # Get stokes I and redistribute over frequency. Axes are rearranged
        # in order (baseline, freq, time)
        vis, weight, baselines = delay.stokes_I(stream, self.telescope)
        vis = vis.redistribute(1).local_array
        weight = weight.redistribute(1).local_array

        # Set up the initial mask
        mask = np.all(weight == 0, axis=0)
        mask |= self._static_rfi_mask_hook(freq, times[0])[:, np.newaxis]
        self.log.debug(f"{100.0 * mask.mean():.2f}% of data initially flagged.")

        # Mask scattered transient rfi for each frequency independently
        # Also get the average power per frequency after applying a low-pass filter
        mask, power = self.mask_single_channel(vis, weight, mask, freq, baselines, ra)

        # Gather the entire mask and power arrays
        mask = mpiarray.MPIArray.wrap(mask, axis=0).allgather()
        power = mpiarray.MPIArray.wrap(power, axis=0).allgather()

        # Mask high power across frequencies
        summask = self.mask_multi_channel(power, mask, times)

        # Create the output containers
        if "ra" in stream.index_map:
            mask_out = containers.SiderealRFIMask(axes_from=stream, attrs_from=stream)
            power_out = containers.SiderealStream(
                axes_from=stream, attrs_from=stream, stack=1
            )
        else:
            mask_out = containers.RFIMask(axes_from=stream, attrs_from=stream)
            power_out = containers.TimeStream(
                axes_from=stream, attrs_from=stream, stack=1
            )

        # Save out the power metric to its own container. Save
        # only the first-stage mask to the weights in order to
        # visualize the differences
        power_out.vis[:].local_array[:] = power[fsel, np.newaxis]
        power_out.weight[:].local_array[:] = mask[fsel, np.newaxis]

        # Combine and save out the full mask
        mask_out.mask[:] = mask | summask

        self.log.debug(f"{100.0 * mask_out.mask[:].mean():.2f}% of data flagged.")

        return mask_out, power_out

    def mask_single_channel(self, vis, weight, mask, freq, baselines, ra):
        """Mask scattered rfi."""
        # Get the per-frequency high-pass and low-pass cuts
        hpf_cut = self._hpf_cut_hook(freq, baselines)
        lpf_cut = self._lpf_cut_hook(freq, baselines)
        # Select cylinders to include in static power estimation.
        # Choose baselines which should not contain much sky structure
        bl_sel = baselines[:, 0] > 2.0 * self.telescope.u_width
        # Set up an array to store mean power from non-sky sources
        power = np.zeros_like(weight[0], dtype=np.float64)

        # Iterate over frequencies
        for fsel in range(vis.shape[1]):
            if np.all(mask[fsel]):
                # Frequency is already masked
                continue

            # Apply a high-pass mmode filter. Scattered emission appears
            # similar to an impulse function in time, so its fourier transform
            # should extend to high m
            v_hpf = self.apply_filter(
                vis[:, fsel], weight[:, fsel], ra, hpf_cut[fsel], type_="high"
            )

            # MAD filter flags scattered emission after beamforming
            map_hpf = abs(np.fft.fft(v_hpf, axis=0))
            mad_mask = np.zeros_like(v_hpf, dtype=bool) | mask[fsel][np.newaxis]
            mad_ = mad(map_hpf, mad_mask, self.mad_base_size, self.mad_dev_size)
            # Hysteresis threshold mask flags anything above `sigma_high` or
            # anything above `sigma_low` ONLY if it is connected to a region
            # above `sigma_high`
            mad_mask |= apply_hysteresis_threshold(
                mad_, self.sigma_low, self.sigma_high
            )
            # Collapse over baselines and flag
            mean_flagged = np.mean(mad_mask, axis=0)

            # Apply a low pass filter
            lp_win = (mean_flagged < 0.5)[np.newaxis]
            v_lpf = self.apply_filter(
                vis[:, fsel], weight[:, fsel] * lp_win, ra, lpf_cut[fsel], type_="low"
            )

            # Take the average over selected baselines
            power[fsel] = np.mean(abs(v_lpf)[bl_sel], axis=0)
            # Apply the hp mask
            mask[fsel] |= mean_flagged > self.frac_samples

        return mask, power

    def mask_multi_channel(self, power, mask, times):
        """Mask slow-moving narrow-band RFI."""
        # Find times where there are bright sources in the sky
        # which should be treated differently
        source_flag = self._source_flag_hook(times)
        sun_flag = self._solar_transit_hook(times)

        # Calculate the weighted variance over time, excluding times
        # flagged to have higher than normal variance
        wvar, ws = self.reduction(power, ~mask & ~source_flag[np.newaxis], axis=1)
        # Get a smoothed estimate of the per-frequency variance
        wvar = tools.arPLS_1d(wvar, ws == 0, lam=1e1)[:, np.newaxis]
        # Ensure this estimate is strictly non-negative. The baseline
        # fit can produce negative values near edges if there is a
        # strong rolloff towards 0 (in which case the variance shoud
        # be zero anyway)
        wvar[wvar < 0] = 0.0

        # Get a background estimate of the sky, assuming that the
        # type of rfi we're looking for is very localised in frequency
        p_med = medfilt(power, mask, size=self.bg_win_size)

        # Create an estimate of the variance for each sample. Find the
        # ratio of a rolling median of the background sky to the overall
        # median in time and multiply this ratio by the per-frequency
        # variance estimate
        med = weighted_median.weighted_median(p_med, (~mask).astype(p_med.dtype))
        rmed = medfilt(p_med, mask, size=self.var_win_size)
        # Get the initial full variance using the lower variance estimate.
        # Increase the variance estimate during solar transit
        var = wvar * rmed * tools.invert_no_zero(med)[:, np.newaxis]
        var[:, sun_flag] *= self.solar_var_boost

        # Generate an RFI mask from the background-subtracted data
        summask = rfi.sumthreshold(
            power - p_med,
            start_flag=mask,
            max_m=self.max_m,
            threshold1=self.nsigma,
            variance=var,
        )

        # Expand the mask in time only. Expanding in frequency generally ends
        # up being too aggressive
        summask |= rfi.sir((summask & ~mask)[:, np.newaxis], only_time=True)[:, 0]

        return summask

    @staticmethod
    def apply_filter(vis, weight, samples, fcut, type_="high"):
        """Apply a high-pass or low-pass mmode filter."""
        # Median sampling rate
        fs = 1 / np.median(abs(np.diff(samples)))
        # Order is sample frequency over cutoff frequency. Ensure order is odd
        order = int(np.ceil(fs / fcut) // 2 * 2 + 1)
        # Make the window. Flattop seems to work well here
        kernel = firwin(order, fcut, window="flattop", fs=fs)[np.newaxis]

        # Low-pass filter the visibilities. `oaconvolve` is significantly
        # faster than the standard convolve method
        vw_lp = oaconvolve(vis * weight, kernel, mode="same")
        ww_lp = oaconvolve(weight, kernel, mode="same")
        vis_lp = vw_lp * tools.invert_no_zero(ww_lp)

        if type_ == "high":
            return vis - vis_lp

        return vis_lp

    def _hpf_cut_hook(self, freq, baselines):
        """Get a high-pass fringe rate cut for each frequency."""
        dec = np.deg2rad(self.telescope.latitude)
        lambda_inv = freq[:, np.newaxis] * 1e6 / units.c

        # Maximum cut per frequency
        return lambda_inv * baselines[:, 0].max() / np.cos(dec)

    def _lpf_cut_hook(self, freq, baselines):
        """Get a low-pass fringe rate cut for each frequency."""
        cut = 1 / np.deg2rad(self.lowpass_cutoff)

        return np.ones(len(freq), dtype=np.float64) * cut

    def _static_rfi_mask_hook(self, freq, timestamp=None):
        """Override to mask entire frequency channels.

        Parameters
        ----------
        freq : np.ndarray[nfreq]
            1D array of frequencies in the data (in MHz).

        timestamp : np.array[float]
            Start observing time (in unix time)

        Returns
        -------
        mask : np.ndarray[nfreq]
            Mask array. True will mask a frequency channel.
        """
        return np.zeros_like(freq, dtype=bool)

    def _source_flag_hook(self, times):
        """Override to mask out bright point sources.

        Parameters
        ----------
        times : np.ndarray[float]
            Array of timestamps.

        Returns
        -------
        mask : np.ndarray[float]
            Mask array. True will mask out a time sample.
        """
        return np.zeros_like(times, dtype=bool)

    def _solar_transit_hook(self, times):
        """Override to flag solar transit times.

        Parameters
        ----------
        times : np.ndarray[float]
            Array of timestamps.

        Returns
        -------
        mask : np.ndarray[float]
            Mask array. True will mask out a time sample.
        """
        return np.zeros_like(times, dtype=bool)


class RFIMaskChisqHighDelay(task.SingleTask):
    """Mask frequencies and times with anomalous chi-squared test statistic.

    Attributes
    ----------
    reg_arpls : float
        Smoothness regularisation used when estimating the baseline
        for flagging bad frequencies. Default is 1e5.
    nsigma_1d : float
        Mask any frequency where the median over unmasked time samples
        deviates from the baseline by more than this number of
        median absolute deviations.  Default is 5.0.
    win_t : float
        Size of the window (in number of time samples)
        used to compute a median filtered version of the test statistic.
    win_f : float
        Size of the window (in number of frequency channels)
        used to compute a median filtered version of the test statistic.
    nsigma_2d: float
        Mask any frequency and time where the absolute deviation
        from the median filtered version is greater than this number
        of expected standard deviations given the number of degrees
        of freedom (i.e., number of baselines).
    estimate_var : bool
        Estimate the variance in the test statistic using the median
        absolute deviation over a region defined by the win_t and
        win_f parameters.
    only_positive : bool
        Only mask large postive excursions in the test statistic,
        leaving large negative excursions unmasked.
    """

    reg_arpls = config.Property(proptype=float, default=1e5)
    nsigma_1d = config.Property(proptype=float, default=5.0)

    win_t = config.Property(proptype=int, default=601)
    win_f = config.Property(proptype=int, default=1)
    nsigma_2d = config.Property(proptype=float, default=5.0)
    estimate_var = config.Property(proptype=bool, default=False)
    only_positive = config.Property(proptype=bool, default=False)

    def setup(self, telescope=None):
        """Save telescope object for time calculations.

        Only used to convert (LSD, RA) to unix time when masking
        sidereal streams.  Not required when masking time streams.

        Parameters
        ----------
        telescope : TransitTelescope
            Telescope object used for time calculations.
        """
        self.telescope = None if telescope is None else io.get_telescope(telescope)

    def process(self, stream):
        """Generate a mask from the data.

        Parameters
        ----------
        stream : dcontainers.TimeStream | dcontainers.SiderealStream |
                 dcontainers.HybridVisStream | dcontainers.RingMap
            Container holding a chi-squared test statistic in the visibility dataset.
            A weighted average will be taken over any axis that is not time/ra or frequency.

        Returns
        -------
        mask : dcontainers.RFIMask | dcontainers.SiderealRFIMask
            Time-frequency mask, where values marked `True` are flagged.
        """
        # Distribute over frequency
        stream.redistribute("freq")
        freq = stream.freq

        # Determine time axis
        multiple_days = False
        if "ra" in stream.index_map:

            if self.telescope is None:
                raise RuntimeError(
                    "For sidereal streams, must provide "
                    "telescope object during setup."
                )

            csd = stream.attrs.get("lsd", stream.attrs.get("csd"))
            if csd is None:
                raise ValueError("Data does not have a `csd` or `lsd` attribute.")

            if not np.isscalar(csd):
                csd = np.floor(np.mean(csd))
                multiple_days = True

            timestamp = self.telescope.lsd_to_unix(csd + stream.ra / 360.0)

        else:
            timestamp = stream.time

        # Expand the weight dataset so that it can broadcast against the data dataset.
        # Assumes that weight contains a subset of the axes in data, with the shared
        # axes in the same order.  This is true for all of the supported input
        # containers listed in the docstring.
        dax = list(stream.data.attrs["axis"])
        wax = list(stream.weight.attrs["axis"])
        wshp = [stream.weight.shape[wax.index(ax)] if ax in wax else 1 for ax in dax]
        wshp[dax.index("freq")] = None

        # Extract the shape of the axes that are missing from the weights dataset,
        # so that we can scale the denominator by this factor.
        wshp_missing = [sz for sz, ax in zip(stream.data.shape, dax) if ax not in wax]
        wfactor = np.prod(wshp_missing) if len(wshp_missing) > 0 else 1.0

        # Sum over any axis that is neither time nor frequency
        axsum = tuple(
            [ii for ii, ax in enumerate(dax) if ax not in ["freq", "time", "ra"]]
        )

        chisq = stream.data[:].real
        weight = stream.weight[:].reshape(*wshp)

        wsum = wfactor * np.sum(weight, axis=axsum)
        chisq = np.sum(weight * chisq, axis=axsum) * tools.invert_no_zero(wsum)

        # Gather all frequencies on all nodes
        chisq = chisq.allgather()
        wsum = wsum.allgather()

        mask_input = wsum == 0.0

        # Determine what time samples should be ignored when constructing the mask
        if multiple_days:
            mask_daytime = np.zeros(timestamp.size, dtype=bool)
        else:
            mask_daytime = self._day_flag_hook(timestamp)

        mask_time = self._source_flag_hook(timestamp, freq)

        mask = mask_input | mask_time

        # Mask using median over time
        mask_output = np.zeros((freq.size, timestamp.size), dtype=bool)

        if self.nsigma_1d > 0.0:
            mask_1d = self.mask_1d(chisq, mask | mask_daytime)[:, np.newaxis]
            mask |= mask_1d
            mask_output |= mask_1d

        # Mask using dynamic spectrum
        if self.nsigma_2d > 0.0:
            # The inverse variance of the chisq per dof test statistic is ndof / 2.
            # This expression assumes ndof is stored in the weight dataset.
            w = ~mask * wsum / 2.0
            mask_2d = self.mask_2d(chisq, w)

            mask_output |= mask_2d & ~mask_daytime

        # Save to output container
        OutputContainer = (
            containers.SiderealRFIMask
            if "ra" in stream.index_map
            else containers.RFIMask
        )
        output = OutputContainer(axes_from=stream, attrs_from=stream)

        output.mask[:] = mask_output

        return output

    def mask_1d(self, y, m):
        """Mask frequency channels where median chi-squared deviates from neighbors.

        Parameters
        ----------
        y : np.ndarray[nfreq, ntime]
            Chi-squared per degree of freedom.
        m : np.ndarray[nfreq, ntime]
            Boolean mask that indicates which samples to ignore
            when calculating the median over time.

        Returns
        -------
        mask : np.ndarray[nfreq]
            Boolean mask that indicates frequency channels where
            the median chi-squared over time deviates significantly
            from that of the neighboring channels.
        """
        # For each frequency, caculate a weighted median
        y = np.ascontiguousarray(y.astype(np.float64))
        w = np.ascontiguousarray((~m).astype(np.float64))

        med_y = weighted_median.weighted_median(y, w)
        med_m = np.all(m, axis=-1)
        med_w = np.ascontiguousarray((~med_m).astype(np.float64))

        # Estimate a baseline
        baseline = tools.arPLS_1d(med_y, mask=med_m, lam=self.reg_arpls)

        # Subtract the baseline and estimate the noise
        abs_dev = np.where(med_m, 0.0, np.abs(med_y - baseline))

        mad = 1.48625 * weighted_median.weighted_median(abs_dev, med_w)

        # Flag outliers
        return abs_dev > (self.nsigma_1d * mad)

    def mask_2d(self, y, w):
        """Mask frequencies and times where the chi-squared deviates from local median.

        Parameters
        ----------
        y : np.ndarray[nfreq, ntime]
            Chi-squared per degree of freedom.
        w : np.ndarray[nfreq, ntime]
            Inverse variance of the chi-squared per degree of freedom,
            with zero indicating previously masked samples.

        Returns
        -------
        mask : np.ndarray[nfreq]
            Boolean mask that indicates frequencies and times where
            chi-squared deviates significantly from the local median.
        """
        # For each frequency, caculate a moving weighted median
        y = np.ascontiguousarray(y.astype(np.float64))
        w = np.ascontiguousarray(w.astype(np.float64))
        win_size = (self.win_f, self.win_t)

        med_y = weighted_median.moving_weighted_median(y, w, win_size)

        # Calculate the deviation from the median, normalized by the
        # expected standard deviation
        dy = (y - med_y) * np.sqrt(w)

        # If requested, estimate the variance in the test statistic
        # using the median absolute deviation.
        if self.estimate_var:
            f = np.ascontiguousarray((w > 0.0).astype(np.float64))
            mad_y = 1.48625 * weighted_median.moving_weighted_median(
                np.abs(dy), f, win_size
            )
            dy *= tools.invert_no_zero(mad_y)

        # Take the absolute value of the relative excursion unless
        # explicitely requested to only flag positive excursions.
        if not self.only_positive:
            dy = np.abs(dy)

        # Flag times and frequencies that deviate by more than some threshold
        return dy > self.nsigma_2d

    def _source_flag_hook(self, times, freq):
        """Override to mask bright point sources.

        Parameters
        ----------
        times : np.ndarray[ntime]
            Array of timestamps.
        freq : np.ndarray[nfreq]
            Array of frequencies.

        Returns
        -------
        mask : np.ndarray[nfreq, ntime]
            Mask array. True will mask out a time sample.
        """
        return np.zeros((freq.size, times.size), dtype=bool)

    def _day_flag_hook(self, times):
        """Override to mask daytime.

        Parameters
        ----------
        times : np.ndarray[ntime]
            Array of timestamps.

        Returns
        -------
        mask : np.ndarray[nfreq, ntime]
            Mask array. True will mask out a time sample.
        """
        return np.zeros(times.size, dtype=bool)


class RFISensitivityMask(task.SingleTask):
    """Identify RFI as deviations in system sensitivity from expected radiometer noise.

    Attributes
    ----------
    mask_type : string, optional
        One of 'mad', 'sumthreshold' or 'combine'.
        Default is combine, which uses the sumthreshold everywhere
        except around the transits of the sun and bright point sources,
        where it applies the MAD mask to avoid masking out the transits.
    include_pol : list of strings, optional
        The list of polarisations to include. Default is to use all
        polarisations.
    nsigma_1d : float, optional
        Construct a static mask by identifying any frequency channel
        whose quantile over time deviates from the median over frequency
        by more than this number of median absolute deviations.
        Default: 5.0
    quantile_1d: float, optional
        The quantile to use along time to construct the static mask.
        Default: 0.15
    win_f_1d : int, optional
        Number of frequency channels used to calculate a rolling median
        and median absolute deviation for the staic mask.  Default: 191
    nsigma : float, optional
        The final threshold for the MAD, TV, and SumThreshold algorithms
        given as number of standard deviations.  Default: 5.0
    niter : int, optional
        Number of iterations.  At each iterations the baseline and standard
        deviation are re-estimated using the mask from the previous iteration.
        Default: 5
    rho : float, optional
        Reduce the threshold by this factor at each iteration.  A value of 1
        will keep the threshold constant for all iterations.  Default: 1.5
    base_size : [int, int]
        The size of the region used to estimate the baseline, provided as
        (number of frequency channels, number of time samples).  Default: (37, 181)
    mad_size : [int, int]
        The size of the region used to estimate the standard deviation, provided
        (number of frequency channels, number of time samples).  Default: (101, 31)
    tv_fraction : float, optional
        Fraction of bad samples in a digital TV channel that cause the whole
        channel to be flagged.  Default: 0.5
    max_m : int, optional
        Maximum size of the SumThreshold window to use.  Default: 64
    sir : bool, optional
        Apply scale invariant rank (SIR) operator on top of final mask.
        Default: False
    eta : float optional
        Aggressiveness of the SIR operator.  With eta=0, no additional samples
        are flagged and with eta=1, all samples will be flagged.  Default: 0.2
    only_time : bool, optinal
        Only apply the SIR operator along the time axis.  Default: False
    """

    mask_type = config.enum(["mad", "sumthreshold", "combine"], default="combine")
    include_pol = config.list_type(str, default=None)

    nsigma_1d = config.Property(proptype=float, default=5.0)
    quantile_1d = config.Property(proptype=float, default=0.15)
    win_f_1d = config.Property(proptype=int, default=191)

    nsigma = config.Property(proptype=float, default=5.0)
    niter = config.Property(proptype=int, default=5)
    rho = config.Property(proptype=float, default=1.5)

    base_size = config.list_type(int, length=2, default=(37, 181))
    mad_size = config.list_type(int, length=2, default=(101, 31))
    tv_fraction = config.Property(proptype=float, default=0.5)
    max_m = config.Property(proptype=int, default=64)

    sir = config.Property(proptype=bool, default=False)
    eta = config.Property(proptype=float, default=0.2)
    only_time = config.Property(proptype=bool, default=False)

    # Convert MAD to RMS
    MAD_TO_RMS = 1.4826

    def setup(self):
        """Define the threshold as a function of iteration."""
        self.threshold = self.nsigma * self.rho ** np.arange(self.niter)[::-1]

    def process(self, sensitivity):
        """Derive an RFI mask from sensitivity data.

        Parameters
        ----------
        sensitivity : containers.SystemSensitivity
            Sensitivity data.

        Returns
        -------
        rfimask : containers.RFIMask
            RFI mask derived from sensitivity.
        """
        # Distribute over polarisation as we need all times and frequencies
        # available simultaneously
        sensitivity.redistribute("pol")

        pol = sensitivity.pol
        npol = len(pol)

        # Divide sensitivity to get a radiometer test
        radiometer = sensitivity.measured[:].local_array * tools.invert_no_zero(
            sensitivity.radiometer[:].local_array
        )
        flag = sensitivity.weight[:].local_array == 0.0

        # Look up static mask if it exists
        freq = sensitivity.freq
        static_flag = ~self._static_rfi_mask_hook(freq, sensitivity.time[0])

        # Look up times to use the mad mask
        if self.mask_type == "combine":
            madtimes = self._combine_st_mad_hook(sensitivity.time, freq)

        # Create arrays to hold final masks
        nfreq, nlocal, ntime = radiometer.shape

        finalmask = mpiarray.MPIArray((npol, nfreq, ntime), axis=0, dtype=bool)
        finalmask[:] = False

        # Loop over polarisations
        for li, ii in finalmask.enumerate(0):
            # Only process this polarisation if we should be including it,
            # otherwise skip and let it be implicitly set to False (i.e. not
            # masked)
            if self.include_pol and pol[ii] not in self.include_pol:
                continue

            # Initial flag on weights equal to zero.
            y = radiometer[:, li, :]
            origflag = flag[:, li, :]

            # Combine weights with static flag
            current_flag = origflag | static_flag[:, None]

            # Mask frequency channels based on their quantile over time
            if self.nsigma_1d is not None:
                flag_1d, y_static = self._mask_1d(y, current_flag)
                current_flag |= flag_1d[:, None]
                y = y - y_static[:, None]

            # Slowly reduce the threshold.  At each iteration generate a new estimate
            # of the background sky and the variance using the current mask.
            for nsigma in self.threshold:

                # Estimate the background by taking a 2D rolling median
                med_y = medfilt(y, current_flag, self.base_size)
                dy = y - med_y

                # Estimate the median absolute deviation
                ady = np.abs(dy)

                med_ady = self.MAD_TO_RMS * medfilt(ady, current_flag, self.mad_size)

                ady_nsigma = ady * tools.invert_no_zero(med_ady)

                # Flag based on median absolute deviation
                madmask = ady_nsigma > nsigma

                # Flag for scattered TV emission
                tvmask = tv_channels_flag(
                    ady_nsigma, freq, sigma=nsigma, f=self.tv_fraction
                )

                madmask |= tvmask

                # Pick which of the MAD or SumThreshold mask to use (or blend them)
                if self.mask_type == "mad":
                    current_flag |= madmask

                else:
                    # Generate sumthreshold mask
                    stmask = rfi.sumthreshold(
                        dy,
                        self.max_m,
                        start_flag=current_flag | tvmask,
                        threshold1=nsigma,
                        remove_median=False,
                        correct_for_missing=True,
                        rho=1.0,
                        variance=med_ady**2,
                    )

                    if self.mask_type == "sumthreshold":
                        current_flag |= stmask

                    else:  # combine
                        tempmask = np.where(madtimes, madmask, stmask)
                        # If SIR is not going to be applied to the final mask,
                        # then apply it here to extend the sumthreshold mask
                        # in time across the transits.
                        if not self.sir:
                            expanded = rfi.sir(
                                tempmask[:, None], eta=0.2, only_time=True
                            )[:, 0]
                            tempmask = np.where(madtimes, expanded, tempmask)

                        current_flag |= tempmask

            finalmask[li] = current_flag

        # Perform an OR (.any) along the pol axis and reform into an MPIArray
        # along the freq axis
        finalmask = mpiarray.MPIArray.wrap(finalmask.redistribute(1).any(0), 0)

        # Collect all parts of the mask onto rank 1 and then broadcast to all ranks
        finalmask = mpiarray.MPIArray.wrap(finalmask, 0).allgather()

        # Log the fraction of data masked
        percent_masked = 100 * np.sum(finalmask) / float(finalmask.size)
        self.log.info(
            f"After RFISensitivityMask, {percent_masked:0.2f} percent "
            "of data will be masked."
        )

        # Apply scale invariant rank (SIR) operator, if asked for.
        if self.sir:
            finalmask = self._apply_sir(finalmask, static_flag)

            # Again log the fraction of data masked, so we can
            # tell how much data is being excised by SIR
            percent_masked = 100 * np.sum(finalmask) / float(finalmask.size)
            self.log.info(
                f"After SIR operator, {percent_masked:0.2f} percent "
                "of data will be masked."
            )

        # Create container to hold mask
        rfimask = containers.RFIMask(axes_from=sensitivity, attrs_from=sensitivity)
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

    def _mask_1d(self, rad, mask):
        """Mask based on the median over time."""
        y = np.ascontiguousarray(rad.astype(np.float64))
        w = np.ascontiguousarray((~mask).astype(np.float64))

        medt_y = weighted_median.quantile(y, w, self.quantile_1d)
        medt_w = np.any(w, axis=-1).astype(np.float64)

        if self.win_f_1d is None:
            medf_medt_y = weighted_median.weighted_median(medt_y, medt_w)
        else:
            medf_medt_y = weighted_median.moving_weighted_median(
                medt_y, medt_w, self.win_f_1d
            )

        absd_medt_y = np.abs(medt_y - medf_medt_y)

        if self.win_f_1d is None:
            mad_1d = self.MAD_TO_RMS * weighted_median.weighted_median(
                absd_medt_y, medt_w
            )
        else:
            mad_1d = self.MAD_TO_RMS * weighted_median.moving_weighted_median(
                absd_medt_y, medt_w, self.win_f_1d
            )

        return absd_medt_y > (self.nsigma_1d * mad_1d), medt_y

    def _apply_sir(self, mask, baseflag, eta=0.2):
        """Expand the mask with SIR."""
        # Remove baseflag from mask and run SIR
        nobaseflag = np.copy(mask)
        nobaseflag[baseflag] = False
        nobaseflagsir = rfi.sir(
            nobaseflag[:, np.newaxis, :], eta=self.eta, only_time=self.only_time
        )[:, 0, :]

        # Make sure the original mask (including baseflag) is still masked
        return nobaseflagsir | mask


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

    Typically this is used to mask out all inputs at times and
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
    freq_frac : float, optional
        Fully mask any frequency where the fraction of unflagged samples
        is less than this value. Default is None.
    """

    bad_freq_ind = config.Property(proptype=list, default=None)
    factorize = config.Property(proptype=bool, default=False)
    all_time = config.Property(proptype=bool, default=False)
    mask_missing_data = config.Property(proptype=bool, default=False)
    freq_frac = config.Property(proptype=float, default=None)

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

        if self.freq_frac is not None:
            mask |= mask.mean(axis=1)[:, np.newaxis] > (1.0 - self.freq_frac)
            self.log.info(f"Fractional mask: {100.0 * mask.mean():.2f}% flagged.")

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
            fun=fmask,
            bounds=(0, 1),
            method="bounded",
            options={"maxiter": 20, "xatol": 1e-4},
        )

        if not res.success:
            self.log.debug("Optimisation did not converge, but this isn't unexpected.")

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
    mask_freq : bool, optional
        Maintain masking if a frequency is entirely flagged - i.e., even if
        blending data exists in those bands, do not blend.
    """

    frac = config.Property(proptype=float, default=1e-4)
    match_median = config.Property(proptype=bool, default=True)
    subtract = config.Property(proptype=bool, default=False)
    mask_freq = config.Property(proptype=bool, default=False)

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
        if "effective_ra" in data.datasets:
            raise TypeError(
                "Blending uncorrected rebinned data not supported. "
                "Please apply a correction such as `sidereal.RebinGradientCorrection."
            )

        if not isinstance(data, type(self.data_stack)):
            raise TypeError(
                f"type(data) (={type(data)}) must match"
                f"type(data_stack) (={type(self.data_stack)}"
            )

        # Try and get both the stack and the incoming data to have the same
        # distribution
        self.data_stack.redistribute(["freq", "time", "ra"])
        data.redistribute(["freq", "time", "ra"])

        if isinstance(data, containers.SiderealStream):
            dset_stack = self.data_stack.vis[:].local_array
            dset = data.vis[:].local_array
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

        weight_stack = self.data_stack.weight[:].local_array
        weight = data.weight[:].local_array

        # Find the median offset between the stack and the daily data
        if self.match_median:
            # Find the parts of the both the stack and the daily data that are both
            # measured
            mask = ((weight[:] > 0) & (weight_stack[:] > 0)).astype(np.float32)

            # Get the median of the stack in this common subset
            stack_med_real = weighted_median.weighted_median(
                np.ascontiguousarray(dset_stack.real), mask
            )
            stack_med_imag = weighted_median.weighted_median(
                np.ascontiguousarray(dset_stack.imag), mask
            )

            # Get the median of the data in the common subset
            data_med_real = weighted_median.weighted_median(
                np.ascontiguousarray(dset.real), mask
            )
            data_med_imag = weighted_median.weighted_median(
                np.ascontiguousarray(dset.imag), mask
            )

            # Construct an offset to match the medians in the time/RA direction
            stack_offset = (
                (data_med_real - stack_med_real)
                + 1.0j * (data_med_imag - stack_med_imag)
            )[..., np.newaxis]

        else:
            stack_offset = 0

        if self.mask_freq:
            # Collapse the frequency selection over baselines and RA
            fsel = np.any(weight, axis=(1, 2), keepdims=True)

            # Apply the frequency selection to the stacked weights
            weight_stack *= fsel.astype(np.float64)

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

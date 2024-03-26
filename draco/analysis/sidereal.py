"""Take timestream data and regridding it into sidereal days which can be stacked.

Usage
=====

Generally you would want to use these tasks together. Sending time stream data
into  :class:`SiderealGrouper`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`SiderealStacker` if you want to combine the different days.
"""

import numpy as np
import scipy.linalg as la
from caput import config, mpiarray, tod, weighted_median
from cora.util import units

from ..core import containers, io, task
from ..util import regrid, tools
from .transform import Regridder


class SiderealGrouper(task.SingleTask):
    """Group individual timestreams together into whole Sidereal days.

    Attributes
    ----------
    padding : float
        Extra amount of a sidereal day to pad each timestream by. Useful for
        getting rid of interpolation artifacts.
    offset : float
        Time in seconds to subtract before determining the LSD.  Useful if the
        desired output is not a full sideral stream, but rather a narrow window
        around source transits on different sideral days.  In that case, one
        should set this quantity to `240 * (source_ra - 180)`.
    min_day_length : float
        Require at least this fraction of a full sidereal day to process.
    """

    padding = config.Property(proptype=float, default=0.0)
    offset = config.Property(proptype=float, default=0.0)
    min_day_length = config.Property(proptype=float, default=0.10)

    def __init__(self):
        super().__init__()

        self._timestream_list = []
        self._current_lsd = None

    def setup(self, manager):
        """Set the local observers position.

        Parameters
        ----------
        manager : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        # Need an observer object holding the geographic location of the telescope.
        self.observer = io.get_telescope(manager)

    def process(self, tstream):
        """Load in each sidereal day.

        Parameters
        ----------
        tstream : containers.TimeStream
            Timestream to group together.

        Returns
        -------
        ts : containers.TimeStream or None
            Returns the timestream of each sidereal day when we have received
            the last file, otherwise returns :obj:`None`.
        """
        # This is the start and the end of the LSDs of the file only if padding
        # is chosen to be 0 (default). If padding is set to some value then 'lsd_start'
        # will actually correspond to the start of of the requested time frame (incl
        # padding)
        lsd_start = int(
            self.observer.unix_to_lsd(tstream.time[0] - self.padding - self.offset)
        )
        lsd_end = int(
            self.observer.unix_to_lsd(tstream.time[-1] + self.padding - self.offset)
        )

        # If current_lsd is None then this is the first time we've run
        if self._current_lsd is None:
            self._current_lsd = lsd_start

        # If this file started during the current lsd add it onto the list
        if self._current_lsd == lsd_start:
            self._timestream_list.append(tstream)

        self.log.info("Adding file into group for LSD:%i", lsd_start)

        # If this file ends during a later LSD then we need to process the
        # current list and restart the system
        if self._current_lsd < lsd_end:
            self.log.info("Concatenating files for LSD:%i", self._current_lsd)

            # Combine timestreams into a single container for the whole day this
            # could get returned as None if there wasn't enough data
            tstream_all = self._process_current_lsd()

            # Reset list and current LSD for the new file
            self._timestream_list = [tstream]
            self._current_lsd = lsd_end

            return tstream_all

        return None

    def process_finish(self):
        """Return the final sidereal day.

        Returns
        -------
        ts : containers.TimeStream or None
            Returns the timestream of the final sidereal day if it's long
            enough, otherwise returns :obj:`None`.
        """
        # If we are here there is no more data coming, we just need to process any remaining data
        return self._process_current_lsd() if self._timestream_list else None

    def _process_current_lsd(self):
        # Combine the current set of files into a timestream

        lsd = self._current_lsd

        # Calculate the length of data in this current LSD
        start = self.observer.unix_to_lsd(self._timestream_list[0].time[0])
        end = self.observer.unix_to_lsd(self._timestream_list[-1].time[-1])
        day_length = min(end, lsd + 1) - max(start, lsd)

        # If the amount of data for this day is too small, then just skip
        if day_length < self.min_day_length:
            return None

        self.log.info("Constructing LSD:%i [%i files]", lsd, len(self._timestream_list))

        # Construct the combined timestream
        ts = tod.concatenate(self._timestream_list)

        # Add attributes for the LSD and a tag for labelling saved files
        ts.attrs["tag"] = "lsd_%i" % lsd
        ts.attrs["lsd"] = lsd

        # Clear the timestream list since these days have already been processed
        self._timestream_list = []

        return ts


class SiderealDirtyRegridder(Regridder):
    """Take a factorized sidereal day and put it on a regular grid.

    Output container includes a dirty estimate of visibilities on a regular
    grid and an estimate of the inverse convolution matrix Ci based on a limited
    number of modes over baselines.
    """

    def setup(self, manager):
        """Set the local observers position.

        Parameters
        ----------
        manager : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        # Need an Observer object holding the geographic location of the telescope.
        self.observer = io.get_telescope(manager)
        self.pad = 5 * self.lanczos_width

    def process(
        self, data: containers.FactorizedTimeStream
    ) -> containers.SiderealDirtyStream:
        """Make a dirty projection of the sidereal day.

        Parameters
        ----------
        data
            Timestream data with factorized weights. Weights can be factorized
            using `draco.analysis.transform.FactorizeWeights`.

        Returns
        -------
        sdata
            Sidereal stream with dirty `vis` projection and factorized
            inverse signal covariance matrix
        """
        self.log.info(f"Making dirty grid LSD:{data.attrs['lsd']}")

        # Redistribute if needed
        data.redistribute("freq")

        # Convert data timestamps into LSD deltas (relative to the LSD of this day)
        timestamp_lsd = self.observer.unix_to_lsd(data.time) - data.attrs["lsd"]

        # Get view of data
        Ni = data.weight[:].local_array
        modes = data.modes[:].local_array
        vis_data = data.vis[:].local_array

        xh, Ci, nw = self._regrid(vis_data, Ni, modes, timestamp_lsd)

        # This will be padded so we have to extend the RA axis accordingly
        new_samples = self.samples + 2 * self.pad
        ra_delta = ((new_samples / self.samples) * 360 - 360) / 2
        ra = np.linspace(-ra_delta, 360 + ra_delta, new_samples, endpoint=False)

        # Make the new container
        sdata = containers.SiderealDirtyStream(
            axes_from=data,
            ra=ra,
            bandwidth=2 * self.lanczos_width,
        )
        sdata.add_dataset("vis_weight")
        sdata.redistribute("freq")

        sdata.vis[:].local_array[:] = xh
        sdata.signal_cov[:].local_array[:] = Ci
        sdata.modes[:].local_array[:] = modes
        sdata.weight[:].local_array[:] = nw

        sdata.attrs["lsd"] = data.attrs["lsd"]
        sdata.attrs["tag"] = f"lsd_{data.attrs['lsd']}"
        # Store this so it can be removed later on
        sdata.attrs["pad"] = self.pad

        return sdata

    def _regrid(self, vis_data, weight, modes, times):
        """Project the visibility data onto a regular grid in RA."""
        # Create a regular grid, padded at either end to supress interpolation issues
        interp_grid = (
            np.arange(-self.pad, self.samples + self.pad, dtype=np.float64)
            / self.samples
        )

        # Construct regridding matrix for reverse problem
        lzf = regrid.lanczos_forward_matrix(
            interp_grid, times, self.lanczos_width
        ).T.copy()

        # Make the signal covariance matrix before reshaping since the
        # stack axis will be reduced
        Ci = regrid.wiener_covariance(lzf, weight, 2 * self.lanczos_width - 1)
        # Store the final shape of the data and flatten across
        # frequency and baselines
        shape_ = (*vis_data.shape[:-1], interp_grid.shape[0])
        # Reconstruct and flatten the weights
        weight = (modes[:, :, np.newaxis] @ weight[:, np.newaxis]).reshape(
            -1, weight.shape[-1]
        )
        # Make the dirty projection into signal space
        vis_data = vis_data.reshape(-1, vis_data.shape[-1])
        xh = regrid.wiener_projection(lzf, vis_data, weight).reshape(shape_)
        nw = (modes[:, :, np.newaxis] @ Ci[:, -1][:, np.newaxis]).reshape(shape_)

        return xh, Ci, nw


class RemoveDirtyWeights(task.SingleTask):
    """Remove weights dataset from a dirty stream."""

    def process(self, data: containers.SiderealDirtyStream):
        """Remove the weights."""
        out = containers.SiderealDirtyStream(attrs_from=data, axes_from=data)
        out.vis[:] = data.vis[:]
        out.signal_cov[:] = data.signal_cov[:]
        out.modes[:] = data.modes[:]

        return out


class SiderealGridDeconvolve(task.SingleTask):
    """Deconvolve a single dirty sidereal day.

    Attributes
    ----------
    snr_cov: float
        Ratio of signal covariance to noise covariance (used for Wiener filter).
    """

    snr_cov = config.Property(proptype=float, default=1e-8)

    def process(
        self, data: containers.SiderealDirtyStream
    ) -> containers.SiderealStream:
        """Deconvolve a dirty sidereal day.

        Parameters
        ----------
        data
            Dirty sidereal data to deconvolve

        Returns
        -------
        sdata
            Deconvolved sidereal day with padding removed.
        """
        self.log.info(f"Deconvolving dirty grid LSD:{data.attrs['lsd']}")

        # Redistribute if needed
        data.redistribute("freq")

        Ci = data.signal_cov[:].local_array
        xh = data.vis[:].local_array
        modes = data.modes[:].local_array

        pad = data.attrs["pad"]

        # Deconvolve the visibilities
        xh, nr, samples = self._deconvolve(xh, Ci, modes, pad)

        sdata = containers.SiderealStream(axes_from=data, ra=samples)
        sdata.redistribute("freq")

        # Save out the deconvolved visibilities and noise realization
        sdata.vis[:] = xh
        sdata.weight[:] = nr

        sdata.attrs["lsd"] = data.attrs["lsd"]
        sdata.attrs["tag"] = f"lsd_{data.attrs['lsd']}"

        return sdata

    def _deconvolve(self, xh, Ci, modes, pad):
        """Deconvolve the data."""
        nbaseline = xh.shape[1]
        # Get number of RA samples and target shape without padding
        samples = xh.shape[-1] - 2 * pad
        shape_ = xh.shape[:-1] + (samples,)
        # Flatten over frequencies and baselines
        xh = xh.reshape(-1, xh.shape[-1])
        modes = modes.reshape(-1)
        nw = np.zeros_like(xh, dtype=np.float32)

        # Iterate over frequency-baseline pairs
        for ki in range(xh.shape[0]):
            # Get the reconstruction of Ci for this frequency-baseline
            Ci_ki = Ci[ki // nbaseline] * modes[ki]
            # Set the weights and remove the signal contribution
            nw[ki] = Ci_ki[-1]
            Ci_ki[-1] += self.snr_cov
            # Solve
            xh[ki] = la.solveh_banded(Ci_ki, xh[ki])

        # Remove padding and reshape
        xh = xh[:, pad:-pad].reshape(shape_)
        nw = nw[:, pad:-pad].reshape(shape_)

        return xh, nw, samples


class SiderealRegridder(Regridder):
    """Take a sidereal days worth of data and put it onto a regular grid.

    Uses a maximum-likelihood inverse of a Lanczos interpolation to do the
    regridding. This gives a reasonably local regridding, that is pretty well
    behaved in m-space.

    Attributes
    ----------
    samples : int
        Number of samples across the sidereal day.
    lanczos_width : int
        Width of the Lanczos interpolation kernel.
    snr_cov: float
        Ratio of signal covariance to noise covariance (used for Wiener filter).
    down_mix: bool
        Down mix the visibility prior to interpolation using the fringe rate
        of a source at zenith.  This is un-done after the interpolation.
    """

    down_mix = config.Property(proptype=bool, default=False)

    def setup(self, manager):
        """Set the local observers position.

        Parameters
        ----------
        manager : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        # Need an Observer object holding the geographic location of the telescope.
        self.observer = io.get_telescope(manager)

    def process(self, data):
        """Regrid the sidereal day.

        Parameters
        ----------
        data : containers.TimeStream
            Timestream data for the day (must have a `LSD` attribute).

        Returns
        -------
        sdata : containers.SiderealStream
            The regularly gridded sidereal timestream.
        """
        self.log.info(f"Regridding LSD:{data.attrs['lsd']}")

        # Redistribute if needed
        data.redistribute("freq")
        freq = data.freq[data.vis[:].local_bounds]

        # Convert data timestamps into LSDs
        timestamp_lsd = self.observer.unix_to_lsd(data.time)

        # Fetch which LSD this is to set bounds
        self.start = data.attrs["lsd"]
        self.end = self.start + 1

        # Get view of data
        weight = data.weight[:].local_array
        vis_data = data.vis[:].local_array

        # Mix down
        if self.down_mix:
            self.log.info("Downmixing before regridding.")
            phase = self._get_phase(freq, data.prodstack, timestamp_lsd)
            vis_data *= phase

        # perform regridding
        new_grid, sts, ni = self._regrid(vis_data, weight, timestamp_lsd)

        # Mix back up
        if self.down_mix:
            phase = self._get_phase(freq, data.prodstack, new_grid).conj()
            sts *= phase
            ni *= (np.abs(phase) > 0.0).astype(ni.dtype)

        # Wrap to produce MPIArray
        sts = mpiarray.MPIArray.wrap(sts, axis=0)
        ni = mpiarray.MPIArray.wrap(ni, axis=0)

        # FYI this whole process creates an extra copy of the sidereal stack.
        # This could probably be optimised out with a little work.
        sdata = containers.SiderealStream(axes_from=data, ra=self.samples)
        sdata.redistribute("freq")

        sdata.vis[:] = sts
        sdata.weight[:] = ni
        sdata.attrs["lsd"] = self.start
        sdata.attrs["tag"] = f"lsd_{self.start}"

        return sdata

    def _get_phase(self, freq, prod, lsd):
        # Determine if any baselines contains masked feeds
        # These baselines will be flagged since they do not
        # have valid baseline distances.
        aa, bb = prod["input_a"], prod["input_b"]

        mask = self.observer.feedmask[(aa, bb)].astype(np.float32)[
            np.newaxis, :, np.newaxis
        ]

        # Calculate the fringe rate assuming that ha = 0.0 and dec = lat
        lmbda = units.c / (freq * 1e6)
        u = self.observer.baselines[np.newaxis, :, 0] / lmbda[:, np.newaxis]

        omega = -2.0 * np.pi * u * np.cos(np.radians(self.observer.latitude))

        # Calculate the local sidereal angle
        dphi = 2.0 * np.pi * (lsd - np.floor(lsd))

        # Construct a complex sinusoid whose frequency
        # is equal to each baselines fringe rate
        return mask * np.exp(
            -1.0j * omega[:, :, np.newaxis] * dphi[np.newaxis, np.newaxis, :]
        )


def _search_nearest(x, xeval):
    index_next = np.searchsorted(x, xeval, side="left")

    index_previous = np.maximum(0, index_next - 1)
    index_next = np.minimum(x.size - 1, index_next)

    return np.where(
        np.abs(xeval - x[index_previous]) < np.abs(xeval - x[index_next]),
        index_previous,
        index_next,
    )


class SiderealRegridderNearest(SiderealRegridder):
    """Regrid onto the sidereal day using nearest neighbor interpolation."""

    def _regrid(self, vis, weight, lsd):
        # Create a regular grid
        interp_grid = np.arange(0, self.samples, dtype=np.float64) / self.samples
        interp_grid = interp_grid * (self.end - self.start) + self.start

        # Find the data points that are closest to the fixed points on the grid
        index = _search_nearest(lsd, interp_grid)

        interp_vis = vis[..., index]
        interp_weight = weight[..., index]

        # Flag the re-gridded data if the nearest neighbor was more than one
        # sample spacing away.  This can occur if the input data does not have
        # complete sidereal coverage.
        delta = np.median(np.abs(np.diff(lsd)))
        distant = np.flatnonzero(np.abs(lsd[index] - interp_grid) > delta)
        interp_weight[..., distant] = 0.0

        return interp_grid, interp_vis, interp_weight


class SiderealRegridderLinear(SiderealRegridder):
    """Regrid onto the sidereal day using linear interpolation."""

    def _regrid(self, vis, weight, lsd):
        # Create a regular grid
        interp_grid = np.arange(0, self.samples, dtype=np.float64) / self.samples
        interp_grid = interp_grid * (self.end - self.start) + self.start

        # Find the data points that lie on either side of each point in the fixed grid
        index = np.searchsorted(lsd, interp_grid, side="left")

        ind1 = index - 1
        ind2 = index

        # If the fixed grid is outside the range covered by the data,
        # then we will extrapolate and later flag as bad.
        below = np.flatnonzero(ind1 == -1)
        if below.size > 0:
            ind1[below] = 0
            ind2[below] = 1

        above = np.flatnonzero(ind2 == lsd.size)
        if above.size > 0:
            ind1[above] = lsd.size - 2
            ind2[above] = lsd.size - 1

        # If the closest data points to the fixed grid point are more than one
        # sample spacing away, then we will later flag that data as bad.
        # This will occur if the input data does not cover the full sidereal day.
        delta = np.median(np.abs(np.diff(lsd)))
        distant = np.flatnonzero(
            (np.abs(lsd[ind1] - interp_grid) > delta)
            | (np.abs(lsd[ind2] - interp_grid) > delta)
        )

        # Calculate the coefficients for the linear interpolation
        dx1 = interp_grid - lsd[ind1]
        dx2 = lsd[ind2] - interp_grid

        norm = tools.invert_no_zero(dx1 + dx2)
        coeff1 = dx2 * norm
        coeff2 = dx1 * norm

        # Initialize the output arrays
        shp = vis.shape[:-1] + (self.samples,)

        interp_vis = np.zeros(shp, dtype=vis.dtype)
        interp_weight = np.zeros(shp, dtype=weight.dtype)

        # Loop over frequencies to reduce memory usage
        for ff in range(shp[0]):
            fvis = vis[ff]
            fweight = weight[ff]

            # Consider the data valid if it has nonzero weight
            fflag = fweight > 0.0

            # Determine the variance from the inverse weight
            fvar = tools.invert_no_zero(fweight)

            # Require both data points to be valid for the interpolated value to be valid
            finterp_flag = fflag[:, ind1] & fflag[:, ind2]

            # Interpolate the visibilities and propagate the weights
            interp_vis[ff] = coeff1 * fvis[:, ind1] + coeff2 * fvis[:, ind2]

            interp_weight[ff] = tools.invert_no_zero(
                coeff1**2 * fvar[:, ind1] + coeff2**2 * fvar[:, ind2]
            ) * finterp_flag.astype(np.float32)

        # Flag as bad any values that were extrapolated or that used distant points
        interp_weight[..., below] = 0.0
        interp_weight[..., above] = 0.0
        interp_weight[..., distant] = 0.0

        return interp_grid, interp_vis, interp_weight


class SiderealRegridderCubic(SiderealRegridder):
    """Regrid onto the sidereal day using cubic Hermite spline interpolation."""

    def _regrid(self, vis, weight, lsd):
        # Create a regular grid
        interp_grid = np.arange(0, self.samples, dtype=np.float64) / self.samples
        interp_grid = interp_grid * (self.end - self.start) + self.start

        # Find the data point just after each point on the fixed grid
        index = np.searchsorted(lsd, interp_grid, side="left")

        # Find the 4 data points that will be used to interpolate
        # each point on the fixed grid
        index = np.vstack([index + i for i in range(-2, 2)])

        # If the fixed grid is outside the range covered by the data,
        # then we will extrapolate and later flag as bad
        below = np.flatnonzero(np.any(index < 0, axis=0))
        if below.size > 0:
            index = np.maximum(index, 0)

        above = np.flatnonzero(np.any(index >= lsd.size, axis=0))
        if above.size > 0:
            index = np.minimum(index, lsd.size - 1)

        # If the closest data points to the fixed grid point are more than one
        # sample spacing away, then we will later flag that data as bad.
        # This will occur if the input data does not cover the full sidereal day.
        delta = np.median(np.abs(np.diff(lsd)))
        distant = np.flatnonzero(
            np.any(np.abs(interp_grid - lsd[index]) > (2.0 * delta), axis=0)
        )

        # Calculate the coefficients for the interpolation
        u = (interp_grid - lsd[index[1]]) * tools.invert_no_zero(
            lsd[index[2]] - lsd[index[1]]
        )

        coeff = np.zeros((4, u.size), dtype=np.float64)
        coeff[0] = u * ((2 - u) * u - 1)
        coeff[1] = u**2 * (3 * u - 5) + 2
        coeff[2] = u * ((4 - 3 * u) * u + 1)
        coeff[3] = u**2 * (u - 1)
        coeff *= 0.5

        # Initialize the output arrays
        shp = vis.shape[:-1] + (self.samples,)

        interp_vis = np.zeros(shp, dtype=vis.dtype)
        interp_weight = np.zeros(shp, dtype=weight.dtype)

        # Loop over frequencies to reduce memory usage
        for ff in range(shp[0]):
            fvis = vis[ff]
            fweight = weight[ff]

            # Consider the data valid if it has nonzero weight
            fflag = fweight > 0.0

            # Determine the variance from the inverse weight
            fvar = tools.invert_no_zero(fweight)

            # Interpolate the visibilities and propagate the weights
            finterp_flag = np.ones(shp[1:], dtype=bool)
            finterp_var = np.zeros(shp[1:], dtype=weight.dtype)

            for ii, cc in zip(index, coeff):
                finterp_flag &= fflag[:, ii]
                finterp_var += cc**2 * fvar[:, ii]

                interp_vis[ff] += cc * fvis[:, ii]

            # Invert the accumulated variances to get the weight
            # Require all data points are valid for the interpolated value to be valid
            interp_weight[ff] = tools.invert_no_zero(finterp_var) * finterp_flag.astype(
                np.float32
            )

        # Flag as bad any values that were extrapolated or that used distant points
        interp_weight[..., below] = 0.0
        interp_weight[..., above] = 0.0
        interp_weight[..., distant] = 0.0

        return interp_grid, interp_vis, interp_weight


class SiderealStacker(task.SingleTask):
    """Take in a set of sidereal days, and stack them up.

    Also computes the variance over sideral days using an
    algorithm that updates the sum of square differences from
    the current mean, which is less prone to numerical issues.
    See West, D.H.D. (1979). https://doi.org/10.1145/359146.359153.

    Attributes
    ----------
    tag : str (default: "stack")
        The tag to give the stack.
    weight: str (default: "inverse_variance")
        The weighting to use in the stack.
        Either `uniform` or `inverse_variance`.
    with_sample_variance : bool (default: False)
        Add a dataset containing the sample variance
        of the visibilities over sidereal days to the
        sidereal stack.
    """

    stack = None

    tag = config.Property(proptype=str, default="stack")
    weight = config.enum(["uniform", "inverse_variance"], default="inverse_variance")
    with_sample_variance = config.Property(proptype=bool, default=False)

    def process(self, sdata):
        """Stack up sidereal days.

        Parameters
        ----------
        sdata : containers.SiderealStream
            Individual sidereal day to add to stack.
        """
        sdata.redistribute("freq")

        # Get the LSD (or CSD) label out of the input's attributes.
        # If there is no label, use a placeholder.
        if "lsd" in sdata.attrs:
            input_lsd = sdata.attrs["lsd"]
        elif "csd" in sdata.attrs:
            input_lsd = sdata.attrs["csd"]
        else:
            input_lsd = -1

        input_lsd = _ensure_list(input_lsd)

        # If this is our first sidereal day, then initialize the
        # container that will hold the stack.
        if self.stack is None:
            self.stack = containers.empty_like(sdata)

            # Add stack-specific datasets
            if "nsample" not in self.stack.datasets:
                self.stack.add_dataset("nsample")

            if self.with_sample_variance and (
                "sample_variance" not in self.stack.datasets
            ):
                self.stack.add_dataset("sample_variance")

            self.stack.redistribute("freq")

            # Initialize all datasets to zero.
            for data in self.stack.datasets.values():
                data[:] = 0

            self.lsd_list = []

            # Keep track of the sum of squared weights
            # to perform Bessel's correction at the end.
            if self.with_sample_variance:
                self.sum_coeff_sq = np.zeros_like(self.stack.weight[:].view(np.ndarray))

        # Accumulate
        self.log.info("Adding to stack LSD(s): %s" % input_lsd)

        self.lsd_list += input_lsd

        if "nsample" in sdata.datasets:
            # The input sidereal stream is already a stack
            # over multiple sidereal days. Use the nsample
            # dataset as the weight for the uniform case.
            count = sdata.nsample[:]
        else:
            # The input sidereal stream contains a single
            # sidereal day.  Use a boolean array that
            # indicates a non-zero weight dataset as
            # the weight for the uniform case.
            dtype = self.stack.nsample.dtype
            count = (sdata.weight[:] > 0.0).astype(dtype)

        # Determine the weights to be used in the average.
        if self.weight == "uniform":
            coeff = count.astype(np.float32)
            # Accumulate the variances in the stack.weight dataset.
            self.stack.weight[:] += (coeff**2) * tools.invert_no_zero(sdata.weight[:])
        else:
            coeff = sdata.weight[:]
            # We are using inverse variance weights.  In this case,
            # we accumulate the inverse variances in the stack.weight
            # dataset.  Do that directly to avoid an unneccessary
            # division in the more general expression above.
            self.stack.weight[:] += sdata.weight[:]

        # Accumulate the total number of samples.
        self.stack.nsample[:] += count

        # Below we will need to normalize by the current sum of coefficients.
        # Can be found in the stack.nsample dataset for uniform case or
        # the stack.weight dataset for inverse variance case.
        if self.weight == "uniform":
            sum_coeff = self.stack.nsample[:].astype(np.float32)
        else:
            sum_coeff = self.stack.weight[:]

        # Calculate weighted difference between the new data and the current mean.
        delta_before = coeff * (sdata.vis[:] - self.stack.vis[:])

        # Update the mean.
        self.stack.vis[:] += delta_before * tools.invert_no_zero(sum_coeff)

        # The calculations below are only required if the sample variance was requested
        if self.with_sample_variance:
            # Accumulate the sum of squared coefficients.
            self.sum_coeff_sq += coeff**2

            # Calculate the difference between the new data and the updated mean.
            delta_after = sdata.vis[:] - self.stack.vis[:]

            # Update the sample variance.
            self.stack.sample_variance[0] += delta_before.real * delta_after.real
            self.stack.sample_variance[1] += delta_before.real * delta_after.imag
            self.stack.sample_variance[2] += delta_before.imag * delta_after.imag

    def process_finish(self):
        """Normalize the stack and return the result.

        Returns
        -------
        stack : containers.SiderealStream
            Stack of sidereal days.
        """
        self.stack.attrs["tag"] = self.tag
        self.stack.attrs["lsd"] = np.array(self.lsd_list)

        # For uniform weighting, normalize the accumulated variances by the total
        # number of samples squared and then invert to finalize stack.weight.
        if self.weight == "uniform":
            norm = self.stack.nsample[:].astype(np.float32)
            self.stack.weight[:] = tools.invert_no_zero(self.stack.weight[:]) * norm**2

        # We need to normalize the sample variance by the sum of coefficients.
        # Can be found in the stack.nsample dataset for uniform case
        # or the stack.weight dataset for inverse variance case.
        if self.with_sample_variance:
            if self.weight != "uniform":
                norm = self.stack.weight[:]

            # Perform Bessel's correction.  In the case of
            # uniform  weighting, norm will be equal to nsample - 1.
            norm = norm - self.sum_coeff_sq * tools.invert_no_zero(norm)

            # Normalize the sample variance.
            self.stack.sample_variance[:] *= np.where(
                self.stack.nsample[:] > 1, tools.invert_no_zero(norm), 0.0
            )[np.newaxis, :]

        return self.stack


class SiderealStackerMatch(task.SingleTask):
    """Take in a set of sidereal days, and stack them up.

    This treats the time average of each input sidereal stream as an extra source of
    noise and uses a Wiener filter approach to consistent stack the individual streams
    together while accounting for their distinct coverage in RA. In practice this is
    used for co-adding stacks with different sidereal coverage while marginalising out
    the effects of the different cross talk contributions that each input stream may
    have.

    There is no uniquely correct solution for the sidereal average (or m=0 mode) of the
    output stream. This task fixes this unknown mode by setting the *median* of each 24
    hour period to zero. Note this is not the same as setting the m=0 mode to be zero.

    Parameters
    ----------
    tag : str
        The tag to give the stack.
    """

    stack = None
    lsd_list = None

    tag = config.Property(proptype=str, default="stack")

    count = 0

    def process(self, sdata):
        """Stack up sidereal days.

        Parameters
        ----------
        sdata : containers.SiderealStream
            Individual sidereal day to stack up.
        """
        sdata.redistribute("freq")

        if self.stack is None:
            self.log.info("Starting new stack.")

            self.stack = containers.empty_like(sdata)
            self.stack.redistribute("freq")
            self.stack.vis[:] = 0.0
            self.stack.weight[:] = 0.0

            self.count = 0
            self.Ni_s = mpiarray.zeros(
                (sdata.weight.shape[0], sdata.weight.shape[2]),
                axis=0,
                comm=sdata.comm,
                dtype=np.float64,
            )
            self.Vm = []
            self.lsd_list = []

        label = sdata.attrs.get("tag", f"stream_{self.count}")
        self.log.info(f"Adding {label} to stack.")

        # Get an estimate of the noise inverse for each time and freq in the file.
        # Average over baselines as we don't have the memory
        Ni_d = sdata.weight[:].mean(axis=1)

        # Calculate the trace of the inverse noise covariance for each frequency
        tr_Ni = Ni_d.sum(axis=1)

        # Calculate the projection vector v
        v = Ni_d * tools.invert_no_zero(tr_Ni[:, np.newaxis]) ** 0.5

        d = sdata.vis[:]

        # Calculate and store the dirty map in the stack container
        self.stack.vis[:] += (
            d * Ni_d[:, np.newaxis, :]
            # - v[:, np.newaxis, :] * np.dot(sdata.vis[:], v.T)[..., np.newaxis]
            - v[:, np.newaxis, :]
            * np.matmul(v[:, np.newaxis, np.newaxis, :], d[..., np.newaxis])[..., 0]
        )

        # Propagate the transformation into the weights, but for the moment we need to
        # store the variance. We don't propagate the change coming from matching the
        # means, as it is small, and the effects are primarily on the off-diagonal
        # covariance entries that we don't store anyway
        self.stack.weight[:] += (
            tools.invert_no_zero(sdata.weight[:]) * Ni_d[:, np.newaxis, :] ** 2
        )

        # Accumulate the total inverse noise
        self.Ni_s += Ni_d

        # We need to keep the projection vector until the end
        self.Vm.append(v)

        # Get the LSD label out of the data (resort to using a CSD if it's
        # present). If there's no label just use a place holder and stack
        # anyway.
        if "lsd" in sdata.attrs:
            input_lsd = sdata.attrs["lsd"]
        elif "csd" in sdata.attrs:
            input_lsd = sdata.attrs["csd"]
        else:
            input_lsd = -1
        self.lsd_list += _ensure_list(input_lsd)

        self.count += 1

    def process_finish(self):
        """Construct and emit sidereal stack.

        Returns
        -------
        stack : containers.SiderealStream
            Stack of sidereal days.
        """
        self.stack.attrs["tag"] = self.tag

        Va = np.array(self.Vm).transpose(1, 2, 0)

        # Dereference for efficiency to avoid MPI calls in the loop
        sv = self.stack.vis[:].local_array
        sw = self.stack.weight[:].local_array

        # Loop over all frequencies to do the deconvolution. The loop is done because
        # of the difficulty mapping the operations we would want to do into what numpy
        # allows
        for lfi in range(self.stack.vis[:].local_shape[0]):
            Ni_s = self.Ni_s.local_array[lfi]
            N_s = tools.invert_no_zero(Ni_s)
            V = Va[lfi] * N_s[:, np.newaxis]

            # Note, need to use a pseudo-inverse in here as there is a singular mode
            A = la.pinv(
                np.identity(self.count) - np.dot(V.T, Ni_s[:, np.newaxis] * V),
                rcond=1e-8,
            )

            # Perform the deconvolution step
            sv[lfi] = sv[lfi] * N_s + np.dot(V, np.dot(A, np.dot(sv[lfi], V).T)).T

            # Normalise the weights
            sw[lfi] = tools.invert_no_zero(sw[lfi]) * Ni_s**2

        # Remove the full day median to set a well defined normalisation, otherwise the
        # mean is undefined
        stack_median = np.median(sv.real, axis=2) + np.median(sv.imag, axis=2) * 1.0j
        sv -= stack_median[:, :, np.newaxis]

        # Set the full LSD list
        self.stack.attrs["lsd"] = np.array(self.lsd_list)

        # Delete a couple of large attributes
        del self.Vm
        del self.Ni_s

        return self.stack


class SiderealStackerDeconvolve(SiderealGridDeconvolve):
    """Stack up a set of dirty sidereal days and deconvolve the final product.

    Attributes
    ----------
    tag : str (default: "stack")
        The tag to give the stack.
    subtract_median : bool
        Subtract the median of the final visibilities in RA.
    """

    tag = config.Property(proptype=str, default="stack")
    subtract_median = config.Property(proptype=bool, default=True)

    stack = None

    def process(self, data: containers.SiderealDirtyStream):
        """Stack up the dirty sidereal days and noise matrices.

        Parameters
        ----------
        data
            Individual sidereal day to add to stack.
        """
        data.redistribute("freq")

        xh = data.vis[:].local_array
        Ci = data.signal_cov[:].local_array
        modes = data.modes[:].local_array

        if self.stack is None:
            # Accumulate stuff here. Assume that all stacked days have
            # the same padding
            self.xh = np.zeros_like(xh)
            self.Ci = []
            self.modes = []
            self.pad = data.attrs["pad"]
            # Make the correct RA axis since input is padded
            samples = xh.shape[-1] - 2 * self.pad
            ra = np.linspace(0, 360, samples, endpoint=False)
            # Don't initialize any datasets for now to save memory
            self.stack = containers.SiderealStream(
                axes_from=data, ra=ra, attrs_from=data, skip_datasets=True
            )
            self.lsd_list = []

        # Accumulate the weighted visibilities
        self.xh += xh
        # Accumulate the signal covariance matrix and modes
        # We need all of these
        self.Ci.append(Ci)
        self.modes.append(modes)
        # Get the LSD/CSD label if available
        if "lsd" in data.attrs:
            input_lsd = data.attrs["lsd"]
        else:
            input_lsd = data.attrs.get("csd", -1)
        self.lsd_list += _ensure_list(input_lsd)

    def process_finish(self) -> containers.SiderealStream:
        """Deconvolve and return the final stacked sidereal stream.

        Returns
        -------
        stack
            Deconvolved stack of sidereal days.
        """
        # Log how much data is missing, excluding bands that are entirely
        # masked due to persistent RFI
        zeros = mpiarray.MPIArray.wrap(self.xh[:] == 0, axis=0, comm=self.stack.comm)
        zeros.local_array[:] &= ~np.all(zeros.local_array, axis=-1)[..., np.newaxis]
        n_zeros = zeros.sum().allreduce()
        self.log.info(
            f"{100 * n_zeros / np.prod(zeros.global_shape):.3f}% of samples missing."
        )

        xh, nw = self._deconvolve(self.xh, self.Ci, self.modes, self.pad)

        # Delete some large items to save memory
        del self.xh
        del self.Ci
        del self.modes

        if self.subtract_median:
            # Subtract the stack median over RA
            mask = (nw != 0).astype(np.float32)
            # Use a weighted median to ignore partially filled bands
            xh_h = (
                weighted_median.weighted_median(np.ascontiguousarray(xh.real), mask)
                + weighted_median.weighted_median(np.ascontiguousarray(xh.imag), mask)
                * 1.0j
            )
            # Subtract the median
            xh -= xh_h[:, :, np.newaxis]
            # Make sure that zeros are still zeros
            xh *= mask

        # Initialize the datasets now
        self.stack.add_dataset("vis")
        self.stack.add_dataset("vis_weight")
        self.stack.add_dataset("input_flags")
        self.stack.redistribute("freq")

        self.stack.vis[:].local_array[:] = xh
        self.stack.weight[:].local_array[:] = nw
        self.stack.input_flags[:] = 0.0

        self.stack.attrs["lsd"] = np.array(self.lsd_list)
        self.stack.attrs["tag"] = self.tag

        return self.stack

    def _deconvolve(self, xh, Ci, modes, pad):
        """Deconvolve the dirty visibilities.

        Reconstructs each Ci matrix on a per frequency-baseline basis
        to save memory.
        """
        # Stack over the last axis
        Ci = np.stack(Ci, axis=-1)
        modes = np.stack(modes, axis=-1)

        nbaseline = xh.shape[1]
        samples = xh.shape[-1] - 2 * pad
        shape_ = xh.shape[:-1] + (samples,)

        # Flatten over frequencies and baselines
        xh = xh.reshape(-1, xh.shape[-1])
        modes = modes.reshape(-1, modes.shape[-1])
        nw = np.zeros_like(xh, dtype=np.float32)

        for ki in range(xh.shape[0]):
            # Reconstruct Ci over modes and sum over the stack (last) axis,
            # which can be quickly implemented with `matmul`
            Ci_ki = np.matmul(Ci[ki // nbaseline], modes[ki])
            # Save out the noise projection and add the signal component
            nw[ki] = Ci_ki[-1]
            Ci_ki[-1] += self.snr_cov

            xh[ki] = la.solveh_banded(Ci_ki, xh[ki])

        xh = xh[:, pad:-pad].reshape(shape_)
        nw = nw[:, pad:-pad].reshape(shape_)

        return xh, nw


def _ensure_list(x):
    if hasattr(x, "__iter__"):
        y = [xx for xx in x]  # noqa: C416
    else:
        y = [x]

    return y

"""Take timestream data and regridding it into sidereal days which can be stacked.

Usage
=====

Generally you would want to use these tasks together. Sending time stream data
into  :class:`SiderealGrouper`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`SiderealStacker` if you want to combine the different days.
"""

import inspect

import numpy as np
import scipy.linalg as la
from caput import config, mpiarray, tod
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
        ts.attrs["tag"] = f"lsd_{lsd:d}"
        ts.attrs["lsd"] = lsd

        # Clear the timestream list since these days have already been processed
        self._timestream_list = []

        return ts


class SiderealRegridder(Regridder):
    """Take a sidereal days worth of data, and put onto a regular grid.

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
        self.log.info("Regridding LSD:%i", data.attrs["lsd"])

        # Redistribute if needed too
        data.redistribute("freq")

        sfreq = data.vis.local_offset[0]
        efreq = sfreq + data.vis.local_shape[0]
        freq = data.freq[sfreq:efreq]

        # Convert data timestamps into LSDs
        timestamp_lsd = self.observer.unix_to_lsd(data.time)

        # Fetch which LSD this is to set bounds
        self.start = data.attrs["lsd"]
        self.end = self.start + 1

        # Get view of data
        weight = data.weight[:].view(np.ndarray)
        vis_data = data.vis[:].view(np.ndarray)

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
        sdata.attrs["tag"] = f"lsd_{self.start:.0f}"

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
        shp = (*vis.shape[:-1], self.samples)

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
        shp = (*vis.shape[:-1], self.samples)

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


class SiderealRebinner(SiderealRegridder):
    """Regrid a sidereal day of data using a binning method.

    Assign a fraction of each time sample to the nearest RA bin based
    on the propotion of the time bin that overlaps the RA bin.

    Tracks the weighted effective centre of RA bin so that a centring
    correction can be applied afterwards. A correction option is
    implemented in `RebinGradientCorrection`.

    Parameters
    ----------
    weight: str (default: "inverse_variance")
        The weighting to use in the stack.  Either `uniform` or `inverse_variance`.
    """

    weight = config.enum(["uniform", "inverse_variance"], default="inverse_variance")

    def process(self, data):
        """Rebin the sidereal day.

        Parameters
        ----------
        data : containers.TimeStream
            Timestream data for the day (must have a `LSD` attribute).

        Returns
        -------
        sdata : containers.SiderealStream
            The regularly gridded sidereal timestream.
        """
        import scipy.sparse as ss

        self.log.info(
            f"Rebinning LSD {data.attrs['lsd']:.0f} with {self.weight} weighting."
        )

        # Determine output container based on input container
        container_map = {
            containers.TimeStream: containers.SiderealStream,
            containers.HybridVisStream: containers.HybridVisStream,
        }

        # We need to be able to check for subclasses in the container map
        for cls in inspect.getmro(data.__class__):
            OutputContainer = container_map.get(cls)

            if OutputContainer is not None:
                break

        if OutputContainer is None:
            raise TypeError(
                f"No valid container mapping.\nGot {data.__class__}.\n"
                f"Mappings exist for {list(container_map.keys())}."
            )

        # Redistribute if needed too
        data.redistribute("freq")

        # Fetch which LSD this is to set bounds
        self.start = data.attrs["lsd"]
        self.end = self.start + 1

        # Convert data timestamps into LSDs
        if "ra" in data.index_map:
            timestamp_lsd = self.start + data.ra / 360.0
        else:
            timestamp_lsd = self.observer.unix_to_lsd(data.time)

        # Create the output container
        sdata = OutputContainer(ra=self.samples, axes_from=data, attrs_from=data)

        # Initialize any missing datasets
        alt_dspec = {}
        contains_covariance = False
        for name, dataset in data.datasets.items():
            if name not in sdata.datasets:
                alt_dspec[name] = dataset.attrs["axis"]
                sdata.add_dataset(name)
                if "freq_cov" in name:
                    contains_covariance = True

        sdata.add_dataset("effective_ra")
        sdata.add_dataset("nsample")

        sdata.redistribute("freq")

        # Get view of data
        weight = data.weight[:].local_array
        vis_data = data.vis[:].local_array
        alt_data = {
            name: data.datasets[name][:].local_array for name in alt_dspec.keys()
        }

        # Get the median time sample width
        width_t = np.median(np.abs(np.diff(timestamp_lsd)))
        # Create the regular grid of RA samples
        target_lsd = np.linspace(self.start, self.end, self.samples, endpoint=False)
        # Create the rebinning matrix
        R = regrid.rebin_matrix(timestamp_lsd, target_lsd, width_t=width_t)
        Rt = ss.csr_array(R.T)
        # The square is used to calculate rebinned weights
        Rtsq = Rt.power(2)

        # dereference arrays before loop
        sera = sdata.datasets["effective_ra"][:].local_array
        ssw = sdata.weight[:].local_array
        ssv = sdata.vis[:].local_array
        ssn = sdata.nsample[:].local_array
        salt = {name: sdata.datasets[name][:].local_array for name in alt_dspec.keys()}

        vax = data.vis.attrs["axis"][:-2]
        lookup = {name: nn for nn, name in enumerate(vax)}

        # If the input data product contains a freq-freq covariance matrix,
        # then we will need access to the weight dataset for all frequencies.
        if contains_covariance:
            if self.weight == "uniform":
                weight_all = (data.weight[:] > 0.0).allgather()
            else:
                weight_all = data.weight[:].allgather()

        # Loop over all but the last two axes.
        # For an input TimeStream, this will be a loop over freq.
        # For an input HybridVisStream, this will be a loop over (pol, freq, ew).
        for ind in np.ndindex(*vis_data.shape[:-2]):

            w = weight[ind]
            m = (w > 0.0).astype(np.float32)
            if self.weight == "uniform":
                v = tools.invert_no_zero(w)
                w = m
            else:
                v = w

            # Normalisation for rebinned datasets
            norm = tools.invert_no_zero(w @ Rt)

            # Weighted rebin of the visibilities
            ssv[ind] = norm * ((vis_data[ind] * w) @ Rt)

            # Count number of samples
            ssn[ind] = m @ Rt

            # Compute factors needed for the covariance calculation
            if contains_covariance:
                iall = tuple(
                    ii if ax != "freq" else slice(None) for ii, ax in zip(ind, vax)
                )
                wall = weight_all[iall]
                nall = tools.invert_no_zero(wall @ Rt)

            # Weighted rebin of other datasets
            for name, axis in alt_dspec.items():
                aind = tuple(
                    [
                        ind[lookup[ax]] if ax in lookup else slice(None)
                        for ii, ax in enumerate(axis)
                    ]
                )

                if "freq_cov" in name:
                    salt[name][aind] = (
                        norm * nall * ((alt_data[name][aind] * w * wall) @ Rtsq)
                    )
                else:
                    salt[name][aind] = norm * ((alt_data[name][aind] * w) @ Rt)

            # Weighted rebin of the effective RA
            effective_lsd = norm * ((timestamp_lsd * w) @ Rt)
            sera[ind] = 360 * (effective_lsd - self.start)

            # Rebin the weights
            rvar = v @ Rtsq
            ssw[ind] = tools.invert_no_zero(norm**2 * rvar)

            # Correct the effective ra where weights are zero. This
            # is required to avoid discontinuities
            imask = np.nonzero(ssw[ind] == 0.0)
            sera[ind][imask] = sdata.ra[imask[-1]]

        return sdata


class RebinGradientCorrection(task.SingleTask):
    """Apply a linear gradient correction to shift RA samples to bin centres.

    Requires a sidereal day with full sidereal coverage to calculate a local
    gradient for each RA bin. The dataset value at the RA bin centre is
    interpolated based on the local gradient and difference between bin centre
    and effective bin centre.

    If the rebinned dataset has full sidereal coverage, it can be used to
    create the gradient.
    """

    def setup(self, sstream_ref: containers.SiderealStream) -> None:
        """Provide the dataset to use in the gradient calculation.

        This dataset must have complete sidereal coverage.

        Parameters
        ----------
        sstream_ref
            Reference SiderealStream
        """
        self.sstream_ref = sstream_ref

    def process(self, sstream: containers.SiderealStream) -> containers.SiderealStream:
        """Apply the gradient correction to the input dataset.

        Parameters
        ----------
        sstream
            sidereal day to apply a correction to

        Returns
        -------
        sstream
            Input sidereal day with gradient correction applied, if needed
        """
        self.sstream_ref.redistribute("freq")
        sstream.redistribute("freq")

        # Allows a normal sidereal stream to pass through this task.
        # Helpful for creating generic configs
        try:
            era = sstream.effective_ra[:].local_array
        except KeyError:
            self.log.info(
                f"Dataset of type ({type(sstream)}) does not have an effective "
                "ra dataset. No correction will be applied."
            )
            return sstream

        try:
            # If the reference dataset has an effective ra dataset, use this
            # when calculating the gradient. This could be true if the reference
            # and target datasets are the same
            ref_ra = self.sstream_ref.effective_ra[:].local_array
        except KeyError:
            # Use fixed ra, which should be regularly sampled
            ref_ra = self.sstream_ref.ra

        vis = sstream.vis[:].local_array
        weight = sstream.weight[:].local_array

        ref_vis = self.sstream_ref.vis[:].local_array
        ref_weight = self.sstream_ref.weight[:].local_array

        # Iterate over frequencies and baselines for memory
        for fi in range(vis.shape[0]):
            # Skip if entirely masked already
            if not np.any(weight[fi]):
                continue

            for vi in range(vis.shape[1]):
                # Skip if entire baseline is masked
                if not np.any(weight[fi, vi]):
                    continue

                # Depends on whether the effective ra has baseline dependence
                rra = ref_ra[fi, vi] if ref_ra.ndim > 1 else ref_ra
                # Calculate the vis gradient at the reference RA points
                ref_mask = ref_weight[fi, vi] == 0.0
                grad, ref_mask = regrid.grad_1d(
                    ref_vis[fi, vi], rra, ref_mask, period=360.0
                )

                # Apply the correction to estimate the sample value at the
                # RA bin centre. Ensure that values that are masked in the
                # original dataset do not get shifted if the reference
                # dataset has a different mask
                sel = weight[fi, vi] > 0.0
                vis[fi, vi] -= grad * sel * (era[fi, vi] - sstream.ra)
                # Keep track of the time mask being applied
                weight[fi, vi] *= (~ref_mask).astype(weight.dtype)

        # Delete the effective ra dataset since it is not needed anymore
        del sstream["effective_ra"]

        return sstream


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
        # Check that the input container is of the correct type
        if (self.stack is not None) and not isinstance(sdata, type(self.stack)):
            raise TypeError(
                f"type(sdata) (={type(sdata)}) does not match "
                f"type(stack) (={type(self.stack)})."
            )

        sdata.redistribute("ra")

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

            # Create a slice into the weight dataset that will allow it
            # to be broadcasted against the vis dataset.
            wax = sdata.weight.attrs["axis"]

            self.weight_slice = {}
            self.weight_slice["vis"] = get_slice_to_broadcast(
                wax, sdata.vis.attrs["axis"]
            )

            # Initialize any missing datasets, which will include effective_ra.
            self.additional_datasets = []
            for name, dataset in sdata.datasets.items():
                if name not in self.stack.datasets:
                    self.log.info(f"Creating {name} dataset in the sidereal stack.")
                    self.stack.add_dataset(name)
                    self.additional_datasets.append(name)

                    # Create a slice into the weight dataset that will allow it
                    # to be broadcasted against the additional dataset.
                    wslc1 = get_slice_to_broadcast(wax, dataset.attrs["axis"])

                    if "freq_cov" in name:
                        wslc2 = get_slice_to_broadcast(wax, sdata.swapped_freq_cov_axis)
                        self.weight_slice[name] = (wslc1, wslc2)
                    else:
                        self.weight_slice[name] = wslc1

            # Now that we have all datasets, redistribute over frequency.
            self.stack.redistribute("ra")

            # Initialize all datasets to zero.
            for data in self.stack.datasets.values():
                data[:] = 0

            self.lsd_list = []

            # Keep track of the sum of squared weights
            # to perform Bessel's correction at the end.
            if self.with_sample_variance:
                self.sum_coeff_sq = mpiarray.zeros(
                    self.stack.weight[:].local_shape,
                    axis=0,
                    comm=self.stack.comm,
                    dtype=np.float32,
                )

        # Accumulate
        self.log.info(f"Adding LSD {input_lsd} to stack with {self.weight} weighting.")

        self.lsd_list += input_lsd

        # Extract weight dataset
        weight = sdata.weight[:]

        # Determine if the input sidereal stream is itself a stack over days
        if "nsample" in sdata.datasets:
            # The input sidereal stream is already a stack
            # over multiple sidereal days. Use the nsample
            # dataset as the weight for the uniform case.
            # Make sure to also zero any samples whose weight
            # is zero in case other tasks did not also zero
            # the nsample dataset.
            count = sdata.nsample[:] * (weight > 0.0)
        else:
            # The input sidereal stream contains a single
            # sidereal day. Use a boolean array that
            # indicates a non-zero weight dataset as
            # the weight for the uniform case.
            count = (weight > 0.0).astype(self.stack.nsample.dtype)

        # Accumulate the total number of samples.
        self.stack.nsample[:] += count

        if self.weight == "uniform":
            coeff = count.astype(np.float32)
            self.stack.weight[:] += (coeff**2) * tools.invert_no_zero(weight)
            sum_coeff = self.stack.nsample[:]

        else:
            coeff = weight
            self.stack.weight[:] += weight
            sum_coeff = self.stack.weight[:]

        # Calculate weighted difference between the new data and the current mean.
        wslc = self.weight_slice["vis"]

        delta_before = coeff[wslc] * (sdata.vis[:] - self.stack.vis[:])
        inv_sum_coeff = tools.invert_no_zero(sum_coeff)

        # Update the mean.
        self.stack.vis[:] += delta_before * inv_sum_coeff[wslc]

        # Update any additional datasets.  Note this will include the effective_ra.
        for name in self.additional_datasets:
            wslc = self.weight_slice[name]
            if "freq_cov" in name:
                self.stack[name][:] += coeff[wslc[0]] * coeff[wslc[1]] * sdata[name][:]
            else:
                delta = coeff[wslc] * (sdata[name][:] - self.stack[name][:])
                self.stack[name][:] += delta * inv_sum_coeff[wslc]

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

        if self.weight == "uniform":
            # For uniform weighting, normalize the accumulated variances by the total
            # number of samples squared and then invert to finalize stack.weight.
            norm = self.stack.nsample[:].astype(np.float32)
            self.stack.weight[:] = tools.invert_no_zero(self.stack.weight[:]) * norm**2

        else:
            # For inverse variance weighting the stack.weight dataset is finalized,
            # no additional normalization is required.
            norm = self.stack.weight[:]

        # Apply the inverse squared norm factor to any dataset that is a
        # freq-freq covariance
        for name in self.additional_datasets:
            if "freq_cov" in name:
                wslc = self.weight_slice[name]
                self.stack[name][:] *= tools.invert_no_zero(
                    norm[wslc[0]] * norm[wslc[1]]
                )

        # We need to normalize the sample variance by the sum of coefficients.
        # Can be found in the stack.nsample dataset for uniform case
        # or the stack.weight dataset for inverse variance case.
        if self.with_sample_variance:

            # Perform Bessel's correction.  In the case of
            # uniform  weighting, norm will be equal to nsample - 1.
            norm = norm - self.sum_coeff_sq * tools.invert_no_zero(norm)

            # Normalize the sample variance.
            wslc = (None,) + self.weight_slice["vis"]
            self.stack.sample_variance[:] *= np.where(
                self.stack.nsample[:] > 1, tools.invert_no_zero(norm), 0.0
            )[wslc]

        # Now redistribute back over frequency
        self.stack.redistribute("freq")

        # For samples where there is no data, the effective ra should
        # be the same as the grid ra
        if "effective_ra" in self.stack.datasets:
            weight = self.stack.weight[:].local_array
            era = self.stack.effective_ra[:].local_array

            # Broadcast the RA array to match the shape of a single frequency,
            # allowing us to select grid ra values with a 2D mask
            grid_ra = np.broadcast_to(self.stack.ra, (*era.shape[1:],))

            for fi in range(era.shape[0]):
                mask = weight[fi] == 0.0
                era[fi][mask] = grid_ra[mask]

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
        # Check that the input container is of the correct type
        if (self.stack is not None) and not isinstance(sdata, type(self.stack)):
            raise TypeError(
                f"type(sdata) (={type(sdata)}) does not match "
                f"type(stack) (={type(self.stack)})."
            )

        sdata.redistribute("freq")

        if self.stack is None:
            self.log.info("Starting new stack.")

            self.stack = containers.empty_like(sdata)
            self.stack.redistribute("freq")

            # Initialise all datasets to zero
            for ds in self.stack.datasets.values():
                ds[:] = 0

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

        if "effective_ra" in self.stack.datasets:
            # Track the effective ra bin centres. We're using the averaged
            # weights here, so this generally isn't optimal
            delta = Ni_d * (sdata.effective_ra[:] - self.stack.effective_ra[:])
            # Update the mean effective ra using the mean of the normalized weights
            sum_weight = tools.invert_no_zero(self.stack.weight[:]) * self.Ni_s**2
            self.stack.effective_ra[:] += delta * tools.invert_no_zero(
                sum_weight.mean(axis=1)
            )

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

        return self.stack


def get_slice_to_broadcast(weight_axis, dataset_axis):
    """Generate a slice that will broadcast the weights against some other dataset.

    Parameters
    ----------
    weight_axis : list of str
        Names of the axes in the weights.
    dataset_axis : list of str
        Names of the axes in the dataset.

    Returns
    -------
    slc : list containing either slice(None) or None
        Slice that when applied to the weights will make them broadcastable
        against the other dataset.
    """
    # The number of dimensions in the weights must be equal or less than
    # the number of dimensions in the dataset.
    assert len(weight_axis) <= len(dataset_axis)

    # The weights cannot have any additional axes that are not present in the dataset.
    assert all(wax in dataset_axis for wax in weight_axis)

    # The axes that are shared between the weights and the other dataset
    # must be in the same order.
    common_axis = [ax for ax in dataset_axis if ax in weight_axis]
    assert all(wax == dax for wax, dax in zip(weight_axis, common_axis))

    # If all checks passed, then return the slice to broadcast.
    return tuple([slice(None) if ax in weight_axis else None for ax in dataset_axis])


def _ensure_list(x):
    if hasattr(x, "__iter__"):
        y = list(x)
    else:
        y = [x]

    return y

"""Take timestream data and regridding it into sidereal days which can be
stacked.

Tasks
=====

.. autosummary::
    :toctree:

    SiderealGrouper
    SiderealRegridder
    SiderealStacker

Usage
=====

Generally you would want to use these tasks together. Sending time stream data
into  :class:`SiderealGrouper`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`SiderealStacker` if you want to combine the different days.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


import numpy as np

from caput import config, mpiutil, mpiarray, tod

from .transform import Regridder
from ..core import task, containers, io
from ..util import tools


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
        super(SiderealGrouper, self).__init__()

        self._timestream_list = []
        self._current_lsd = None

    def setup(self, manager):
        """Set the local observers position.

        Parameters
        ----------
        observer : :class:`~caput.time.Observer`
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
        else:
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
    """

    def setup(self, manager):
        """Set the local observers position.

        Parameters
        ----------
        observer : :class:`~caput.time.Observer`
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

        # Convert data timestamps into LSDs
        timestamp_lsd = self.observer.unix_to_lsd(data.time)

        # Fetch which LSD this is to set bounds
        self.start = data.attrs["lsd"]
        self.end = self.start + 1

        # Get view of data
        weight = data.weight[:].view(np.ndarray)
        vis_data = data.vis[:].view(np.ndarray)

        # perform regridding
        new_grid, sts, ni = self._regrid(vis_data, weight, timestamp_lsd)

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
        sdata.attrs["tag"] = "lsd_%i" % self.start

        return sdata


class SiderealStacker(task.SingleTask):
    """Take in a set of sidereal days, and stack them up.

    Also computes the variance over sideral days using an
    algorithm that updates the sum of square differences from
    the current mean, which is less prone to numerical issues.

    Attributes
    ----------
    weight: str (default: "inverse_variance")
        The weighting to use in the stack.
        Either `uniform` or `inverse_variance`.
    """

    stack = None

    weight = config.enum(["uniform", "inverse_variance"], default="inverse_variance")

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
            if "sample_variance" not in self.stack.datasets:
                self.stack.add_dataset("sample_variance")

            if "nsample" not in self.stack.datasets:
                self.stack.add_dataset("nsample")

            self.stack.redistribute("freq")

            # Initialize all datasets to zero.
            for data in self.stack.datasets.values():
                data[:] = 0

            self.lsd_list = []

            # Keep track of the sum of squared weights
            # to perform Bessel's correction at the end.
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
            self.stack.weight[:] += (coeff ** 2) * tools.invert_no_zero(sdata.weight[:])
        else:
            coeff = sdata.weight[:]
            # We are using inverse variance weights.  In this case,
            # we accumulate the inverse variances in the stack.weight
            # dataset.  Do that directly to avoid an unneccessary
            # division in the more general expression above.
            self.stack.weight[:] += sdata.weight[:]

        # Accumulate the total number of samples and the sum of squared coefficients.
        self.stack.nsample[:] += count
        self.sum_coeff_sq += coeff ** 2

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
        self.stack.attrs["tag"] = "stack"
        self.stack.attrs["lsd"] = np.array(self.lsd_list)

        nsample = self.stack.nsample[:]

        # We need to normalize the sample variance by the sum of coefficients.
        # Can be found in the stack.nsample dataset for uniform case
        # or the stack.weight dataset for inverse variance case.
        if self.weight == "uniform":
            norm = nsample.astype(np.float32)
            # Normalize the accumulated variances by the total number of samples
            # and then invert to finalize stack.weight.
            self.stack.weight[:] = (
                tools.invert_no_zero(self.stack.weight[:]) * norm ** 2
            )
        else:
            norm = self.stack.weight[:]

        # Perform Bessel's correction.  In the case of
        # uniform  weighting, norm will be equal to nsample - 1.
        norm = norm - self.sum_coeff_sq * tools.invert_no_zero(norm)

        # Normalize the sample variance.
        self.stack.sample_variance[:] *= np.where(
            nsample > 1, tools.invert_no_zero(norm), 0.0
        )[np.newaxis, :]

        return self.stack


def _ensure_list(x):

    if hasattr(x, "__iter__"):
        y = [xx for xx in x]
    else:
        y = [x]

    return y

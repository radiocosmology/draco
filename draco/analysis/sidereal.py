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


import numpy as np

from caput import config, mpiutil, mpiarray, tod

from .transform import Regridder
from ..core import task, containers, io


class SiderealGrouper(task.SingleTask):
    """Group individual timestreams together into whole Sidereal days.

    Attributes
    ----------
    padding : float
        Extra amount of a sidereal day to pad each timestream by. Useful for
        getting rid of interpolation artifacts.
    """

    padding = config.Property(proptype=float, default=0.)

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

        # Get the start and end LSDs of the file
        lsd_start = int(self.observer.unix_to_lsd(tstream.time[0] - self.padding))
        lsd_end = int(self.observer.unix_to_lsd(tstream.time[-1] + self.padding))

        # If current_lsd is None then this is the first time we've run
        if self._current_lsd is None:
            self._current_lsd = lsd_start

        # If this file started during the current lsd add it onto the list
        if self._current_lsd  == lsd_start:
            self._timestream_list.append(tstream)

        self.log.info("Adding file into group for LSD:%i", lsd_start)

        # If this file ends during a later LSD then we need to process the
        # current list and restart the system
        if self._current_lsd < lsd_end:
            self.log.info("Concatenating files for LSD:%i", lsd_start)

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
        tstream_all = self._process_current_lsd()

        return tstream_all

    def _process_current_lsd(self):
        # Combine the current set of files into a timestream

        lsd = self._current_lsd

        # Calculate the length of data in this current LSD
        start = self.observer.unix_to_lsd(self._timestream_list[0].time[0])
        end = self.observer.unix_to_lsd(self._timestream_list[-1].time[-1])
        day_length = min(end, lsd + 1) - max(start, lsd)

        # If the amount of data for this day is too small, then just skip
        if day_length < 0.1:
            return None

        self.log.info("Constructing LSD:%i [%i files]",
                      lsd, len(self._timestream_list))

        # Construct the combined timestream
        ts = tod.concatenate(self._timestream_list)

        # Add attributes for the LSD and a tag for labelling saved files
        ts.attrs['tag'] = ('lsd_%i' % lsd)
        ts.attrs['lsd'] = lsd

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

        self.log.info("Regridding LSD:%i", data.attrs['lsd'])

        # Redistribute if needed too
        data.redistribute('freq')

        # Convert data timestamps into LSDs
        timestamp_lsd = self.observer.unix_to_lsd(data.time)

        # Fetch which LSD this is to set bounds
        self.start = data.attrs['lsd']
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
        sdata.redistribute('freq')
        sdata.vis[:] = sts
        sdata.weight[:] = ni
        sdata.attrs['lsd'] = self.start
        sdata.attrs['tag'] = 'lsd_%i' % self.start

        return sdata


class SiderealStacker(task.SingleTask):
    """Take in a set of sidereal days, and stack them up.

    This will apply relative calibration.
    """

    stack = None
    lsd_list = None

    def process(self, sdata):
        """Stack up sidereal days.

        Parameters
        ----------
        sdata : containers.SiderealStream
            Individual sidereal day to stack up.
        """

        sdata.redistribute('freq')

        # Get the LSD label out of the data (resort to using a CSD if it's
        # present). If there's no label just use a place holder and stack
        # anyway.
        if 'lsd' in sdata.attrs:
            input_lsd = sdata.attrs['lsd']
        elif 'csd' in sdata.attrs:
            input_lsd = sdata.attrs['csd']
        else:
            input_lsd = -1

        input_lsd = _ensure_list(input_lsd)

        if self.stack is None:

            self.stack = containers.empty_like(sdata)
            self.stack.redistribute('freq')

            self.stack.vis[:] = sdata.vis[:] * sdata.weight[:]
            self.stack.weight[:] = sdata.weight[:]

            self.lsd_list = input_lsd

            self.log.info("Starting stack with LSD:%i", sdata.attrs['lsd'])

            return

        self.log.info("Adding LSD:%i to stack", sdata.attrs['lsd'])

        # note: Eventually we should fix up gains

        # Combine stacks with inverse `noise' weighting
        self.stack.vis[:] += (sdata.vis[:] * sdata.weight[:])
        self.stack.weight[:] += sdata.weight[:]

        self.lsd_list += input_lsd

    def process_finish(self):
        """Construct and emit sidereal stack.

        Returns
        -------
        stack : containers.SiderealStream
            Stack of sidereal days.
        """

        self.stack.attrs['tag'] = 'stack'
        self.stack.attrs['lsd'] = np.array(self.lsd_list)

        self.stack.vis[:] = np.where(self.stack.weight[:] == 0,
                                     0.0,
                                     self.stack.vis[:] / self.stack.weight[:])

        return self.stack


def _ensure_list(x):

    if hasattr(x, '__iter__'):
        y = [xx for xx in x]
    else:
        y = [x]

    return y

"""
============================================================
Tasks for sidereal regridding (:mod:`~ch_pipeline.sidereal`)
============================================================

.. currentmodule:: ch_pipeline.sidereal

Tasks for taking the timestream data and regridding it into sidereal days
which can be stacked.

Tasks
=====

.. autosummary::
    :toctree: generated/

    LoadTimeStreamSidereal
    SiderealRegridder
    SiderealStacker

Usage
=====

Generally you would want to use these tasks together. Starting with a
:class:`LoadTimeStreamSidereal`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`SiderealStacker` if you want to combine the different days.
"""


import numpy as np

from caput import pipeline, config
from caput import mpiutil, mpiarray
from ch_util import andata, ephemeris

from . import task
from . import containers
from . import regrid


def get_times(acq_files):
    """Extract the start and end times of a list of acquisition files.

    Parameters
    ----------
    acq_files : list
        List of filenames.

    Returns
    -------
    times : np.ndarray[nfiles, 2]
        Start and end times.
    """
    if isinstance(acq_files, list):
        return np.array([get_times(acq_file) for acq_file in acq_files])
    elif isinstance(acq_files, basestring):
        # Load in file (but ignore all datasets)
        ad_empty = andata.AnData.from_acq_h5(acq_files, datasets=())
        start = ad_empty.timestamp[0]
        end = ad_empty.timestamp[-1]
        return start, end
    else:
        raise Exception('Input %s, not understood' % repr(acq_files))


def _days_in_csd(day, se_csd, extra=0.005):
    # Find which days are in each CSD
    stest = se_csd[:, 1] > day - extra
    etest = se_csd[:, 0] < day + 1 - extra

    return np.where(np.logical_and(stest, etest))[0]


class LoadTimeStreamSidereal(task.SingleTask):
    """Load data in sidereal days.

    This task takes an input list of data, and loads in a sidereal day at a
    time, and passes it on.

    Attributes
    ----------
    padding : float
        Extra amount of a sidereal day to pad each timestream by. Useful for
        getting rid of interpolation artifacts.
    """

    padding = config.Property(proptype=float, default=0.005)

    freq_start = config.Property(proptype=int, default=None)
    freq_end = config.Property(proptype=int, default=None)

    def setup(self, files):
        """Divide the list of files up into sidereal days.

        Parameters
        ----------
        files : list
            List of files to load.
        """

        self.files = files

        filemap = None
        if mpiutil.rank0:

            se_times = get_times(self.files)
            se_csd = ephemeris.csd(se_times)
            days = np.unique(np.floor(se_csd).astype(np.int))

            # Construct list of files in each day
            filemap = [ (day, _days_in_csd(day, se_csd, extra=self.padding)) for day in days ]

            # Filter our days with only a few files in them.
            filemap = [ (day, dmap) for day, dmap in filemap if dmap.size > 1 ]
            filemap.sort()

        self.filemap = mpiutil.world.bcast(filemap, root=0)

        # Set up frequency selection
        if self.freq_start is not None:
            self.freq_sel = np.arange(self.freq_start, self.freq_end)
        else:
            self.freq_sel = None

    def process(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : andata.CorrData
            The timestream of each sidereal day.
        """

        if len(self.filemap) == 0:
            raise pipeline.PipelineStopIteration

        csd, fmap = self.filemap.pop(0)
        dfiles = [ self.files[fi] for fi in fmap ]

        if mpiutil.rank0:
            print "Starting read of CSD:%i [%i files]" % (csd, len(fmap))

        ts = andata.CorrData.from_acq_h5(sorted(dfiles), distributed=True, freq_sel=self.freq_sel)

        # Add attributes for the CSD and a tag for labelling saved files
        ts.attrs['tag'] = ('csd_%i' % csd)
        ts.attrs['csd'] = csd

        # Add a weight dataset if needed
        if 'vis_weight' not in ts.datasets:
            weight_dset = ts.create_dataset('vis_weight', shape=ts.vis.shape, dtype=np.uint8,
                                            distributed=True, distributed_axis=0)
            weight_dset.attrs['axis'] = ts.vis.attrs['axis']
            weight_dset[:] = 128

        return ts


class SiderealGrouper(task.SingleTask):
    """Group individual timestreams together into whole Sidereal days.

    Attributes
    ----------
    padding : float
        Extra amount of a sidereal day to pad each timestream by. Useful for
        getting rid of interpolation artifacts.
    """

    padding = config.Property(proptype=float, default=0.005)

    def __init__(self):
        self._timestream_list = []
        self._current_csd = None

    def process(self, tstream):
        """Load in each sidereal day.

        Parameters
        ----------
        tstream : andata.CorrData
            Timestream to group together.

        Returns
        -------
        ts : andata.CorrData or None
            Returns the timestream of each sidereal day when we have received
            the last file, otherwise returns :obj:`None`.
        """

        # Get the start and end CSDs of the file
        csd_start = int(ephemeris.csd(tstream.time[0]))
        csd_end = int(ephemeris.csd(tstream.time[-1]))

        # If current_csd is None then this is the first time we've run
        if self._current_csd is None:
            self._current_csd = csd_start

        # If this file started during the current CSD add it onto the list
        if self._current_csd == csd_start:
            self._timestream_list.append(tstream)

        if tstream.vis.comm.rank == 0:
            print "Adding file into group for CSD:%i" % csd_start

        # If this file ends during a later CSD then we need to process the
        # current list and restart the system
        if self._current_csd < csd_end:

            if tstream.vis.comm.rank == 0:
                print "Concatenating files for CSD:%i" % csd_start

            # Combine timestreams into a single container for the whole day this
            # could get returned as None if there wasn't enough data
            tstream_all = self._process_current_csd()

            # Reset list and current CSD for the new file
            self._timestream_list = [tstream]
            self._current_csd = csd_end

            return tstream_all
        else:
            return None

    def process_finish(self):
        """Return the final sidereal day.

        Returns
        -------
        ts : andata.CorrData or None
            Returns the timestream of the final sidereal day if it's long
            enough, otherwise returns :obj:`None`.
        """

        # If we are here there is no more data coming, we just need to process any remaining data
        tstream_all = self._process_current_csd()

        return tstream_all

    def _process_current_csd(self):
        # Combine the current set of files into a timestream

        csd = self._current_csd

        # Calculate the length of data in this current CSD
        start = ephemeris.csd(self._timestream_list[0].time[0])
        end = ephemeris.csd(self._timestream_list[-1].time[-1])
        day_length = min(end, csd + 1) - max(start, csd)

        # If the amount of data for this day is too small, then just skip
        if day_length < 0.1:
            return None

        if self._timestream_list[0].vis.comm.rank == 0:
            print "Constructing CSD:%i [%i files]" % (csd, len(self._timestream_list))

        # Construct the combined timestream
        ts = andata.concatenate(self._timestream_list)

        # Add attributes for the CSD and a tag for labelling saved files
        ts.attrs['tag'] = ('csd_%i' % csd)
        ts.attrs['csd'] = csd

        return ts


class SiderealRegridder(task.SingleTask):
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
    """

    samples = config.Property(proptype=int, default=1024)
    lanczos_width = config.Property(proptype=int, default=5)

    def process(self, data):
        """Regrid the sidereal day.

        Parameters
        ----------
        data : andata.CorrData
            Timestream data for the day (must have a `csd` attribute).

        Returns
        -------
        sdata : containers.SiderealStream
            The regularly gridded sidereal timestream.
        """

        if mpiutil.rank0:
            print "Regridding CSD:%i" % data.attrs['csd']

        # Redistribute if needed too
        data.redistribute('freq')

        # Convert data timestamps into CSDs
        timestamp_csd = ephemeris.csd(data.time)

        # Fetch which CSD this is
        csd = data.attrs['csd']

        # Create a regular grid in CSD, padded at either end to supress interpolation issues
        pad = 5 * self.lanczos_width
        csd_grid = csd + np.arange(-pad, self.samples + pad, dtype=np.float64) / self.samples

        # Construct regridding matrix
        lzf = regrid.lanczos_forward_matrix(csd_grid, timestamp_csd, self.lanczos_width).T.copy()

        # Mask data
        imask = data.weight[:].view(np.ndarray)
        vis_data = data.vis[:].view(np.ndarray)

        # Reshape data
        vr = vis_data.reshape(-1, vis_data.shape[-1])
        nr = imask.reshape(-1, vis_data.shape[-1])

        # Construct a signal 'covariance'
        Si = np.ones_like(csd_grid) * 1e-8

        # Calculate the interpolated data and a noise weight at the points in the padded grid
        sts, ni = regrid.band_wiener(lzf, nr, Si, vr, 2*self.lanczos_width-1)

        # Throw away the padded ends
        sts = sts[:, pad:-pad].copy()
        ni = ni[:, pad:-pad].copy()

        # Reshape to the correct shape
        sts = sts.reshape(vis_data.shape[:-1] + (self.samples,))
        ni = ni.reshape(vis_data.shape[:-1] + (self.samples,))

        # Wrap to produce MPIArray
        sts = mpiarray.MPIArray.wrap(sts, axis=data.vis.distributed_axis)
        ni = mpiarray.MPIArray.wrap(ni,  axis=data.vis.distributed_axis)

        # FYI this whole process creates an extra copy of the sidereal stack.
        # This could probably be optimised out with a little work.
        sdata = containers.SiderealStream(axes_from=data, ra=self.samples)
        sdata.redistribute('freq')
        sdata.vis[:] = sts
        sdata.weight[:] = ni
        sdata.attrs['csd'] = csd
        sdata.attrs['tag'] = 'csd_%i' % csd

        return sdata


class SiderealStacker(task.SingleTask):
    """Take in a set of sidereal days, and stack them up.

    This will apply relative calibration.
    """

    stack = None

    def process(self, sdata):
        """Stack up sidereal days.

        Parameters
        ----------
        sdata : containers.SiderealStream
            Individual sidereal day to stack up.
        """

        sdata.redistribute('freq')

        if self.stack is None:

            self.stack = containers.SiderealStream(axes_from=sdata)
            self.stack.redistribute('freq')

            self.stack.vis[:] = (sdata.vis[:] * sdata.weight[:])
            self.stack.weight[:] = sdata.weight[:]

            if mpiutil.rank0:
                print "Starting stack with CSD:%i" % sdata.attrs['csd']

            return

        if mpiutil.rank0:
            print "Adding CSD:%i to stack" % sdata.attrs['csd']

        # note: Eventually we should fix up gains

        # Combine stacks with inverse `noise' weighting
        self.stack.vis[:] += (sdata.vis[:] * sdata.weight[:])
        self.stack.weight[:] += sdata.weight[:]

    def process_finish(self):
        """Construct and emit sidereal stack.

        Returns
        -------
        stack : containers.SiderealStream
            Stack of sidereal days.
        """

        self.stack.attrs['tag'] = 'stack'

        self.stack.vis[:] = np.where(self.stack.weight[:] == 0,
                                     0.0,
                                     self.stack.vis[:] / self.stack.weight[:])

        return self.stack

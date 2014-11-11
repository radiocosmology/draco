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
from caput import mpiutil, mpidataset
from ch_util import andata, ephemeris

from . import dataspec
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


class LoadTimeStreamSidereal(pipeline.TaskBase):
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

    def setup(self, dspec):
        """Divide the list of files up into sidereal days.

        Parameters
        ----------
        dspec : dict
            Dataspec dictionary.
        """

        self.files = dataspec.files_from_spec(dspec)

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

    def next(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : containers.TimeStream
            The timestream of each sidereal day.
        """

        if len(self.filemap) == 0:
            raise pipeline.PipelineStopIteration

        csd, fmap = self.filemap.pop(0)
        dfiles = [ self.files[fi] for fi in fmap ]

        if mpiutil.rank0:
            print "Starting read of CSD:%i [%i files]" % (csd, len(fmap))

        ts = containers.TimeStream.from_acq_files(sorted(dfiles))  # Ensure file list if sorted
        ts.attrs['tag'] = ('csd_%i' % csd)
        ts.attrs['csd'] = csd

        return ts


class SiderealRegridder(pipeline.TaskBase):
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

    def next(self, data):
        """Regrid the sidereal day.

        Parameters
        ----------
        data : containers.TimeStream
            Timestream data for the day (must have a `csd` attribute).

        Returns
        -------
        sdata : containers.SiderealStream
            The regularly gridded sidereal timestream.
        """

        if mpiutil.rank0:
            print "Regridding CSD:%i" % data.attrs['csd']

        # Redistribute if needed too
        data.redistribute(axis=1)

        # Convert data timestamps into CSDs
        timestamp_csd = ephemeris.csd(data.timestamp)

        # For regular grid in CSD
        csd = data.attrs['csd']
        csd_grid = np.linspace(csd, csd + 1.0, self.samples, endpoint=False)

        # Construct regridding matrix
        lzf = regrid.lanczos_forward_matrix(csd_grid, timestamp_csd, self.lanczos_width).T.copy()

        # Mask data
        imask = (1.0 - data.mask).view(np.ndarray)
        vis_data = data.vis.view(np.ndarray)

        # Reshape data
        vr = vis_data.reshape(-1, vis_data.shape[-1])
        nr = imask.reshape(-1, vis_data.shape[-1])

        # Construct a signal 'covariance'
        Si = np.ones_like(csd_grid) * 1e-8

        sts, ni = regrid.band_wiener(lzf, nr, Si, vr, 2*self.lanczos_width-1)
        sts = sts.reshape(vis_data.shape[:-1] + (self.samples,))
        ni = ni.reshape(vis_data.shape[:-1] + (self.samples,))
        # Construct inverse noise weighting - assuming imask is the inverse
        # noise matrix, this calculates the diagonal of the inverse covariance
        # matrix
        ni = np.dot(imask.reshape(-1, vis_data.shape[-1]), (lzf * lzf))

        # Wrap to produce MPIArray
        sts = mpidataset.MPIArray.wrap(sts, axis=data.vis.axis)
        ni  = mpidataset.MPIArray.wrap(ni,  axis=data.vis.axis)

        sdata = containers.SiderealStream(self.samples, 1, 1)
        sdata._distributed['vis'] = sts
        sdata._distributed['weight'] = ni
        sdata.attrs['csd'] = csd
        sdata.attrs['tag'] = 'csd_%i' % csd

        return sdata


class SiderealStacker(pipeline.TaskBase):
    """Take in a set of sidereal days, and stack them up.

    This will apply relative calibration.
    """

    vis_stack = None
    noise_stack = None
    count = 0

    def next(self, sdata):
        """Stack up sidereal days.

        Parameters
        ----------
        sdata : containers.SiderealStream
            Individual sidereal day to stack up.
        """
        if self.count == 0:
            self.vis_stack = mpidataset.MPIArray.wrap(sdata.vis * sdata.weight, axis=sdata.vis.axis)
            self.weight_stack = sdata.weight
            self.count = 1

            if mpiutil.rank0:
                print "Starting stack with CSD:%i" % sdata.attrs['csd']

            return

        if mpiutil.rank0:
            print "Adding CSD:%i to stack" % sdata.attrs['csd']

        # Eventually we should fix up gains

        # Ensure we are distributed over the same axis
        axis = self.vis_stack.axis
        sdata.redistribute(axis=axis)

        # Combine stacks with inverse `noise' weighting
        self.vis_stack += sdata.vis * sdata.weight
        self.weight_stack += sdata.weight

        self.count += 1

    def finish(self):
        """Construct and emit sidereal stack.

        Returns
        -------
        stack : containers.SiderealStream
            Stack of sidereal days.
        """

        sstack = containers.SiderealStream(self.vis_stack.global_shape[-1], 1, 1)

        vis = np.where(self.weight_stack == 0, np.zeros_like(self.vis_stack), self.vis_stack / self.weight_stack)
        vis = mpidataset.MPIArray.wrap(vis, self.vis_stack.axis)

        sstack._distributed['vis'] = vis
        sstack._distributed['weight'] = self.weight_stack
        sstack.attrs['tag'] = 'stack'

        return sstack

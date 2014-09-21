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

import glob
import numpy as np

from caput import pipeline, config
from caput import mpiutil, mpidataset
from ch_util import andata, ephemeris

import containers


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
    elif isinstance(acq_files, str):
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
    files : glob pattern
        List of filenames as a glob pattern.
    padding : float
        Extra amount of a sidereal day to pad each timestream by. Useful for
        getting rid of interpolation artifacts.
    """

    filepat = config.Property(proptype=str)
    padding = config.Property(proptype=float, default=0.005)

    def setup(self):
        """Divide the list of files up into sidereal days.
        """

        self.files = glob.glob(self.filepat)

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




def lanczos_kernel(x, a):
    """Lanczos interpolation kernel.

    Parameters
    ----------
    x : array_like
        Point separation.
    a : integer
        Lanczos kernel width.

    Returns
    -------
    kernel : np.ndarray
    """

    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x/a), np.zeros_like(x))


def lanczos_forward_matrix(x, y, a=5):
    """Regrid data using a maximum likelihood inverse Lanczos.

    Parameters
    ----------
    x : np.ndarray[m]
        Points to regrid data onto. Must be regularly spaced.
    y : np.ndarray[n]
        Points we have data at. Irregular spacing.
    a : integer, optional
        Lanczos width parameter.
    cond : float
        Relative condition number for pseudo-inverse.

    Returns
    -------
    matrix : np.ndarray[m, n]
        Lanczos regridding matrix. Apply to data with `np.dot(matrix, data)`.
    """
    dx = x[1] - x[0]

    sep = (x[np.newaxis, :] - y[:, np.newaxis]) / dx

    lz_forward = lanczos_kernel(sep, a)

    return lz_forward


def lanczos_inverse_matrix(x, y, a=5, cond=1e-1):
    """Regrid data using a maximum likelihood inverse Lanczos.

    Parameters
    ----------
    x : np.ndarray[m]
        Points to regrid data onto. Must be regularly spaced.
    y : np.ndarray[n]
        Points we have data at. Irregular spacing.
    a : integer, optional
        Lanczos width parameter.
    cond : float
        Relative condition number for pseudo-inverse.

    Returns
    -------
    matrix : np.ndarray[m, n]
        Lanczos regridding matrix. Apply to data with `np.dot(matrix, data)`.
    """

    import scipy.linalg as la

    lz_forward = lanczos_forward_matrix(x, y, a)
    lz_inverse = la.pinv(lz_forward, rcond=cond)

    return lz_inverse


class SiderealRegridder(pipeline.TaskBase):
    """Take a sidereal days worth of data, and put onto a regular grid.

    Uses a maximum-likelihood inverse of a Lanczos interpolation to do the
    regridding. This gives a reasonably local regridding, that is pretty well
    behaved in m-space.

    Attributes
    ----------
    samples : int
        Number of samples across the sidereal day.
    """

    samples = config.Property(proptype=int, default=1024)

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
        lzi = lanczos_inverse_matrix(csd_grid, timestamp_csd, 5)

        # Mask data
        vis_data = (data.vis * (1 - data.mask)).view(np.ndarray)

        # Regrid data
        sts = np.dot(vis_data.reshape(-1, vis_data.shape[-1]), lzi.T)
        sts = sts.reshape(vis_data.shape[:-1] + (self.samples,))

        # Wrap to produce MPIArray
        sts = mpidataset.MPIArray.wrap(sts, axis=data.vis.axis)

        sdata = containers.SiderealStream(self.samples, 1, 1)
        sdata._distributed['vis'] = sts
        sdata.attrs['csd'] = csd
        sdata.attrs['tag'] = 'csd_%i' % csd

        return sdata


class SiderealStacker(pipeline.TaskBase):
    """Take in a set of sidereal days, and stack them up.

    This will apply relative calibration.
    """

    vis_stack = None
    mask_stack = None
    count = 0

    def next(self, sdata):
        """Stack up sidereal days.

        Parameters
        ----------
        sdata : containers.SiderealStream
            Individual sidereal day to stack up.
        """
        if self.count == 0:
            self.vis_stack = sdata.vis
            self.count = 1

            print "Starting stack with CSD:%i" % sdata.attrs['csd']

            return

        print "Adding CSD:%i to stack" % sdata.attrs['csd']

        # Eventually, we should fix up gains, and combine masks

        # Ensure we are distributed over the same axis
        axis = self.vis_stack.axis
        sdata.redistribute(axis=axis)

        self.vis_stack += sdata.vis
        self.count += 1

    def finish(self):
        """Construct and emit sidereal stack.

        Returns
        -------
        stack : containers.SiderealStream
            Stack of sidereal days.
        """

        self.vis_stack /= self.count

        sstack = containers.SiderealStream(self.vis_stack.global_shape[-1], 1, 1)
        sstack._distributed['vis'] = self.vis_stack
        sstack.attrs['tag'] = 'stack'

        return sstack

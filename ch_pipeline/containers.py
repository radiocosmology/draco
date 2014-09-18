"""Containers for parallel datasets of various types.
"""

import numpy as np

from caput import mpidataset
from ch_util import andata
from mpi4py import MPI


class SiderealStream(mpidataset.MPIDataset):
    """Parallel container for holding sidereal timestream data.

    Parameters
    ----------
    nra : integer
        Number of samples in RA.
    nfreq : integer
        Number of frequencies.
    ncorr : integer
        Number of correlation products.
    comm : MPI.Comm
        MPI communicator to distribute over.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    ra : np.ndarray
        RA samples.
    """
    _common = { 'ra': None }

    _distributed = { 'vis': None }

    @property
    def vis(self):
        return self['vis']

    @property
    def ra(self):
        return self['ra']

    def __init__(self, nra, nfreq, ncorr, comm=None):

        mpidataset.MPIDataset.__init__(self, comm)

        self.common['ra'] = np.linspace(0.0, 360.0, nra, endpoint=False)
        self.distributed['data'] = mpidataset.MPIArray((nfreq, ncorr, nra), dtype=np.complex128, comm=comm)


class TimeStream(mpidataset.MPIDataset):
    """Parallel container for holding timestream data.

    Parameters
    ----------
    times : np.ndarray
        Array of UNIX times in this dataset.
    nfreq : integer
        Number of frequencies.
    ncorr : integer
        Number of correlation products.
    comm : MPI.Comm
        MPI communicator to distribute over.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    timestamp : np.ndarray
        Timestamps.

    Methods
    -------
    from_acq_files
    """

    _common = { 'timestamp': None }

    _distributed = { 'vis': None }

    @property
    def vis(self):
        return self['vis']

    @property
    def timestamp(self):
        return self['timestamp']

    def __init__(self, times, nfreq, ncorr, comm=None):

        mpidataset.MPIDataset.__init__(self, comm)

        self.common['timestamp'] = times
        self.distributed['vis'] = mpidataset.MPIArray((nfreq, ncorr, times.shape[0]), dtype=np.complex128, comm=comm)

    @classmethod
    def from_acq_files(cls, files, comm=None):
        """Load a set of acquisition files into a parallel timestream object.

        Parameters
        ----------
        files : list
            List of filenames to load. Presumed to be contiguous, and in-order.
        comm : MPI.Comm, optional
            MPI communicator to distribute over. Defaults to `MPI.COMM_WORLD`

        Returns
        -------
        ts : TimeStream
            Parallel timestream. Initially distributed across frequency.
        """

        if comm is None:
            comm = MPI.COMM_WORLD

        # Extract data shape from first file, and distribute to all ranks
        vis_shape = None
        if comm.rank == 0:
            # Open first file and check shape
            d0 = andata.CorrData.from_acq_h5(files[0])
            vis_shape = d0.vis.shape
            del d0

        vis_shape = comm.bcast(vis_shape, root=0)

        # Unpack to get the individual lengths
        nfreq, nprod, ntime = vis_shape
        nfile = len(files)

        # Create distribute dataset
        dset = mpidataset.MPIArray((nfile, nfreq, nprod, ntime),
                                   dtype=np.complex128, comm=comm)

        # Timestamps
        timestamps = []

        for li, gi in dset.enumerate(0):

            lfile = files[gi]

            print "Rank %i reading %s" % (comm.rank, lfile)
            # Load file
            df = andata.CorrData.from_acq_h5(lfile)

            # Check shape is correct
            if df.vis.shape != vis_shape:
                raise Exception("Data from %s is not the right shape" % lfile)

            # Copy into local dataset
            dset[li] = df.vis[:]

            # Get timestamps
            timestamps.append((gi, df.timestamp.copy()))

            del df

        ## Merge timestamps
        tslist = comm.allgather(timestamps)
        tsflat = [ts for proclist in tslist for ts in proclist]  # Flatten list

        # Create list of order timestamp arrays only
        tsflat = zip(*sorted(tsflat))[1]
        timestamp_array = np.concatenate(tsflat)

        # Redistribute by frequency
        dset2 = dset.redistribute(1)
        del dset
        dset = dset2

        # Merge file and time axes
        dset = dset.transpose((1, 2, 0, 3))
        dset = dset.reshape((None, nprod, nfile * ntime))

        # Create TimeStream class (set zeros sizes to stop allocation)
        ts = cls(timestamp_array, 1, 1, comm=comm)

        # Replace vis dataset with real data
        ts.distributed['vis'] = dset

        return ts


class MaskedTimeStream(TimeStream):
    """Parallel container for holding *masked* timestream data.

    Parameters
    ----------
    times : np.ndarray
        Array of UNIX times in this dataset.
    nfreq : integer
        Number of frequencies.
    ncorr : integer
        Number of correlation products.
    comm : MPI.Comm
        MPI communicator to distribute over.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    mask : mpidataset.MPIArray
        Boolean mask of bad values.
    timestamp : np.ndarray
        Timestamps.

    Methods
    -------
    from_timestream_and_mask
    """

    _distributed = { 'vis': None,
                     'mask': None }

    @property
    def mask(self):
        return self['mask']

    @property
    def timestamp(self):
        return self['timestamp']

    def __init__(self, times, nfreq, ncorr, comm=None):

        TimeStream.__init__(self, times, nfreq, ncorr, comm)
        self.distributed['mask'] = mpidataset.MPIArray((nfreq, ncorr, times.shape[0]), dtype=np.bool, comm=comm)
        self.mask[:] = False

    @classmethod
    def from_timestream_and_mask(cls, ts, mask):
        """Create from a ``TimeStream`` object and a mask.

        Parameters
        ----------
        ts : TimeStream
            Timestream object to use.
        mask : mpidataset.MPIArray
            Distributed array of the mask.

        Returns
        -------
        mts : MaskedTimeStream
        """

        mts = cls(np.zeros(1), 1, 1, comm=ts.comm)

        mts._attrs = ts._attrs.copy()
        mts._common = ts._common.copy()
        mts._distributed = ts._distributed.copy()

        mts._distributed['mask'] = mask

        return mts

"""
=========================================================
Parallel data containers (:mod:`~ch_pipeline.containers`)
=========================================================

.. currentmodule:: ch_pipeline.containers

Containers for holding various types of analysis data in a dsitributed fashion.

Containers
==========

.. autosummary::
    :toctree: generated/

    TimeStream
    MaskedTimeStream
    SiderealStream
"""

import numpy as np

from caput import mpidataset, mpiutil
from ch_util import andata
from mpi4py import MPI

import gc


class Map(mpidataset.MPIDataset):

    _common = {'freq': None}

    _distributed = {'map': None}

    @property
    def freq(self):
        return self.common['freq']

    @property
    def map(self):
        return self.distributed['map']


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
    weight : mpidataset.MPIArray
        Array of weights for each point.
    ra : np.ndarray
        RA samples.
    """
    _common = { 'ra': None,
                'freq': None,
                'input': None }

    _distributed = { 'vis': None,
                     'weight': None }

    @property
    def vis(self):
        return self['vis']

    @property
    def weight(self):
        return self['weight']

    @property
    def ra(self):
        return self['ra']

    @property
    def freq(self):
        return self.common['freq']

    @property
    def input(self):
        return self.common['input']

    def __init__(self, nra, freq, ncorr, comm=None):

        mpidataset.MPIDataset.__init__(self, comm)

        nfreq = len(freq)

        self.common['ra'] = np.linspace(0.0, 360.0, nra, endpoint=False)
        self.common['freq'] = freq
        self.distributed['vis'] = mpidataset.MPIArray((nfreq, ncorr, nra), dtype=np.complex128, comm=comm)


class MModes(mpidataset.MPIDataset):
    """Parallel container for holding m-mode data.

    Parameters
    ----------
    mmax : integer
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
    weight : mpidataset.MPIArray
        Array of weights for each point.
    """

    _common = { 'freq': None,
                'input': None }

    _distributed = { 'vis': None }

    @property
    def vis(self):
        return self['vis']

    @property
    def weight(self):
        return self['weight']

    @property
    def freq(self):
        return self.common['freq']

    @property
    def input(self):
        return self.common['input']

    def __init__(self, mmax, freq, ncorr, comm=None):

        mpidataset.MPIDataset.__init__(self, comm)

        nfreq = len(freq)

        self.common['freq'] = freq
        self.distributed['vis'] = mpidataset.MPIArray((mmax+1, 2, nfreq, ncorr), dtype=np.complex128, comm=comm)


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
    weight : boolean, optional
        Add a weight array or not.
    gain : boolean, optional
        Add a gain array or not.
    copy_attrs : MPIDataset, optional
        If set, copy the attrs from this dataset.

    Attributes
    ----------
    timestamp : np.ndarray
        Timestamps.
    vis : mpidataset.MPIArray
        Visibility array.
    weight : mpidataset.MPIArray
        Array for storing weights used for tracking noise and RFI.
    gain : mpidataset.MPIArray
        Gains that have been applied to the dataset.
    gain_dr : mpidataset.MPIArray
        Dynamic range of gain solution.

    Methods
    -------
    from_acq_files
    add_weight
    add_gain
    """

    _common = { 'timestamp': None,
                'freq': None,
                'input': None }

    _distributed = { 'vis': None,
                     'weight': None,
                     'gain': None,
                     'gain_dr': None }

    @property
    def timestamp(self):
        return self['timestamp']

    @property
    def freq(self):
        return self.common['freq']

    @property
    def input(self):
        return self.common['input']

    @property
    def vis(self):
        return self['vis']

    @property
    def weight(self):
        return self['weight']

    @property
    def gain(self):
        return self['gain']

    @property
    def gain_dr(self):
        return self['gain_dr']

    def __init__(self, times, freq, ncorr, comm=None, weight=False, gain=False, copy_attrs=None):

        mpidataset.MPIDataset.__init__(self, comm)

        nfreq = len(freq)

        self.common['timestamp'] = times
        self.common['freq'] = freq
        self.distributed['vis'] = mpidataset.MPIArray((nfreq, ncorr, times.shape[0]), dtype=np.complex128, comm=comm)

        # Add gains if required
        if weight:
            self.add_weight()

        # Add weights if required
        if gain:
            self.add_gains()

        # Copy attributes from another dataset
        if copy_attrs is not None and copy_attrs.attrs is not None:
            self._attrs = copy_attrs.attrs.copy()

    def add_weight(self):
        """Add a weight array to a timestream without one.
        """

        if self.weight is None:
            self._distributed['weight'] = mpidataset.MPIArray(self.vis.global_shape, axis=self.vis.axis,
                                                              dtype=np.float64, comm=self.vis.comm)

    def add_gains(self):
        """Add a gain array to a timestream without one.
        """

        if self.gain is None:
            nfreq, ncorr, ntime = self.vis.global_shape
            ninput = int((2 * ncorr)**0.5)

            self.distributed['gain'] = mpidataset.MPIArray((nfreq, ninput, ntime), axis=self.vis.axis,
                                                           dtype=np.complex128, comm=self.vis.comm)
            self.distributed['gain_dr'] = mpidataset.MPIArray((nfreq, 1, ntime), axis=self.vis.axis,
                                                              dtype=np.float32, comm=self.vis.comm)

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
        ## NOTE: we have frequency explicit garbage collections to free large
        ## AnData objects which may have many cyclic references.

        if comm is None:
            comm = MPI.COMM_WORLD

        # Extract data shape from first file, and distribute to all ranks
        vis_shape = None
        freq = None
        inputs = None

        if comm.rank == 0:
            # Open first file and check shape
            d0 = andata.CorrData.from_acq_h5(files[0])
            vis_shape = d0.vis.shape

            freq = d0.index_map['freq']
            inputs = d0.index_map['input']

            d0.close()

            del d0
            gc.collect()

        vis_shape = comm.bcast(vis_shape, root=0)
        freq = comm.bcast(freq, root=0)
        inputs = comm.bcast(inputs, root=0)

        # Unpack to get the individual lengths
        nfreq, nprod, ntime = vis_shape
        nfile = len(files)

        # Create distribute dataset
        dset = mpidataset.MPIArray((nfile, nfreq, nprod, ntime),
                                   dtype=np.complex64, comm=comm)

        # Timestamps
        timestamps = []

        for li, gi in dset.enumerate(0):

            lfile = files[gi]

            print "Rank %i reading %s" % (comm.rank, lfile)
            # Load file
            df = andata.CorrData.from_acq_h5(lfile)

            # Copy data only if shape is correct
            if df.vis.shape == vis_shape:
                # Copy into local dataset
                dset[li] = df.vis[:]
            # However, allow short time axis on last file
            elif (gi == (nfile - 1)) and (df.vis.shape[:-1] == vis_shape[:-1]):
                nt = df.vis.shape[-1]
                dset[li, ..., :nt] = df.vis[:]
            else:
                raise Exception("Data from %s is not the right shape" % lfile)

            # Get timestamps
            timestamps.append((gi, df.timestamp.copy()))

            # Explicitly close to break reference cycles and delete
            df.close()

            del df
            gc.collect()

        # Merge timestamps
        tslist = comm.allgather(timestamps)
        tsflat = [ts for proclist in tslist for ts in proclist]  # Flatten list

        # Create list of order timestamp arrays only
        tsflat = zip(*sorted(tsflat))[1]
        timestamp_array = np.concatenate(tsflat)

        # Redistribute by frequency
        dset = dset.redistribute(1)

        gc.collect()

        # Merge file and time axes
        dset = dset.transpose((1, 2, 0, 3))
        dset = dset.reshape((None, nprod, nfile * ntime))

        # If the last file was short there will be less timestamps, if so trim
        # the dataset to the correct length in time.
        ttime = timestamp_array.size
        if ttime != dset.shape[-1]:
            dset = dset[..., :ttime]

        dset = mpidataset.MPIArray.wrap(dset.astype(np.complex128), axis=0)

        # Create TimeStream class (set zeros sizes to stop allocation)
        ts = cls(timestamp_array, freq, 1, comm=comm)

        # Add input map
        ts.common['input'] = inputs

        # Replace vis dataset with real data
        ts.distributed['vis'] = dset

        # Final garbage collection to free large AnData objects
        gc.collect()

        return ts

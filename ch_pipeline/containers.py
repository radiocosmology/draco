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

from caput import mpidataset, memh5
from ch_util import andata
from mpi4py import MPI

import gc


class ContainerBase(memh5.BasicCont):
    """A base class for pipeline containers.

    This class is designed to do much of the work of setting up pipeline
    containers. It should be derived from, and two variables set `_axes` and
    `_dataset_spec`.

    The variable `_axes` should be a tuple containing the names of axes that
    datasets in this container will use.

    The variable `_dataset_spec` should define the datasets. It's a dictionary
    with the name of the dataset as key. Each entry should be another
    dictionary, the entry 'axes' is mandatory and should be a list of the axes
    the dataset has (these should correspond to entries in `_axes`), as is
    `dtype` which should be a datatype understood by numpy. Other possible
    entries are:

    - `initialise` : if set to `True` the dataset will be created as the container is initialised.

    - `distributed` : the dataset will be distributed if the entry is `True`, if
      `False` it won't be, and if not set it will be distributed if the
      container is set to be.

    - `distributed_axis` : the axis to distribute over. Should be a name given in the `axes` entry.

    Parameters
    ----------
    axes_from : `memh5.BasicCont`, optional
        Another container to copy axis definitions from.
    kwargs : dict
        Should contain entries for all other axes.
    """

    _axes = ()

    _dataset_spec = {}

    def __init__(self, axes_from=None, *args, **kwargs):

        dist = kwargs['distributed'] if 'distributed' in kwargs else True
        comm = kwargs['comm'] if 'comm' in kwargs and dist else None

        # Run base initialiser
        memh5.BasicCont.__init__(self, distributed=dist, comm=comm)

        # Create axis entries
        for axis in self._axes:

            axis_map = None

            # Check if axis is specified in initialiser
            if axis in kwargs:

                # If axis is an integer, turn into an arange as a default definition
                if isinstance(kwargs[axis], int):
                    axis_map = np.arange(kwargs[axis])
                else:
                    axis_map = kwargs[axis]

            # If not set in the arguments copy from another object if set
            elif axes_from is not None and axis in axes_from.index_map:
                axis_map = axes_from.index_map[axis]

            # Set the index_map[axis] if we have a definition, otherwise throw an error
            if axis_map is not None:
                self.create_index_map(axis, axis_map)
            else:
                raise RuntimeError('No definition of axis supplied.')

        # Iterate over datasets and initialise any that specify it
        for name, spec in self._dataset_spec.items():
            if 'initialise' in spec and spec['initialise']:
                self.add_dataset(name)

    def add_dataset(self, name):
        """Create an empty dataset.

        The dataset must be defined in the specification for the container.

        Parameters
        ----------
        name : string
            Name of the dataset to create.

        Returns
        -------
        dset : `memh5.MemDataset`
        """

        # Dataset must be specified
        if name not in self._dataset_spec:
            raise RuntimeError('Dataset name not known.')

        dspec = self._dataset_spec[name]

        # Fetch dataset properties
        axes = dspec['axes']
        dtype = dspec['dtype']

        # Get distribution properties
        dist = dspec['distributed'] if 'distributed' in dspec else self._data._distributed
        comm = self._data._comm if dist else None
        shape = ()

        # Check that all the specified axes are defined, and fetch their lengths
        for axis in axes:
            if axis not in self.index_map:
                raise RuntimeError('Axis not defined in index_map')

            l = len(self.index_map[axis])

            shape += (l,)

        # Fetch distributed axis, and turn into axis index
        dist_axis = dspec['distributed_axis'] if 'distributed_axis' in dspec else axes[0]
        dist_axis = list(axes).index(dist_axis)

        # Create dataset
        dset = self.create_dataset(name, shape=shape, dtype=dtype, distributed=dist,
                                   comm=comm, distributed_axis=dist_axis)

        dset.attrs['axis'] = np.array(axes)

        return dset

    @property
    def datasets(self):
        """Return the datasets in this container.

        Do not try to add a new dataset by assigning to an item of this
        property. Use `create_dataset` instead.

        Returns
        -------
        datasets : read only dictionary
            Entries are :mod:`caput.memh5` datasets.

        """
        out = {}
        for name, value in self._data.iteritems():
            if not memh5.is_group(value):
                out[name] = value
        return memh5.ro_dict(out)


class Map(ContainerBase):
    """Container for holding multifrequency sky maps.

    The maps are packed in format `[freq, pol, pixel]` where the polarisations
    are Stokes I, Q, U and V, and the pixel dimension stores a Healpix map.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    polarisation : bool, optional
        If `True` all Stokes parameters are stored, if `False` only Stokes I is
        stored.
    """

    _axes = ('freq', 'pol', 'pixel')

    _dataset_spec = {
        'map': {
            'axes': ['freq', 'pol', 'pixel'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            }
        }

    def __init__(self, nside=None, polarisation=None, *args, **kwargs):

        # Set up axes from passed arguments
        if nside is not None:
            kwargs['pixel'] = 12*nside**2
        if polarisation is not None:
            kwargs['pol'] = 4 if polarisation else 1

        super(Map, self).__init__(*args, **kwargs)

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def map(self):
        return self.datasets['map']


class SiderealStream(ContainerBase):
    """A container for holding a visibility dataset in sidereal time.

    Parameters
    ----------
    ra : int
        The number of points to divide the RA axis up into.
    """

    _axes = ('freq', 'prod', 'input', 'ra')

    _dataset_spec = {
        'vis': {
            'axes': ['freq', 'prod', 'ra'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },

        'vis_weight': {
            'axes': ['freq', 'prod', 'ra'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
            },

        'gain': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.complex128,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
            }
        }

    def __init__(self, ra=None, *args, **kwargs):

        # Set up axes passed ra langth
        if ra is not None:
            if isinstance(ra, int):
                ra = np.linspace(0.0, 360.0, ra, endpoint=False)
            kwargs['ra'] = ra

        super(SiderealStream, self).__init__(*args, **kwargs)

    @property
    def vis(self):
        return self.datasets['vis']

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def weight(self):
        return self.datasets['vis_weight']

    @property
    def ra(self):
        return self.index_map['ra']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']


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


class GainData(mpidataset.MPIDataset):
    """Parallel container for holding gain data.

    Attributes
    ----------
    gain : mpidataset.MPIArray
        Gain array.
    timestamp : np.ndarray
        Time samples.
    """
    _common = { 'timestamp': None }

    _distributed = { 'gain': None }

    @property
    def gain(self):
        return self['gain']

    @property
    def timestamp(self):
        return self['timestamp']

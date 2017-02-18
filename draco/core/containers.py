"""Distributed containers for holding various types of analysis data.

Containers
==========

.. autosummary::
    :toctree:

    TimeStream
    SiderealStream
    GainData
    StaticGainData
    Map
    MModes
    RingMap

Container Base Classes
----------------------

.. autosummary::
    :toctree:

    ContainerBase
    TODContainer

Helper Routines
---------------

These routines are designed to be replaced by other packages trying to insert
their own custom container types.

.. autosummary::
    :toctree:

    empty_like
    empty_timestream
"""

import numpy as np

from caput import memh5, tod


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
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : `memh5.BasicCont`, optional
        Another container to copy attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    kwargs : dict
        Should contain entries for all other axes.
    """

    _axes = ()

    _dataset_spec = {}

    def __init__(self, *args, **kwargs):

        # Pull out the values of needed arguments
        axes_from = kwargs.pop('axes_from', None)
        attrs_from = kwargs.pop('attrs_from', None)
        dist = kwargs.pop('distributed', True)
        comm = kwargs.pop('comm', None)

        # Run base initialiser
        memh5.BasicCont.__init__(self, distributed=dist, comm=comm)

        # Check to see if this call looks like it was called like
        # memh5.MemDiskGroup would have been. If it is, we're probably trying to
        # create a bare container, so don't initialise any datasets. This
        # behaviour is needed to support tod.concatenate
        if len(args) or 'data_group' in kwargs:
            return

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
                raise RuntimeError('No definition of axis %s supplied.' % axis)

        # Iterate over datasets and initialise any that specify it
        for name, spec in self._dataset_spec.items():
            if 'initialise' in spec and spec['initialise']:
                self.add_dataset(name)

        # Copy over attributes
        if attrs_from is not None:

            # Copy attributes from container root
            memh5.copyattrs(attrs_from.attrs, self.attrs)

            # Copy attributes over from any common datasets
            for name in self._dataset_spec.keys():
                if name in self.datasets and name in attrs_from.datasets:
                    memh5.copyattrs(attrs_from.datasets[name].attrs,
                                    self.datasets[name].attrs)

            # Make sure that the __memh5_subclass attribute is accurate
            clspath = self.__class__.__module__ + '.' + self.__class__.__name__
            clsattr = self.attrs.get('__memh5_subclass', None)
            if clsattr and (clsattr != clspath):
                self.attrs['__memh5_subclass'] = clspath


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
        shape = ()

        # Check that all the specified axes are defined, and fetch their lengths
        for axis in axes:
            if axis not in self.index_map:
                if isinstance(axis, int):
                    l = axis
                else:
                    raise RuntimeError('Axis not defined in index_map')
            else:
                l = len(self.index_map[axis])

            shape += (l,)

        # Fetch distributed axis, and turn into axis index
        dist_axis = dspec['distributed_axis'] if 'distributed_axis' in dspec else axes[0]
        dist_axis = list(axes).index(dist_axis)

        # Create dataset
        dset = self.create_dataset(name, shape=shape, dtype=dtype, distributed=dist,
                                   distributed_axis=dist_axis)

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


class TODContainer(ContainerBase, tod.TOData):
    """A pipeline container for time ordered data.

    This works like a normal :class:`ContainerBase` container, with the added
    ability to be concatenated, and treated like a a :class:`tod.TOData`
    instance.
    """

    @property
    def time(self):
        try:
            return self.index_map['time'][:]['ctime']
        # Need to check for both types as different numpy versions return
        # different exceptions.
        except (IndexError, ValueError):
            return self.index_map['time'][:]


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

    def __init__(self, nside=None, polarisation=True, *args, **kwargs):

        # Set up axes from passed arguments
        if nside is not None:
            kwargs['pixel'] = 12 * nside**2

        kwargs['pol'] = np.array(['I', 'Q', 'U', 'V']) if polarisation else np.array(['I'])

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
            'dtype': np.complex64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },

        'vis_weight': {
            'axes': ['freq', 'prod', 'ra'],
            'dtype': np.float32,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },

        'gain': {
            'axes': ['freq', 'input', 'ra'],
            'dtype': np.complex64,
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

        # Resolve product map
        prod = None
        if 'prod' in kwargs:
            prod = kwargs['prod']
        elif ('axes_from' in kwargs) and ('prod' in kwargs['axes_from'].index_map):
            prod = kwargs['axes_from'].index_map['prod']

        # Resolve input map
        inputs = None
        if 'input' in kwargs:
            inputs = kwargs['input']
        elif ('axes_from' in kwargs) and ('input' in kwargs['axes_from'].index_map):
            inputs = kwargs['axes_from'].index_map['input']

        # Automatically construct product map from inputs if not given
        if prod is None and inputs is not None:
            nfeed = inputs if isinstance(inputs, int) else len(inputs)
            kwargs['prod'] = np.array([[fi, fj] for fi in range(nfeed) for fj in range(fi, nfeed)])

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


class TimeStream(TODContainer):
    """A container for holding a visibility dataset in time.

    This should look similar enough to the CHIME
    :class:`~ch_util.andata.CorrData` container that they can be used
    interchangably in most cases.
    """

    _axes = ('freq', 'prod', 'input', 'time')

    _dataset_spec = {
        'vis': {
            'axes': ['freq', 'prod', 'time'],
            'dtype': np.complex64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },

        'vis_weight': {
            'axes': ['freq', 'prod', 'time'],
            'dtype': np.float32,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },

        'gain': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.complex64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

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
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']


class MModes(ContainerBase):
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

    _axes = ('m', 'msign', 'freq', 'prod', 'input')

    _dataset_spec = {
        'vis': {
            'axes': ['m', 'msign', 'freq', 'prod'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'm'
        },

        'vis_weight': {
            'axes': ['m', 'msign', 'freq', 'prod'],
            'dtype': np.float64,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'm'
        },
    }

    @property
    def vis(self):
        return self.datasets['vis']

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

    def __init__(self, mmax=None, *args, **kwargs):

        # Set up axes from passed arguments
        if mmax is not None:
            kwargs['m'] = mmax + 1

        # Ensure the sign axis is set correctly
        kwargs['msign'] = np.array(['+', '-'])

        super(MModes, self).__init__(*args, **kwargs)


class GainData(TODContainer):
    """Parallel container for holding gain data.
    """

    _axes = ('freq', 'input', 'time')

    _dataset_spec = {
        'gain': {
            'axes': ['freq', 'input', 'time'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'weight': {
            'axes': ['freq', 'time'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def weight(self):
        try:
            return self.datasets['weight']
        except KeyError:
            return None

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']


class StaticGainData(ContainerBase):
    """Parallel container for holding static gain data (i.e. non time varying).
    """

    _axes = ('freq', 'input')

    _dataset_spec = {
        'gain': {
            'axes': ['freq', 'input'],
            'dtype': np.complex128,
            'initialise': True,
            'distributed': True,
            'distributed_axis': 'freq'
        },
        'weight': {
            'axes': ['freq'],
            'dtype': np.float64,
            'initialise': False,
            'distributed': True,
            'distributed_axis': 'freq'
        }
    }

    @property
    def gain(self):
        return self.datasets['gain']

    @property
    def weight(self):
        return self.datasets['weight']

    @property
    def freq(self):
        return self.index_map['freq']

    @property
    def input(self):
        return self.index_map['input']


def empty_like(obj, **kwargs):
    """Create an empty container like `obj`.

    Parameters
    ----------
    obj : ContainerBase
        Container to base this one off.
    kwargs : optional
        Optional definitions of specific axes we want to override. Works in the
        same way as the `ContainerBase` constructor, though `axes_from=obj` and
        `attrs_from=obj` are implied.

    Returns
    -------
    newobj : container.ContainerBase
        New data container.
    """

    if isinstance(obj, ContainerBase):
        return obj.__class__(axes_from=obj, attrs_from=obj, **kwargs)
    else:
        raise RuntimeError("I don't know how to deal with data type %s" % obj.__class__.__name__)


def empty_timestream(**kwargs):
    """Create a new timestream container.

    This indirect call exists so it can be replaced to return custom timestream
    types.

    Parameters
    ----------
    kwargs : optional
        Arguments to pass to the timestream constructor.

    Returns
    -------
    ts : TimeStream
    """
    return TimeStream(**kwargs)

"""Distributed containers for holding various types of analysis data.

Containers
==========

.. autosummary::
    :toctree:

    TimeStream
    SiderealStream
    Map
    GridBeam
    TrackBeam
    MModes
    SVDModes
    KLModes
    CommonModeGainData
    CommonModeSiderealGainData
    GainData
    SiderealGainData
    StaticGainData
    DelaySpectrum
    Powerspectrum2D
    SVDSpectrum
    FrequencyStack
    SourceCatalog
    SpectroscopicCatalog
    FormedBeam
    FormedBeamHA


Container Base Classes
----------------------

.. autosummary::
    :toctree:

    ContainerBase
    TODContainer
    VisContainer
    SampleVarianceContainer

Helper Routines
---------------

These routines are designed to be replaced by other packages trying to insert
their own custom container types.

.. autosummary::
    :toctree:

    empty_like
    empty_timestream
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
from past.builtins import basestring

# === End Python 2/3 compatibility

import inspect

import numpy as np

from caput import memh5, tod

from ..util import tools

# Try to import bitshuffle to set the default compression options
try:
    import bitshuffle.h5

    COMPRESSION = bitshuffle.h5.H5FILTER
    COMPRESSION_OPTS = (0, bitshuffle.h5.H5_COMPRESS_LZ4)
except ImportError:
    COMPRESSION = None
    COMPRESSION_OPTS = None


class ContainerBase(memh5.BasicCont):
    """A base class for pipeline containers.

    This class is designed to do much of the work of setting up pipeline
    containers. It should be derived from, and two variables set `_axes` and
    `_dataset_spec`. See the `Notes`_ section for details.

    Parameters
    ----------
    axes_from : `memh5.BasicCont`, optional
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : `memh5.BasicCont`, optional
        Another container to copy attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    skip_datasets : bool, optional
        Skip creating datasets. They must all be added manually with
        `.add_dataset` regardless of the entry in `.dataset_spec`. Default is False.
    kwargs : dict
        Should contain entries for all other axes.

    Notes
    -----

    Inheritance from other `ContainerBase` subclasses should work as expected,
    with datasets defined in super classes appearing as expected, and being
    overriden where they are redefined in the derived class.

    The variable `_axes` should be a tuple containing the names of axes that
    datasets in this container will use.

    The variable `_dataset_spec` should define the datasets. It's a dictionary
    with the name of the dataset as key. Each entry should be another
    dictionary, the entry 'axes' is mandatory and should be a list of the axes
    the dataset has (these should correspond to entries in `_axes`), as is
    `dtype` which should be a datatype understood by numpy. Other possible
    entries are:

    - `initialise` : if set to `True` the dataset will be created as the
      container is initialised.

    - `distributed` : the dataset will be distributed if the entry is `True`, if
      `False` it won't be, and if not set it will be distributed if the
      container is set to be.

    - `distributed_axis` : the axis to distribute over. Should be a name given
      in the `axes` entry.
    """

    _axes = ()

    _dataset_spec = {}

    convert_attribute_strings = True
    convert_dataset_strings = True

    def __init__(self, *args, **kwargs):

        # Pull out the values of needed arguments
        axes_from = kwargs.pop("axes_from", None)
        attrs_from = kwargs.pop("attrs_from", None)
        skip_datasets = kwargs.pop("skip_datasets", False)
        dist = kwargs.pop("distributed", True)
        comm = kwargs.pop("comm", None)
        self.allow_chunked = kwargs.pop("allow_chunked", False)

        # Run base initialiser
        memh5.BasicCont.__init__(self, distributed=dist, comm=comm)

        # Check to see if this call looks like it was called like
        # memh5.MemDiskGroup would have been. If it is, we're probably trying to
        # create a bare container, so don't initialise any datasets. This
        # behaviour is needed to support tod.concatenate
        if len(args) or "data_group" in kwargs:
            return

        # Create axis entries
        for axis in self.axes:

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
                raise RuntimeError("No definition of axis %s supplied." % axis)

        # Iterate over datasets and initialise any that specify it
        if not skip_datasets:
            for name, spec in self.dataset_spec.items():
                if "initialise" in spec and spec["initialise"]:
                    self.add_dataset(name)

        # Copy over attributes
        if attrs_from is not None:

            # Copy attributes from container root
            memh5.copyattrs(attrs_from.attrs, self.attrs)

            # Copy attributes over from any common datasets
            for name in self.dataset_spec.keys():
                if name in self.datasets and name in attrs_from.datasets:
                    attrs_no_axis = {
                        k: v
                        for k, v in attrs_from.datasets[name].attrs.items()
                        if k != "axis"
                    }
                    memh5.copyattrs(attrs_no_axis, self.datasets[name].attrs)

            # Make sure that the __memh5_subclass attribute is accurate
            clspath = self.__class__.__module__ + "." + self.__class__.__name__
            clsattr = self.attrs.get("__memh5_subclass", None)
            if clsattr and (clsattr != clspath):
                self.attrs["__memh5_subclass"] = clspath

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
        if name not in self.dataset_spec:
            raise RuntimeError("Dataset name not known.")

        dspec = self.dataset_spec[name]

        # Fetch dataset properties
        axes = dspec["axes"]
        dtype = dspec["dtype"]
        chunks, compression, compression_opts = None, None, None
        if self.allow_chunked:
            chunks = dspec.get("chunks", None)
            compression = dspec.get("compression", None)
            compression_opts = dspec.get("compression_opts", None)

        # Get distribution properties
        dist = self.distributed and dspec.get("distributed", True)
        shape = ()

        # Check that all the specified axes are defined, and fetch their lengths
        for axis in axes:
            if axis not in self.index_map:
                if isinstance(axis, int):
                    l = axis
                else:
                    raise RuntimeError("Axis not defined in index_map")
            else:
                l = len(self.index_map[axis])

            shape += (l,)

        # Fetch distributed axis, and turn into axis index
        dist_axis = (
            dspec["distributed_axis"] if "distributed_axis" in dspec else axes[0]
        )
        dist_axis = list(axes).index(dist_axis)

        # Check chunk dimensions are consistent with axis
        if chunks is not None:
            final_chunks = ()
            for i, l in enumerate(shape):
                final_chunks += (min(chunks[i], l),)
            chunks = final_chunks

        # Create dataset
        dset = self.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            distributed=dist,
            distributed_axis=dist_axis,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )

        dset.attrs["axis"] = np.array(axes)

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
        for name, value in self._data.items():
            if not memh5.is_group(value):
                out[name] = value
        return memh5.ro_dict(out)

    @property
    def dataset_spec(self):
        """Return a copy of the fully resolved dataset specifiction as a
        dictionary.
        """

        ddict = {}

        # Iterate over the reversed MRO and look for _table_spec attributes
        # which get added to a temporary dict. We go over the reversed MRO so
        # that the `tdict.update` overrides tables in base classes.`
        for cls in inspect.getmro(self.__class__)[::-1]:

            try:
                # NOTE: this is a little ugly as the following line will drop
                # down to base classes if dataset_spec isn't present, and thus
                # try and `update` with the same values again.
                ddict.update(cls._dataset_spec)
            except AttributeError:
                pass

        # Add in any _dataset_spec found on the instance
        ddict.update(self.__dict__.get("_dataset_spec", {}))

        # Ensure that the dataset_spec is the same order on all ranks
        return {k: ddict[k] for k in sorted(ddict)}

    @classmethod
    def _class_axes(cls):
        """Return the set of axes for this container defined by this class and the base classes.
        """
        axes = set()

        # Iterate over the reversed MRO and look for _table_spec attributes
        # which get added to a temporary dict. We go over the reversed MRO so
        # that the `tdict.update` overrides tables in base classes.
        for c in inspect.getmro(cls)[::-1]:

            try:
                axes |= set(c._axes)
            except AttributeError:
                pass

        # This must be the same order on all ranks, so we need to explicitly sort to get around the
        # hash randomization
        return tuple(sorted(axes))

    @property
    def axes(self):
        """The set of axes for this container including any defined on the instance.
        """
        axes = set(self._class_axes())

        # Add in any axes found on the instance (this is needed to support the table classes where
        # the axes get added at run time)
        axes |= set(self.__dict__.get("_axes", []))

        # This must be the same order on all ranks, so we need to explicitly sort to get around the
        # hash randomization
        return tuple(sorted(axes))

    @classmethod
    def _make_selections(cls, sel_args):
        """
        Match down-selection arguments to axes of datasets.

        Parses sel_* argument and returns dict mapping dataset names to selections.

        Parameters
        ----------
        sel_args : dict
            Should contain valid numpy indexes as values and axis names (str) as keys.

        Returns
        -------
        dict
            Mapping of dataset names to numpy indexes for downselection of the data.
            Also includes another dict under the key "index_map" that includes
            the selections for those.
        """
        # Check if all those axes exist
        for axis in sel_args.keys():
            if axis not in cls._class_axes():
                raise RuntimeError("No '{}' axis found to select from.".format(axis))

        # Build selections dict
        selections = {}
        for name, dataset in cls._dataset_spec.items():
            ds_axes = dataset["axes"]
            sel = []
            ds_relevant = False
            for axis in ds_axes:
                if axis in sel_args:
                    sel.append(sel_args[axis])
                    ds_relevant = True
                else:
                    sel.append(slice(None))
            if ds_relevant:
                selections["/" + name] = tuple(sel)

        # add index maps selections
        for axis, sel in sel_args.items():
            selections["/index_map/" + axis] = sel

        return selections

    def copy(self, shared=None):
        """Copy this container, optionally sharing the source datasets.

        This routine will create a copy of the container. By default this is
        as full copy with the contents fully independent. However, a set of
        dataset names can be given that will share the same data as the
        source to save memory for large datasets. These will just view the
        same memory, so any modification to either the original or the copy
        will be visible to the other. This includes all write operations,
        addition and removal of attributes, redistribution etc. This
        functionality should be used with caution and clearly documented.

        Parameters
        ----------
        shared : list, optional
            A list of datasets whose content will be shared with the original.

        Returns
        -------
        copy : subclass of ContainerBase
            The copied container.
        """
        new_cont = self.__class__(
            attrs_from=self,
            axes_from=self,
            skip_datasets=True,
            distributed=self.distributed,
            comm=self.comm,
        )

        # Loop over datasets that exist in the source and either add a view of
        # the source dataset, or perform a full copy
        for name, data in self.datasets.items():

            if shared and name in shared:
                # TODO: find a way to do this that doesn't depend on the
                # internal implementation of BasicCont and MemGroup
                # NOTE: we don't use `.view()` on the RHS here as we want to
                # preserve the shared data through redistributions
                new_cont._data._get_storage()[name] = self._data._get_storage()[name]
            else:
                dset = new_cont.add_dataset(name)

                # Ensure that we have exactly the same distribution
                if dset.distributed:
                    dset.redistribute(data.distributed_axis)

                # Copy over the data and attributes
                dset[:] = data[:]
                memh5.copyattrs(data.attrs, dset.attrs)

        return new_cont


class TableBase(ContainerBase):
    """A base class for containers holding tables of data.

    Similar to the `ContainerBase` class, the container is defined through a
    dictionary given as a `_table_spec` class attribute. The container may also
    hold generic datasets by specifying `_dataset_spec` as with `ContainerBase`.
    See `Notes`_ for details.

    Parameters
    ----------
    axes_from : `memh5.BasicCont`, optional
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : `memh5.BasicCont`, optional
        Another container to copy attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    kwargs : dict
        Should contain definitions for all other table axes.

    Notes
    -----

    A `_table_spec` consists of a dictionary mapping table names into a
    description of the table. That description is another dictionary containing
    several entries.

    - `columns` : the set of columns in the table. Given as a list of
      `(name, dtype)` pairs.

    - `axis` : an optional name for the rows of the table. This is automatically
      generated as `'<tablename>_index'` if not explicitly set. This corresponds
      to an `index_map` entry on the container.

    - `initialise` : whether to create the table by default.

    - `distributed` : whether the table is distributed, or common across all MPI ranks.

    An example `_table_spec` entry is::

        _table_spec = {
            'quasars': {
                'columns': [
                    ['ra': np.float64],
                    ['dec': np.float64],
                    ['z': np.float64]
                ],
                'distributed': False,
                'axis': 'quasar_id'
            }
            'quasar_mask': {
                'columns': [
                    ['mask', np.bool]
                ],
                'axis': 'quasar_id'
            }
        }
    """

    _table_spec = {}

    def __init__(self, *args, **kwargs):

        # Get the dataset specifiction for this class (not any base classes), or
        # an empty dictionary if it does not exist. Do the same for the axes entry..
        dspec = self.__class__.__dict__.get("_dataset_spec", {})
        axes = self.__class__.__dict__.get("_axes", ())

        # Iterate over all table_spec entries and construct dataset specifications for them.
        for name, spec in self.table_spec.items():

            # Get the specifieid axis or if not present create a unique one for
            # this table entry
            axis = spec.get("axis", name + "_index")

            dtype = self._create_dtype(spec["columns"])

            _dataset = {
                "axes": [axis],
                "dtype": dtype,
                "initialise": spec.get("initialise", True),
                "distributed": spec.get("distributed", False),
                "distributed_axis": axis,
            }

            dspec[name] = _dataset

            if axis not in axes:
                axes += (axis,)

        self._dataset_spec = dspec
        self._axes = axes

        super(TableBase, self).__init__(*args, **kwargs)

    def _create_dtype(self, columns):
        """Take a dictionary of columns and turn into the
        appropriate compound data type.
        """

        dt = []
        for ci, (name, dtype) in enumerate(columns):
            if not isinstance(name, basestring):
                raise ValueError("Column %i is invalid" % ci)
            dt.append((name, dtype))

        return dt

    @property
    def table_spec(self):
        """Return a copy of the fully resolved table specifiction as a
        dictionary.
        """
        import inspect

        tdict = {}

        for cls in inspect.getmro(self.__class__)[::-1]:

            try:
                tdict.update(cls._table_spec)
            except AttributeError:
                pass

        return tdict


class TODContainer(ContainerBase, tod.TOData):
    """A pipeline container for time ordered data.

    This works like a normal :class:`ContainerBase` container, with the added
    ability to be concatenated, and treated like a a :class:`tod.TOData`
    instance.
    """

    _axes = ("time",)

    @property
    def time(self):
        try:
            return self.index_map["time"][:]["ctime"]
        # Need to check for both types as different numpy versions return
        # different exceptions.
        except (IndexError, ValueError):
            return self.index_map["time"][:]


class VisContainer(ContainerBase):
    """A base container for holding a visibility dataset.

    This works like a :class:`ContainerBase` container, with the
    ability to create visibility specific axes, if they are not
    passed as a kwargs parameter.

    Additionally this container has visibility specific defined properties
    such as 'vis', 'weight', 'freq', 'input', 'prod', 'stack',
    'prodstack', 'conjugate'.

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

    _axes = ("freq", "input", "prod", "stack")

    def __init__(self, *args, **kwargs):
        # Resolve product map
        prod = None
        if "prod" in kwargs:
            prod = kwargs["prod"]
        elif ("axes_from" in kwargs) and ("prod" in kwargs["axes_from"].index_map):
            prod = kwargs["axes_from"].index_map["prod"]

        # Resolve input map
        inputs = None
        if "input" in kwargs:
            inputs = kwargs["input"]
        elif ("axes_from" in kwargs) and ("input" in kwargs["axes_from"].index_map):
            inputs = kwargs["axes_from"].index_map["input"]

        # Resolve stack map
        stack = None
        if "stack" in kwargs:
            stack = kwargs["stack"]
        elif ("axes_from" in kwargs) and ("stack" in kwargs["axes_from"].index_map):
            stack = kwargs["axes_from"].index_map["stack"]

        # Automatically construct product map from inputs if not given
        if prod is None and inputs is not None:
            nfeed = inputs if isinstance(inputs, int) else len(inputs)
            kwargs["prod"] = np.array(
                [[fi, fj] for fi in range(nfeed) for fj in range(fi, nfeed)]
            )

        if stack is None and prod is not None:
            stack = np.empty_like(prod, dtype=[("prod", "<u4"), ("conjugate", "u1")])
            stack["prod"][:] = np.arange(len(prod))
            stack["conjugate"] = 0
            kwargs["stack"] = stack

        # Call initializer from `ContainerBase`
        super(VisContainer, self).__init__(*args, **kwargs)

        reverse_map_stack = None
        # Create reverse map
        if "reverse_map_stack" in kwargs:
            # If axis is an integer, turn into an arange as a default definition
            if isinstance(kwargs["reverse_map_stack"], int):
                reverse_map_stack = np.arange(kwargs["reverse_map_stack"])
            else:
                reverse_map_stack = kwargs["reverse_map_stack"]
        # If not set in the arguments copy from another object if set
        elif ("axes_from" in kwargs) and ("stack" in kwargs["axes_from"].reverse_map):
            reverse_map_stack = kwargs["axes_from"].reverse_map["stack"]

        # Set the reverse_map['stack'] if we have a definition,
        # otherwise do NOT throw an error, errors are thrown in
        # classes that actually need a reverse stack
        if reverse_map_stack is not None:
            self.create_reverse_map("stack", reverse_map_stack)

    @property
    def vis(self):
        """The visibility like dataset."""
        return self.datasets["vis"]

    @property
    def weight(self):
        """The visibility weights."""
        return self.datasets["vis_weight"]

    @property
    def freq(self):
        """The frequency axis."""
        return self.index_map["freq"]["centre"]

    @property
    def input(self):
        """The correlated inputs."""
        return self.index_map["input"]

    @property
    def prod(self):
        """All the pairwise products that are represented in the data."""
        return self.index_map["prod"]

    @property
    def stack(self):
        """The stacks definition as an index (and conjugation) of a member product."""
        return self.index_map["stack"]

    @property
    def prodstack(self):
        """A pair of input indices representative of those in the stack.

        Note, these are correctly conjugated on return, and so calculations
        of the baseline and polarisation can be done without additionally
        looking up the stack conjugation.
        """
        if not self.is_stacked:
            return self.prod

        t = self.index_map["prod"][:][self.index_map["stack"]["prod"]]

        prodmap = t.copy()
        conj = self.stack["conjugate"]
        prodmap["input_a"] = np.where(conj, t["input_b"], t["input_a"])
        prodmap["input_b"] = np.where(conj, t["input_a"], t["input_b"])

        return prodmap

    @property
    def is_stacked(self):
        """Test if the data has been stacked or not."""
        return len(self.stack) != len(self.prod)


class SampleVarianceContainer(ContainerBase):
    """Base container for holding the sample variance over observations.

    This works like :class:`ContainerBase` but provides additional capabilities
    for containers that may be used to hold the sample mean and variance over
    complex-valued observations.  These capabilities include automatic definition
    of the component axis, properties for accessing standard datasets, properties
    that rotate the sample variance into common bases, and a `sample_weight` property
    that provides an equivalent to the `weight` dataset that is determined from the
    sample variance over observations.

    Subclasses must include a `sample_variance` and `nsample` dataset
    in there `_dataset_spec` dictionary.  They must also specify a
    `_mean` property that returns the dataset containing the mean over observations.
    """

    _axes = ("component",)

    def __init__(self, *args, **kwargs):

        # Set component axis to default real-imaginary basis if not already provided
        if "component" not in kwargs:
            kwargs["component"] = np.array(
                [("real", "real"), ("real", "imag"), ("imag", "imag")],
                dtype=[("component_a", "<U8"), ("component_b", "<U8")],
            )

        super(SampleVarianceContainer, self).__init__(*args, **kwargs)

    @property
    def component(self):
        return self.index_map["component"]

    @property
    def sample_variance(self):
        """Convience access to the sample variance dataset.

        Returns
        -------
        C: np.ndarray[ncomponent, ...]
            The variance over the dimension that was stacked
            (e.g., sidereal days, holographic observations)
            in the default real-imaginary basis. The array is packed
            into upper-triangle format such that the component axis
            contains [('real', 'real'), ('real', 'imag'), ('imag', 'imag')].
        """
        if "sample_variance" in self.datasets:
            return self.datasets["sample_variance"]
        else:
            raise KeyError("Dataset 'sample_variance' not initialised.")

    @property
    def sample_variance_iq(self):
        """Rotate the sample variance to the in-phase/quadrature basis.

        Returns
        -------
        C: np.ndarray[ncomponent, ...]
            The `sample_variance` dataset in the in-phase/quadrature basis,
            packed into upper triangle format such that the component axis
            contains [('I', 'I'), ('I', 'Q'), ('Q', 'Q')].
        """
        C = self.sample_variance[:].view(np.ndarray)

        # Construct rotation coefficients from average vis angle
        phi = np.angle(self._mean[:].view(np.ndarray))
        cc = np.cos(phi) ** 2
        cs = np.cos(phi) * np.sin(phi)
        ss = np.sin(phi) ** 2

        # Rotate the covariance matrix from real-imag to in-phase/quadrature
        Cphi = np.zeros_like(C)
        Cphi[0] = cc * C[0] + 2 * cs * C[1] + ss * C[2]
        Cphi[1] = -cs * C[0] + (cc - ss) * C[1] + cs * C[2]
        Cphi[2] = ss * C[0] - 2 * cs * C[1] + cc * C[2]

        return Cphi

    @property
    def sample_variance_amp_phase(self):
        """Calculate the amplitude/phase covariance.

        This interpretation is only valid if the fractional
        variations in the amplitude and phase are small.

        Returns
        -------
        C: np.ndarray[ncomponent, ...]
            The observed amplitude/phase covariance matrix, packed
            into upper triangle format such that the component axis
            contains [('amp', 'amp'), ('amp', 'phase'), ('phase', 'phase')].
        """
        # Rotate to in-phase/quadrature basis and then
        # normalize by squared amplitude to convert to
        # fractional units (amplitude) and radians (phase).
        return self.sample_variance_iq * tools.invert_no_zero(
            np.abs(self._mean[:][np.newaxis, ...]) ** 2
        )

    @property
    def nsample(self):
        if "nsample" in self.datasets:
            return self.datasets["nsample"]
        else:
            raise KeyError("Dataset 'nsample' not initialised.")

    @property
    def sample_weight(self):
        """Calculate a weight from the sample variance.

        Returns
        -------
        weight: np.ndarray[...]
            The trace of the `sample_variance` dataset is used
            as an estimate of the total variance and divided by the
            `nsample` dataset to yield the uncertainty on the mean.
            The inverse of this quantity is returned, and can be compared
            directly to the `weight` dataset.
        """
        C = self.sample_variance[:].view(np.ndarray)
        nsample = self.nsample[:].view(np.ndarray)

        return nsample * tools.invert_no_zero(C[0] + C[2])


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

    _axes = ("freq", "pol", "pixel")

    _dataset_spec = {
        "map": {
            "axes": ["freq", "pol", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    def __init__(self, nside=None, polarisation=True, *args, **kwargs):

        # Set up axes from passed arguments
        if nside is not None:
            kwargs["pixel"] = 12 * nside ** 2

        kwargs["pol"] = (
            np.array(["I", "Q", "U", "V"]) if polarisation else np.array(["I"])
        )

        super(Map, self).__init__(*args, **kwargs)

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]

    @property
    def map(self):
        return self.datasets["map"]


class SiderealStream(VisContainer, SampleVarianceContainer):
    """A container for holding a visibility dataset in sidereal time.

    Parameters
    ----------
    ra : int
        The number of points to divide the RA axis up into.
    """

    _axes = ("ra",)

    _dataset_spec = {
        "vis": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 256, 128),
        },
        "vis_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 256, 128),
        },
        "input_flags": {
            "axes": ["input", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "gain": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "sample_variance": {
            "axes": ["component", "freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (3, 64, 256, 128),
        },
        "nsample": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 256, 128),
        },
    }

    def __init__(self, ra=None, *args, **kwargs):

        # Set up axes passed ra langth
        if ra is not None:
            if isinstance(ra, int):
                ra = np.linspace(0.0, 360.0, ra, endpoint=False)
            kwargs["ra"] = ra

        super(SiderealStream, self).__init__(*args, **kwargs)

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def input_flags(self):
        return self.datasets["input_flags"]

    @property
    def ra(self):
        return self.index_map["ra"]

    @property
    def _mean(self):
        return self.datasets["vis"]


class SystemSensitivity(TODContainer):
    """A container for holding the total system sensitivity.

    This should be averaged/collapsed in the stack/prod axis
    to provide an overall summary of the system sensitivity.
    Two datasets are available: the measured noise from the
    visibility weights and the radiometric estimate of the
    noise from the autocorrelations.
    """

    _axes = ("freq", "pol")

    _dataset_spec = {
        "measured": {
            "axes": ["freq", "pol", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
        "radiometer": {
            "axes": ["freq", "pol", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
        "weight": {
            "axes": ["freq", "pol", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
        "frac_lost": {
            "axes": ["freq", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
    }

    @property
    def measured(self):
        return self.datasets["measured"]

    @property
    def radiometer(self):
        return self.datasets["radiometer"]

    @property
    def weight(self):
        return self.datasets["weight"]

    @property
    def frac_lost(self):
        return self.datasets["frac_lost"]

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]

    @property
    def pol(self):
        return self.index_map["pol"]


class RFIMask(TODContainer):
    """A container for holding RFI mask.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _axes = ("freq",)

    _dataset_spec = {
        "mask": {
            "axes": ["freq", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        return self.datasets["mask"]

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]


class TimeStream(VisContainer, TODContainer):
    """A container for holding a visibility dataset in time.

    This should look similar enough to the CHIME
    :class:`~ch_util.andata.CorrData` container that they can be used
    interchangably in most cases.
    """

    _dataset_spec = {
        "vis": {
            "axes": ["freq", "stack", "time"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 256, 128),
        },
        "vis_weight": {
            "axes": ["freq", "stack", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 256, 128),
        },
        "input_flags": {
            "axes": ["input", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "gain": {
            "axes": ["freq", "input", "time"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    def __init__(self, *args, **kwargs):

        super(TimeStream, self).__init__(*args, **kwargs)

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def input_flags(self):
        return self.datasets["input_flags"]


class GridBeam(ContainerBase):
    """ Generic container for representing the 2-d beam in spherical
        coordinates on a rectangular grid.
    """

    _axes = ("freq", "pol", "input", "theta", "phi")

    _dataset_spec = {
        "beam": {
            "axes": ["freq", "pol", "input", "theta", "phi"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "pol", "input", "theta", "phi"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "gain": {
            "axes": ["freq", "input"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    def __init__(self, coords="celestial", *args, **kwargs):

        self.attrs["coords"] = coords
        super(GridBeam, self).__init__(*args, **kwargs)

    @property
    def beam(self):
        return self.datasets["beam"]

    @property
    def weight(self):
        return self.datasets["weight"]

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def coords(self):
        return self.attrs["coords"]

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def pol(self):
        return self.index_map["pol"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def theta(self):
        return self.index_map["theta"]

    @property
    def phi(self):
        return self.index_map["phi"]


class TrackBeam(SampleVarianceContainer):
    """ Container for a sequence of beam samples at arbitrary locations
        on the sphere. The axis of the beam samples is 'pix', defined by
        the numpy.dtype [('theta', np.float32), ('phi', np.float32)].
    """

    _axes = ("freq", "pol", "input", "pix")

    _dataset_spec = {
        "beam": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (128, 2, 128, 128),
        },
        "weight": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (128, 2, 128, 128),
        },
        "sample_variance": {
            "axes": ["component", "freq", "pol", "input", "pix"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (3, 128, 2, 128, 128),
        },
        "nsample": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.uint8,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (128, 2, 128, 128),
        },
    }

    def __init__(
        self,
        theta=None,
        phi=None,
        coords="celestial",
        track_type="drift",
        *args,
        **kwargs
    ):

        if theta is not None and phi is not None:
            if len(theta) != len(phi):
                raise RuntimeError(
                    "theta and phi axes must have same length: "
                    "({:d} != {:d})".format(len(theta), len(phi))
                )
            else:
                pix = np.zeros(
                    len(theta), dtype=[("theta", np.float32), ("phi", np.float32)]
                )
                pix["theta"] = theta
                pix["phi"] = phi
                kwargs["pix"] = pix
        elif (theta is None) != (phi is None):
            raise RuntimeError("Both theta and phi coordinates must be specified.")

        super(TrackBeam, self).__init__(*args, **kwargs)

        self.attrs["coords"] = coords
        self.attrs["track_type"] = track_type

    @property
    def beam(self):
        return self.datasets["beam"]

    @property
    def weight(self):
        return self.datasets["weight"]

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def coords(self):
        return self.attrs["coords"]

    @property
    def track_type(self):
        return self.attrs["track_type"]

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def pol(self):
        return self.index_map["pol"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def pix(self):
        return self.index_map["pix"]

    @property
    def _mean(self):
        return self.datasets["beam"]


class MModes(VisContainer):
    """Parallel container for holding m-mode data.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    weight : mpidataset.MPIArray
        Array of weights for each point.
    """

    _axes = ("m", "msign")

    _dataset_spec = {
        "vis": {
            "axes": ["m", "msign", "freq", "stack"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
        "vis_weight": {
            "axes": ["m", "msign", "freq", "stack"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
    }

    def __init__(self, mmax=None, *args, **kwargs):

        # Set up axes from passed arguments
        if mmax is not None:
            kwargs["m"] = mmax + 1

        # Ensure the sign axis is set correctly
        kwargs["msign"] = np.array(["+", "-"])

        super(MModes, self).__init__(*args, **kwargs)


class SVDModes(ContainerBase):
    """Parallel container for holding SVD m-mode data.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    weight : mpidataset.MPIArray
        Array of weights for each point.
    """

    _axes = ("m", "mode")

    _dataset_spec = {
        "vis": {
            "axes": ["m", "mode"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
        "vis_weight": {
            "axes": ["m", "mode"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
        "nmode": {
            "axes": ["m"],
            "dtype": np.int32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
    }

    @property
    def vis(self):
        return self.datasets["vis"]

    @property
    def nmode(self):
        return self.datasets["nmode"]

    @property
    def weight(self):
        return self.datasets["vis_weight"]

    def __init__(self, mmax=None, *args, **kwargs):

        # Set up axes from passed arguments
        if mmax is not None:
            kwargs["m"] = mmax + 1

        super(SVDModes, self).__init__(*args, **kwargs)


class KLModes(SVDModes):
    """Parallel container for holding KL filtered m-mode data.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    weight : mpidataset.MPIArray
        Array of weights for each point.
    """

    pass


class CommonModeGainData(TODContainer):
    """Parallel container for holding gain data common to all inputs.
    """

    _axes = ("freq",)

    _dataset_spec = {
        "gain": {
            "axes": ["freq", "time"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "time"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def weight(self):
        try:
            return self.datasets["weight"]
        except KeyError:
            return None

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]


class CommonModeSiderealGainData(ContainerBase):
    """Parallel container for holding sidereal gain data common to all inputs.
    """

    _axes = ("freq", "ra")

    _dataset_spec = {
        "gain": {
            "axes": ["freq", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def weight(self):
        try:
            return self.datasets["weight"]
        except KeyError:
            return None

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def ra(self):
        return self.index_map["ra"]


class GainData(TODContainer):
    """Parallel container for holding gain data.
    """

    _axes = ("freq", "input")

    _dataset_spec = {
        "gain": {
            "axes": ["freq", "input", "time"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "time"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def weight(self):
        try:
            return self.datasets["weight"]
        except KeyError:
            return None

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]

    @property
    def input(self):
        return self.index_map["input"]


class SiderealGainData(ContainerBase):
    """Parallel container for holding sidereal gain data.
    """

    _axes = ("freq", "input", "ra")

    _dataset_spec = {
        "gain": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def weight(self):
        try:
            return self.datasets["weight"]
        except KeyError:
            return None

    @property
    def freq(self):
        return self.index_map["freq"]

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def ra(self):
        return self.index_map["ra"]


class StaticGainData(ContainerBase):
    """Parallel container for holding static gain data (i.e. non time varying).
    """

    _axes = ("freq", "input")

    _dataset_spec = {
        "gain": {
            "axes": ["freq", "input"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "input"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def gain(self):
        return self.datasets["gain"]

    @property
    def weight(self):
        return self.datasets["weight"]

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]

    @property
    def input(self):
        return self.index_map["input"]


class DelaySpectrum(ContainerBase):
    """Container for a delay spectrum.
    """

    _axes = ("baseline", "delay")

    _dataset_spec = {
        "spectrum": {
            "axes": ["baseline", "delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        }
    }

    @property
    def spectrum(self):
        return self.datasets["spectrum"]


class Powerspectrum2D(ContainerBase):
    """Container for a 2D cartesian power spectrum.

    Generally you should set the standard attributes `z_start` and `z_end` with
    the redshift range included in the power spectrum estimate, and the `type`
    attribute with a description of the estimator type. Suggested valued for
    `type` are:

    `unwindowed`
        The standard unbiased quadratic estimator.

    `minimum_variance`
        The minimum variance, but highly correlated, estimator. Just a rescaled
        version of the q-estimator.

    `uncorrelated`
        The uncorrelated estimator using the root of the Fisher matrix.

    Parameters
    ----------
    kpar_edges, kperp_edges : np.ndarray
        Array of the power spectrum bin boundaries.
    """

    _axes = ("kperp", "kpar")

    _dataset_spec = {
        "powerspectrum": {
            "axes": ["kperp", "kpar"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "C_inv": {
            "axes": ["kperp", "kpar", "kperp", "kpar"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    def __init__(self, kperp_edges=None, kpar_edges=None, *args, **kwargs):

        # Construct the kperp axis from the bin edges
        if kperp_edges is not None:
            centre = 0.5 * (kperp_edges[1:] + kperp_edges[:-1])
            width = kperp_edges[1:] - kperp_edges[:-1]

            kwargs["kperp"] = np.rec.fromarrays(
                [centre, width], names=["centre", "width"]
            ).view(np.ndarray)

        # Construct the kpar axis from the bin edges
        if kpar_edges is not None:
            centre = 0.5 * (kpar_edges[1:] + kpar_edges[:-1])
            width = kpar_edges[1:] - kpar_edges[:-1]

            kwargs["kpar"] = np.rec.fromarrays(
                [centre, width], names=["centre", "width"]
            ).view(np.ndarray)

        super(Powerspectrum2D, self).__init__(*args, **kwargs)

    @property
    def powerspectrum(self):
        return self.datasets["powerspectrum"]

    @property
    def C_inv(self):
        return self.datasets["C_inv"]


class SVDSpectrum(ContainerBase):
    """Container for an m-mode SVD spectrum.
    """

    _axes = ("m", "singularvalue")

    _dataset_spec = {
        "spectrum": {
            "axes": ["m", "singularvalue"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        }
    }

    @property
    def spectrum(self):
        return self.datasets["spectrum"]


class FrequencyStack(ContainerBase):
    """Container for a frequency stack.

    In general used to hold the product of `draco.analysis.SourceStack`
    The stacked signal of frequency slices of the data in the direction
    of sources of interest.
    """

    _axes = ("freq",)

    _dataset_spec = {
        "stack": {
            "axes": ["freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def stack(self):
        return self.datasets["stack"]

    @property
    def weight(self):
        return self.datasets["weight"]

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]


class SourceCatalog(TableBase):
    """A basic container for holding astronomical source catalogs.

    Notes
    -----
    The `ra` and `dec` coordinates should be ICRS.
    """

    _table_spec = {
        "position": {
            "columns": [["ra", np.float64], ["dec", np.float64]],
            "axis": "object_id",
        }
    }


class SpectroscopicCatalog(SourceCatalog):
    """A container for spectroscopic catalogs.
    """

    _table_spec = {
        "redshift": {
            "columns": [["z", np.float64], ["z_error", np.float64]],
            "axis": "object_id",
        }
    }


class FormedBeam(ContainerBase):
    """Container for formed beams.
    """

    _axes = ("object_id", "pol", "freq")

    _dataset_spec = {
        "beam": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "position": {
            "axes": ["object_id"],
            "dtype": np.dtype([("ra", np.float64), ("dec", np.float64)]),
            "initialise": True,
            "distributed": False,
        },
        "redshift": {
            "axes": ["object_id"],
            "dtype": np.dtype([("z", np.float64), ("z_error", np.float64)]),
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def beam(self):
        return self.datasets["beam"]

    @property
    def weight(self):
        return self.datasets["weight"]

    @property
    def freq(self):
        return self.index_map["freq"]["centre"]

    @property
    def frequency(self):
        return self.index_map["freq"]

    @property
    def id(self):
        return self.index_map["object_id"]

    @property
    def pol(self):
        return self.index_map["pol"]


class FormedBeamHA(FormedBeam):
    """Container for formed beams.
       These have not been collapsed in the hour angle (HA) axis
    """

    _axes = ("object_id", "pol", "freq", "ha")

    _dataset_spec = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "object_ha": {
            "axes": ["object_id", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def ha(self):
        return self.datasets["object_ha"]


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
        raise RuntimeError(
            "I don't know how to deal with data type %s" % obj.__class__.__name__
        )


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

"""Distributed containers for holding various types of analysis data.

Containers
==========
- :py:class:`Map`
- :py:class:`SiderealStream`
- :py:class:`SystemSensitivity`
- :py:class:`RFIMask`
- :py:class:`TimeStream`
- :py:class:`GridBeam`
- :py:class:`TrackBeam`
- :py:class:`MModes`
- :py:class:`SVDModes`
- :py:class:`KLModes`
- :py:class:`VisGridStream`
- :py:class:`HybridVisStream`
- :py:class:`HybridVisMModes`
- :py:class:`RingMap`
- :py:class:`RingMapMask`
- :py:class:`RingMapTaper`
- :py:class:`CommonModeGainData`
- :py:class:`CommonModeSiderealGainData`
- :py:class:`GainData`
- :py:class:`SiderealGainData`
- :py:class:`StaticGainData`
- :py:class:`DelayCutoff`
- :py:class:`DelaySpectrum`
- :py:class:`Powerspectrum2D`
- :py:class:`SVDSpectrum`
- :py:class:`FrequencyStack`
- :py:class:`FrequencyStackByPol`
- :py:class:`MockFrequencyStack`
- :py:class:`MockFrequencyStackByPol`
- :py:class:`SourceCatalog`
- :py:class:`SpectroscopicCatalog`
- :py:class:`FormedBeam`
- :py:class:`FormedBeamHA`
- :py:class:`FormedBeamMask`
- :py:class:`FormedBeamHAMask`
- :py:class:`LocalizedRFIMask`
- :py:class:`LocalizedSiderealRFIMask`

Container Base Classes
----------------------
- :py:class:`ContainerBase`
- :py:class:`TableBase`
- :py:class:`TODContainer`
- :py:class:`VisContainer`
- :py:class:`SampleVarianceContainer`
- :py:class:`FreqContainer`
- :py:class:`SiderealContainer`
- :py:class:`MContainer`

Helper Routines
---------------
These routines are designed to be replaced by other packages trying to insert
their own custom container types.

- :py:meth:`empty_like`
- :py:meth:`empty_timestream`
"""

import inspect
from typing import ClassVar

import numpy as np
from caput import memh5, mpiarray, tod
from cora.util.cosmology import Cosmology

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
    `_dataset_spec`. See the :ref:`Notes <containerbase_notes>` section for details.

    Parameters
    ----------
    data_group : `memh5.MemDiskGroup`
        A container to pass through for making a shallow copy. This is used by
        routine like `caput.tod.concatenate` and generally shouldn't be used
        directly. Either a keyword argument, or the first positional argument.
    axes_from : `memh5.BasicCont`, optional
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : `memh5.BasicCont`, optional
        Another container to copy attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    dsets_from : `memh5.BasicCont`, optional
        A container to copy datasets from. Any dataset which an axis whose definition
        has been explicitly set (i.e. does not come from `axes_from`) will not be
        copied.
    copy_from : `memh5.BasicCont`, optional
        Set `axes_from`, `attrs_from` and `dsets_from` to this instance if they are
        not set explicitly.
    skip_datasets : bool, optional
        Skip creating datasets. They must all be added manually with
        `.add_dataset` regardless of the entry in `.dataset_spec`. Default is False.
    distributed : bool, optional
        Should this be a distributed container. Defaults to True.
    comm : mpi4py.MPI.Comm, optional
        The MPI communicator to distribute over. Use COMM_WORLD if not set.
    allow_chunked : bool, optional
        Allow the datasets to be chunked. Default is True.

    kwargs : dict
        Should contain entries for all other axes.

    Notes
    -----
    .. _containerbase_notes:

    Inheritance from other `ContainerBase` subclasses should work as expected,
    with datasets defined in super classes appearing as expected, and being
    overridden where they are redefined in the derived class.

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

    _dataset_spec: ClassVar = {}

    convert_attribute_strings = True
    convert_dataset_strings = True
    allow_chunked = True

    def __init__(self, *args, **kwargs):
        # Arguments for pulling in definitions from other containers
        copy_from = kwargs.pop("copy_from", None)
        axes_from = kwargs.pop("axes_from", copy_from)
        attrs_from = kwargs.pop("attrs_from", copy_from)
        dsets_from = kwargs.pop("dsets_from", copy_from)

        # MPI distribution arguments
        dist = kwargs.pop("distributed", True)
        comm = kwargs.pop("comm", None)

        # Extract misc options
        self.allow_chunked = kwargs.pop("allow_chunked", self.allow_chunked)
        skip_datasets = kwargs.pop("skip_datasets", False)

        # Handle the data_group argument. We need to identify if the argument
        # was actually supplied or not (both as a positional or keyword
        # argument), and infer what its value should be, or None if not
        # provided
        if args and "data_group" in kwargs:
            raise ValueError(
                "Received conflicting definitions of `data_group`, as both the first "
                "positional and a keyword argument."
            )
        has_data_group = args or ("data_group" in kwargs)
        data_group = args[0] if args else kwargs.get("data_group", None)

        # Run base initialiser, and exit early if data_group was provided
        super().__init__(data_group=data_group, distributed=dist, comm=comm)

        # If data_group was provided we need to exit early to behave like
        # memh5.MemDiskGroup would have. In this case we're probably trying to
        # create a bare container, or a shallow clone, so don't initialise any
        # datasets. This behaviour is needed to support tod.concatenate
        if has_data_group:
            return

        # Create axis entries
        for axis in self.axes:
            axis_map = None

            # Check if axis is specified in initialiser
            if axis in kwargs:
                axis_map = kwargs[axis]
                copy_axis_attrs = False

                # If axis is an integer, turn into an arange as a default definition
                if isinstance(axis_map, int | np.integer):
                    axis_map = np.arange(axis_map)

            # If no valid map provided in arguments copy from another object if set
            elif axes_from is not None:
                axis_map = axes_from.index_map.get(axis, None)
                copy_axis_attrs = True

            # Set the index_map[axis] if we have a definition, otherwise throw an error
            if axis_map is None:
                raise RuntimeError(f"No definition of axis {axis} supplied.")

            self.create_index_map(axis, axis_map)

            if copy_axis_attrs:
                # Copy over axis attributes if we're copying the axis from another dataset
                memh5.copyattrs(axes_from.index_attrs[axis], self.index_attrs[axis])

        # Iterate over datasets and initialise any that specify it
        if not skip_datasets:
            for name, spec in self.dataset_spec.items():
                if spec.get("initialise"):
                    self.add_dataset(name)

        # Copy over datasets that have compatible axes
        if dsets_from is not None:
            # Get the list of axes names that have been overriden
            changed_axes = {ax for ax in self.axes if ax in kwargs}

            for name in self.dataset_spec.keys():
                if name not in dsets_from:
                    continue

                source_dset = dsets_from[name]
                source_axes = set(source_dset.attrs["axis"])

                # Check if any of the axes of this dataset have been changed, if that's
                # the case then we can't copy the data over
                if not source_axes.isdisjoint(changed_axes):
                    continue

                # The dataset may not have been initialised by default, if not, create
                # it
                if name not in self:
                    self.add_dataset(name)

                self[name][:] = source_dset[:]

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
        # Normalise name
        name = name.strip("/")

        # Dataset must be specified
        if name not in self.dataset_spec:
            raise RuntimeError(f"Dataset {name} not known.")

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
                    raise RuntimeError(f"Axis {axis} not defined in index_map")
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

    def _ensure_chunked(self):
        """Ensure datasets that have chunk/compression specs are chunked.

        For every dataset, check if chunks and compression are set, and
        if not set them to dataset_spec values.
        """
        for dset in self.dataset_spec:
            if dset not in self:
                continue
            if "chunks" in self.dataset_spec[dset] and self[dset].chunks is None:
                # ensure chunks aren't larger than dataset shape
                chunks = ()
                for i, l in enumerate(self[dset].shape):
                    chunks += (min(self.dataset_spec[dset]["chunks"][i], l),)
                self._data._storage_root[dset].chunks = chunks
            if (
                "compression" in self.dataset_spec[dset]
                and self[dset].compression is None
            ):
                self._data._storage_root[dset].compression = self.dataset_spec[dset][
                    "compression"
                ]
            if (
                "compression_opts" in self.dataset_spec[dset]
                and self[dset].compression_opts is None
            ):
                self._data._storage_root[dset].compression_opts = self.dataset_spec[
                    dset
                ]["compression_opts"]

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

    @classmethod
    def _class_dataset_spec(cls):
        """Get the inherited set of dataset spec entries."""
        ddict = {}

        # Iterate over the reversed MRO and look for _dataset_spec attributes
        # which get added to a temporary dict. We go over the reversed MRO so
        # that the `ddict.update` overrides datasets in base classes.`
        for cls in inspect.getmro(cls)[::-1]:
            try:
                # NOTE: this is a little ugly as the following line will drop
                # down to base classes if dataset_spec isn't present, and thus
                # try and `update` with the same values again.
                ddict.update(cls._dataset_spec)
            except AttributeError:
                pass

        # Ensure that the dataset_spec is the same order on all ranks
        return {k: ddict[k] for k in sorted(ddict)}

    @property
    def dataset_spec(self):
        """Return a copy of the fully resolved dataset specifiction as a dictionary."""
        ddict = self.__class__._class_dataset_spec()

        # Add in any _dataset_spec found on the instance
        ddict.update(self.__dict__.get("_dataset_spec", {}))

        # Ensure that the dataset_spec is the same order on all ranks
        return {k: ddict[k] for k in sorted(ddict)}

    @classmethod
    def _class_axes(cls):
        """Get the set of axes defined by the container and it's base classes."""
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
        """The set of axes for this container including any defined on the instance."""
        axes = set(self._class_axes())

        # Add in any axes found on the instance (this is needed to support the table
        # classes where the axes get added at run time)
        axes |= set(self.__dict__.get("_axes", []))

        # This must be the same order on all ranks, so we need to explicitly sort to
        # get around the hash randomization
        return tuple(sorted(axes))

    @classmethod
    def _make_selections(cls, sel_args):
        """Match down-selection arguments to axes of datasets.

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
                raise RuntimeError(f"No '{axis}' axis found to select from.")

        # Build selections dict
        selections = {}
        for name, dataset in cls._class_dataset_spec().items():
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


class TableBase(ContainerBase):
    """A base class for containers holding tables of data.

    Similar to the `ContainerBase` class, the container is defined through a
    dictionary given as a `_table_spec` class attribute. The container may also
    hold generic datasets by specifying `_dataset_spec` as with `ContainerBase`.
    See :ref:`Notes <tablebase_notes>` for details.

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
    .. _tablebase_notes:

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
                    ['mask', bool]
                ],
                'axis': 'quasar_id'
            }
        }
    """

    _table_spec: ClassVar = {}

    def __init__(self, *args, **kwargs):
        # Get the dataset specifiction for this class (not any base classes), or
        # an empty dictionary if it does not exist. Do the same for the axes entry..
        dspec = self.__class__.__dict__.get("_dataset_spec", {})
        axes = self.__class__.__dict__.get("_axes", ())

        # Iterate over all table_spec entries and construct dataset specifications for
        # them.
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

        super().__init__(*args, **kwargs)

    def _create_dtype(self, columns):
        """Take a dictionary of columns and turn into the appropriate compound data type."""
        dt = []
        for ci, (name, dtype) in enumerate(columns):
            if not isinstance(name, str):
                raise ValueError(f"Column {ci:d} is invalid")
            dt.append((name, dtype))

        return dt

    @property
    def table_spec(self):
        """Return a copy of the fully resolved table specifiction as a dictionary."""
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


class DataWeightContainer(ContainerBase):
    """A base class for containers with generic data/weight datasets.

    This is meant such that tasks can operate generically over containers with this
    common structure. The data and weight datasets are expected to have the same size,
    though this isn't checked. Subclasses must define `_data_dset_name` and
    `_weight_dset_name`.
    """

    _data_dset_name: str | None = None
    _weight_dset_name: str | None = None

    @property
    def data(self) -> memh5.MemDataset:
        """The main dataset."""
        if self._data_dset_name is None:
            raise RuntimeError(f"Type {type(self)} has not defined `_data_dset_name`.")

        dset = self[self._data_dset_name]

        if not isinstance(dset, memh5.MemDataset):
            raise TypeError(f"/{self._data_dset_name} is not a dataset")

        return dset

    @property
    def weight(self) -> memh5.MemDataset:
        """The weights for each data point."""
        if not self._weight_dset_name:
            raise RuntimeError(
                f"Type {type(self)} has not defined `_weight_dset_name`."
            )

        dset = self[self._weight_dset_name]

        if not isinstance(dset, memh5.MemDataset):
            raise TypeError(f"/{self._weight_dset_name} is not a dataset")

        return dset


class VisBase(DataWeightContainer):
    """A very basic class for visibility data.

    For better support for input/prod/stack structured data use `VisContainer`.
    """

    _data_dset_name = "vis"
    _weight_dset_name = "vis_weight"

    @property
    def vis(self):
        """The visibility like dataset."""
        return self.datasets["vis"]


class VisContainer(VisBase):
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

    _axes = ("input", "prod", "stack")

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
        super().__init__(*args, **kwargs)

        # `axes_from` can be provided via the `copy_from` argument, so
        # need to update kwargs to account
        if ("axes_from" not in kwargs) and ("copy_from" in kwargs):
            kwargs["axes_from"] = kwargs.pop("copy_from")

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

        super().__init__(*args, **kwargs)

    @property
    def component(self):
        """Get the component axis."""
        return self.index_map["component"]

    @property
    def sample_variance(self):
        """Convenience access to the sample variance dataset.

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
        """Get the nsample dataset if it exists."""
        if "nsample" in self.datasets:
            return self.datasets["nsample"]

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


class FreqContainer(ContainerBase):
    """A pipeline container for data with a frequency axis.

    This works like a normal :class:`ContainerBase` container, but already has a freq
    axis defined, and specific properties for dealing with frequencies.
    """

    _axes = ("freq",)

    @property
    def freq(self):
        """The physical frequency associated with each entry of the time axis.

        By convention this property should return the frequency in MHz at the centre
        of each of frequency channel.
        """
        try:
            return self.index_map["freq"][:]["centre"]
        # Need to check for both types as different numpy versions return
        # different exceptions.
        except (IndexError, ValueError):
            return self.index_map["freq"][:]


class SiderealContainer(ContainerBase):
    """A pipeline container for data with an RA axis.

    This works like a normal :class:`ContainerBase` container, but already has an RA
    axis defined, and specific properties for dealing with this axis.

    Note that Right Ascension is a fairly ambiguous term. What is typically meant
    here is the Local Stellar Angle, which is the transiting RA in CIRS coordinates.
    This is similar to J2000/ICRS with the minimal amount of coordinate rotation to
    account for the polar axis precession.

    Parameters
    ----------
    ra : array or int, optional
        Either the explicit locations of samples of the RA axis, or if passed an
        integer interpret this as a number of samples dividing the full sidereal day
        and create an axis accordingly.
    """

    _axes = ("ra",)

    def __init__(self, ra=None, *args, **kwargs):
        # Allow the passing of a number of samples for the RA axis
        if ra is not None:
            if isinstance(ra, int):
                ra = np.linspace(0.0, 360.0, ra, endpoint=False)
            kwargs["ra"] = ra

        super().__init__(*args, **kwargs)

    @property
    def ra(self):
        """The RA in degrees associated with each sample of the RA axis."""
        return self.index_map["ra"][:]


class MContainer(ContainerBase):
    """Container for holding m-mode type data.

    Note this container will have an `msign` axis even though not all m-mode based
    data needs one. As always this is not an issue, datasets that don't need it are
    not required to list it in their `axes` list.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.
    oddra : bool, optional
        Does this MContainer come from an underlying odd number of RA points. This
        determines if the largest negative m is filled or not (it is for odd=True, not
        for odd=False). Default is odd=False.
    """

    _axes = ("m", "msign")

    def __init__(
        self, mmax: int | None = None, oddra: bool | None = None, *args, **kwargs
    ):
        # Set up axes from passed arguments
        if mmax is not None:
            kwargs["m"] = mmax + 1

        # Ensure the sign axis is set correctly
        kwargs["msign"] = np.array(["+", "-"])

        super().__init__(*args, **kwargs)

        # Set oddra, prioritising an explicit keyword argument over anything else
        if oddra is not None:
            self.attrs["oddra"] = oddra
        elif "oddra" not in self.attrs:
            self.attrs["oddra"] = False

    @property
    def mmax(self) -> int:
        """The maximum m stored."""
        return int(self.index_map["m"][-1])

    @property
    def oddra(self) -> bool:
        """Whether this represents an odd or even number of RA points."""
        return self.attrs["oddra"]


class HealpixContainer(ContainerBase):
    """Base class container for holding Healpix map data.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    """

    _axes = ("pixel",)

    def __init__(self, nside=None, *args, **kwargs):
        # Set up axes from passed arguments
        if nside is not None:
            kwargs["pixel"] = 12 * nside**2

        super().__init__(*args, **kwargs)

    @property
    def nside(self):
        """Get the nside of the map."""
        return int((len(self.index_map["pixel"]) // 12) ** 0.5)


class Map(FreqContainer, HealpixContainer):
    """Container for holding multi-frequency sky maps.

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

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "map": {
            "axes": ["freq", "pol", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    def __init__(self, polarisation=True, *args, **kwargs):
        # Set up axes from passed arguments
        kwargs["pol"] = (
            np.array(["I", "Q", "U", "V"]) if polarisation else np.array(["I"])
        )

        super().__init__(*args, **kwargs)

    @property
    def map(self):
        """Get the map dataset."""
        return self.datasets["map"]


class SiderealStream(
    FreqContainer, VisContainer, SiderealContainer, SampleVarianceContainer
):
    """A container for holding a visibility dataset in sidereal time.

    Parameters
    ----------
    ra : int
        The number of points to divide the RA axis up into.
    """

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
            "truncate": {
                "weight_dataset": "vis_weight",
            },
        },
        "vis_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
            "truncate": True,
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
            "chunks": (1, 32, 512, 2048),
            "truncate": True,
        },
        "nsample": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
        },
        "effective_ra": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
            "truncate": True,
        },
    }

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def input_flags(self):
        """Get the input_flags dataset."""
        return self.datasets["input_flags"]

    @property
    def _mean(self):
        """Get the vis dataset."""
        return self.datasets["vis"]

    @property
    def effective_ra(self):
        """Get the effective_ra dataset if it exists, None otherwise."""
        if "effective_ra" in self.datasets:
            return self.datasets["effective_ra"]

        raise KeyError("Dataset 'effective_ra' not initialised.")


class SystemSensitivity(FreqContainer, TODContainer):
    """A container for holding the total system sensitivity.

    This should be averaged/collapsed in the stack/prod axis
    to provide an overall summary of the system sensitivity.
    Two datasets are available: the measured noise from the
    visibility weights and the radiometric estimate of the
    noise from the autocorrelations.
    """

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
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
        """Get the measured noise dataset."""
        return self.datasets["measured"]

    @property
    def radiometer(self):
        """Get the radiometer estimate dataset."""
        return self.datasets["radiometer"]

    @property
    def weight(self):
        """Get the weight dataset."""
        return self.datasets["weight"]

    @property
    def frac_lost(self):
        """Get the frac_lost dataset."""
        return self.datasets["frac_lost"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]


class RFIMask(FreqContainer, TODContainer):
    """A container for holding an RFI mask for a timestream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _dataset_spec: ClassVar = {
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
        """Get the mask dataset."""
        return self.datasets["mask"]


class RFIMaskByPol(RFIMask):
    """A container for holding a polarisation-dependent RFI mask as a function of time.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["pol", "freq", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def pol(self):
        """Get the pol index map."""
        return self.index_map["pol"]


class SiderealRFIMask(FreqContainer, SiderealContainer):
    """A container for holding an RFI mask for a sidereal stream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]


class SiderealRFIMaskByPol(SiderealRFIMask):
    """A container for holding a polarisation-dependent RFI mask as a function of RA.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["pol", "freq", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def pol(self):
        """Get the pol index map."""
        return self.index_map["pol"]


class BaselineMask(FreqContainer, TODContainer):
    """A container for holding a baseline-dependent mask for a timestream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.

    Unlike RFIMask, this is distributed by default.
    """

    _axes = ("stack",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "stack", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def stack(self):
        """The stack definition as an index (and conjugation) of a member product."""
        return self.index_map["stack"]


class SiderealBaselineMask(FreqContainer, SiderealContainer):
    """A container for holding a baseline-dependent mask for a sidereal stream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.

    Unlike SiderealRFIMask, this is distributed by default.
    """

    _axes = ("stack",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "stack", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def stack(self):
        """The stack definition as an index (and conjugation) of a member product."""
        return self.index_map["stack"]


class TimeStream(FreqContainer, VisContainer, TODContainer):
    """A container for holding a visibility dataset in time.

    This should look similar enough to the CHIME
    :class:`~ch_util.andata.CorrData` container that they can be used
    interchangably in most cases.
    """

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["freq", "stack", "time"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 128, 128),
            "truncate": {
                "weight_dataset": "vis_weight",
            },
        },
        "vis_weight": {
            "axes": ["freq", "stack", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 128, 128),
            "truncate": True,
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

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def input_flags(self):
        """Get the input_flags dataset."""
        return self.datasets["input_flags"]


class GridBeam(FreqContainer, DataWeightContainer):
    """Generic container for representing a 2D beam on a rectangular grid."""

    _axes = ("pol", "input", "theta", "phi")

    _dataset_spec: ClassVar = {
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
        "quality": {
            "axes": ["freq", "pol", "input", "theta", "phi"],
            "dtype": np.uint8,
            "initialise": False,
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

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    def __init__(self, coords="celestial", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs["coords"] = coords

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def quality(self):
        """Get the quality dataset."""
        return self.datasets["quality"]

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def coords(self):
        """Get the coordinates attribute."""
        return self.attrs["coords"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]

    @property
    def theta(self):
        """Get the theta axis."""
        return self.index_map["theta"]

    @property
    def phi(self):
        """Get the phi axis."""
        return self.index_map["phi"]


class HEALPixBeam(FreqContainer, HealpixContainer, DataWeightContainer):
    """Container for representing the spherical 2-d beam in a HEALPix grid.

    Parameters
    ----------
    ordering : {"nested", "ring"}
        The HEALPix ordering scheme used for the beam map.
    coords : {"celestial", "galactic", "telescope"}
        The coordinate system that the beam map is defined on.
    """

    _axes = ("pol", "input")

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["freq", "pol", "input", "pixel"],
            "dtype": [("Et", np.complex64), ("Ep", np.complex64)],
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "pol", "input", "pixel"],
            "dtype": [("Et", np.float32), ("Ep", np.float32)],
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    def __init__(self, coords="unknown", ordering="unknown", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs["coords"] = coords
        self.attrs["ordering"] = ordering

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def ordering(self):
        """Get the ordering attribute."""
        return self.attrs["ordering"]

    @property
    def coords(self):
        """Get the coordinate attribute."""
        return self.attrs["coords"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]

    @property
    def nside(self):
        """Get the nsides of the map."""
        return int(np.sqrt(len(self.index_map["pixel"]) / 12))


class TrackBeam(FreqContainer, SampleVarianceContainer, DataWeightContainer):
    """Container for a sequence of beam samples at arbitrary locations on the sphere.

    The axis of the beam samples is 'pix', defined by the numpy.dtype
    [('theta', np.float32), ('phi', np.float32)].
    """

    _axes = ("pol", "input", "pix")

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 2, 64, 128),
            "truncate": {
                "weight_dataset": "weight",
            },
        },
        "weight": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 2, 64, 128),
            "truncate": True,
        },
        "sample_variance": {
            "axes": ["component", "freq", "pol", "input", "pix"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (3, 64, 2, 64, 128),
            "truncate": True,
        },
        "nsample": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.uint8,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 2, 64, 128),
        },
    }

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    def __init__(
        self,
        theta=None,
        phi=None,
        coords="celestial",
        track_type="drift",
        *args,
        **kwargs,
    ):
        if theta is not None and phi is not None:
            if len(theta) != len(phi):
                raise RuntimeError(
                    "theta and phi axes must have same length: "
                    f"({len(theta)} != {len(phi)})"
                )

            pix = np.zeros(
                len(theta), dtype=[("theta", np.float32), ("phi", np.float32)]
            )
            pix["theta"] = theta
            pix["phi"] = phi
            kwargs["pix"] = pix
        elif (theta is None) != (phi is None):
            raise RuntimeError("Both theta and phi coordinates must be specified.")

        super().__init__(*args, **kwargs)

        self.attrs["coords"] = coords
        self.attrs["track_type"] = track_type

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def coords(self):
        """Get the coordinates attribute."""
        return self.attrs["coords"]

    @property
    def track_type(self):
        """Get the track type attribute."""
        return self.attrs["track_type"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]

    @property
    def pix(self):
        """Get the pix axis."""
        return self.index_map["pix"]

    @property
    def _mean(self):
        """Get the beam dataset."""
        return self.datasets["beam"]


class MModes(FreqContainer, VisContainer, MContainer):
    """Parallel container for holding m-mode data.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    weight : mpidataset.MPIArray
        Array of weights for each point.
    """

    _dataset_spec: ClassVar = {
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


class SVDModes(MContainer, VisBase):
    """Parallel container for holding SVD m-mode data.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.
    """

    _axes = ("mode",)

    _dataset_spec: ClassVar = {
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
    def nmode(self):
        """Get the nmode dataset."""
        return self.datasets["nmode"]


class KLModes(SVDModes):
    """Parallel container for holding KL filtered m-mode data.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.
    """

    pass


class VisGridStream(FreqContainer, SiderealContainer, VisBase):
    """Visibilities gridded into a 2D array.

    Only makes sense for an array which is a cartesian grid.
    """

    _axes = ("pol", "ew", "ns")

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 1, 64, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": {
                "weight_dataset": "weight",
            },
        },
        "vis_weight": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 1, 64, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "redundancy": {
            "axes": ["pol", "ew", "ns", "ra"],
            "dtype": np.int32,
            "initialise": False,
            "distributed": False,
            "chunks": (1, 64, 1, 64, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def redundancy(self):
        """Get the redundancy dataset."""
        if "redundancy" in self.datasets:
            return self.datasets["redundancy"]

        raise KeyError("Dataset 'redundancy' not initialised.")


class FilterFreqContainer(ContainerBase):
    """Base container for data that has undergone filtering along the frequency axis.

    This container behaves like a standard `ContainerBase`, but is tailored for
    datasets where a frequency-domain filter has been applied. It defines a
    `freq_sum` axis by default and provides additional logic to manage mutually
    exclusive filter and frequency covariance datasets.

    Expected Datasets:
    ------------------
    - `filter` or `complex_filter`
      Real or complex filter applied in the frequency domain.
    - `freq_cov` or `complex_freq_cov`:
      Real or complex covariance matrix in the frequency domain.
    """

    _axes = ("freq_sum",)

    def __init__(self, *args, **kwargs):

        for ax in ["freq_sum"]:

            if ax not in kwargs:
                if "axes_from" in kwargs and ax in kwargs["axes_from"].index_map:
                    kwargs[ax] = kwargs["axes_from"].index_map[ax]
                elif "freq" in kwargs:
                    kwargs[ax] = kwargs["freq"]
                elif "axes_from" in kwargs and "freq" in kwargs["axes_from"].index_map:
                    kwargs[ax] = kwargs["axes_from"].index_map["freq"]
            else:
                raise RuntimeError(f"Must provide {ax} or freq axis.")

        super().__init__(*args, **kwargs)

    def add_dataset(self, name):
        """Ensure that multiple filters and covariances are not created."""
        if name == "filter" and "complex_filter" in self.datasets:
            raise RuntimeError(
                "Requesting creation of real-valued filter but "
                "complex filter already exists."
            )
        if name == "complex_filter" and "filter" in self.datasets:
            raise RuntimeError(
                "Requesting creation of complex-valued filter but "
                "real filter already exists."
            )
        if name == "freq_cov" and "complex_freq_cov" in self.datasets:
            raise RuntimeError(
                "Requesting creation of real-valued freq_cov but "
                "complex_freq_cov already exists."
            )
        if name == "complex_freq_cov" and "freq_cov" in self.datasets:
            raise RuntimeError(
                "Requesting creation of complex_freq_cov but "
                "real-valued freq_cov already exists."
            )
        return super().add_dataset(name)

    @property
    def filter(self):
        """Return the filter dataset, if available."""
        if "filter" in self.datasets:
            return self.datasets["filter"]
        if "complex_filter" in self.datasets:
            return self.datasets["complex_filter"]

        raise KeyError("Dataset 'filter' not initialised.")

    @property
    def freq_cov(self):
        """Return the freq_cov dataset, if available."""
        if "freq_cov" in self.datasets:
            return self.datasets["freq_cov"]
        if "complex_freq_cov" in self.datasets:
            return self.datasets["complex_freq_cov"]

        raise KeyError("Dataset 'freq_cov' not initialised.")

    @property
    def swapped_freq_cov_axis(self):
        """Return the axis names of the freq_cov dataset with freq <--> freq_sum.

        This is useful for broadcasting the weight dataset when propagating the
        covariance matrix through linear operators.
        """
        swap = {"freq": "freq_sum", "freq_sum": "freq"}
        return np.array([swap.get(ax, ax) for ax in self.freq_cov.attrs["axis"]])


class HybridVisStream(FilterFreqContainer, FreqContainer, SiderealContainer, VisBase):
    """Visibilities beamformed only in the NS direction.

    This container has visibilities beam formed only in the NS direction to give a
    grid in elevation.
    """

    _axes = ("pol", "ew", "el")

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["pol", "freq", "ew", "el", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 1, 512, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "dirty_beam": {
            "axes": ["pol", "freq", "ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 1, 512, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "vis_weight": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "elevation_vis_weight": {
            "axes": ["pol", "freq", "ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 512, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "effective_ra": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "nsample": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "filter": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_filter": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    def add_dataset(self, name):
        """Override base class to deal with elevation dependent vis weight."""
        if name == "vis_weight" and "elevation_vis_weight" in self.datasets:
            raise RuntimeError(
                "Requesting creation of elevation-independent weights but "
                "elevation-dependent weights already exist."
            )
        if name == "elevation_vis_weight":
            if "vis_weight" in self.datasets:
                raise RuntimeError(
                    "Requesting creation of elevation-dependent weights but "
                    "elevation-independent weights already exist."
                )
            # Make this the default weight dataset
            self._weight_dset_name = "elevation_vis_weight"
        return super().add_dataset(name)

    @property
    def dirty_beam(self):
        """Not useful at this stage, but it's needed to propagate onward."""
        return self.datasets["dirty_beam"]

    @property
    def effective_ra(self):
        """Get the effective_ra dataset if it exists, None otherwise."""
        if "effective_ra" in self.datasets:
            return self.datasets["effective_ra"]

        raise KeyError("Dataset 'effective_ra' not initialised.")

    @property
    def nsample(self):
        """Get the nsample dataset if it exists, None otherwise."""
        if "nsample" in self.datasets:
            return self.datasets["nsample"]

        raise KeyError("Dataset 'nsample' not initialised.")

    @property
    def pol(self):
        """Get the polarisation index map."""
        return self.index_map["pol"]

    @property
    def ew(self):
        """Get the east-west baseline index map."""
        return self.index_map["ew"]


class HybridVisMModes(FreqContainer, MContainer, VisBase):
    """Visibilities beamformed in the NS direction and m-mode transformed in RA.

    This container has visibilities beam formed only in the NS direction to give a
    grid in elevation.
    """

    _axes = ("pol", "ew", "el")

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["m", "msign", "pol", "freq", "ew", "el"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "vis_weight": {
            "axes": ["m", "msign", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class RingMap(
    FilterFreqContainer, FreqContainer, SiderealContainer, DataWeightContainer
):
    """Container for holding multifrequency ring maps.

    The maps are packed in format `[freq, pol, ra, EW beam, el]` where
    the polarisations are Stokes I, Q, U and V.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    polarisation : bool, optional
        If `True` all Stokes parameters are stored, if `False` only Stokes I is
        stored.
    """

    _axes = ("pol", "beam", "el")

    _dataset_spec: ClassVar = {
        "map": {
            "axes": ["beam", "pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 1, 64, 128, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": {
                "weight_dataset": "weight",
            },
        },
        "weight": {
            "axes": ["pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 128, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "dirty_beam": {
            "axes": ["beam", "pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 1, 64, 128, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "dirty_beam_power": {
            "axes": ["beam", "pol", "freq", "el"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 1, 128, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "rms": {
            "axes": ["pol", "freq", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (4, 512, 512),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "filter": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 64, 256),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_filter": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 64, 256),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 64, 256),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 64, 256),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    _data_dset_name = "map"
    _weight_dset_name = "weight"

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]

    @property
    def map(self):
        """Get the map dataset."""
        return self.datasets["map"]

    @property
    def rms(self):
        """Get the rms dataset."""
        return self.datasets["rms"]

    @property
    def dirty_beam(self):
        """Get the dirty_beam dataset."""
        return self.datasets["dirty_beam"]

    @property
    def dirty_beam_power(self):
        """Get the dirty_beam_power dataset."""
        return self.datasets["dirty_beam_power"]


class RingMapMask(FreqContainer, SiderealContainer):
    """Mask bad ringmap pixels."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["pol", "freq", "ra", "el"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]


class RingMapTaper(FreqContainer, SiderealContainer):
    """Container for a smooth transition from good to bad ringmap pixels."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "taper": {
            "axes": ["pol", "freq", "ra", "el"],
            "dtype": float,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def taper(self):
        """Get the mask dataset."""
        return self.datasets["taper"]

    @property
    def weight(self):
        """Map weight to taper so that it can easily be updated with masks."""
        return self.datasets["taper"]


class FreqNoiseModel(FilterFreqContainer, FreqContainer, SiderealContainer):
    """Container storing Cholesky factors of frequency-frequency noise covariance.

    This container is intended for generating noise realizations in visibility space
    that include the desired correlations as a function of freq and el.  This is
    typically used to populate a VisGridStream container with synthetic noise having
    a specific spectral structure.
    """

    _axes = ("pol", "ew", "ns")

    _dataset_spec: ClassVar = {
        "redundancy": {
            "axes": ["pol", "ew", "ns"],
            "dtype": np.int32,
            "initialise": True,
            "distributed": False,
            "chunks": (1, 1, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "chunks": (1, 64, 1, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "freq_cov": {
            "axes": ["pol", "ew", "ra", "freq", "freq_sum"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 1, 2048, 64, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_freq_cov": {
            "axes": ["pol", "ew", "ra", "freq", "freq_sum"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 1, 2048, 64, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def redundancy(self):
        """Get the redundancy dataset."""
        return self.datasets["redundancy"]

    @property
    def weight(self):
        """Get the weight dataset."""
        return self.datasets["weight"]


class GainDataBase(DataWeightContainer):
    """A container interface for gain-like data.

    To support the previous behaviour of gain type data the weight dataset is optional,
    and returns None if it is not present.
    """

    _data_dset_name = "gain"
    _weight_dset_name = "weight"

    @property
    def gain(self) -> memh5.MemDataset:
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def weight(self) -> memh5.MemDataset | None:
        """The weights for each data point.

        Returns None is no weight dataset exists.
        """
        try:
            return super().weight
        except KeyError:
            return None


class CommonModeGainData(FreqContainer, TODContainer, GainDataBase):
    """Parallel container for holding gain data common to all inputs."""

    _dataset_spec: ClassVar = {
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


class CommonModeSiderealGainData(FreqContainer, SiderealContainer, GainDataBase):
    """Parallel container for holding sidereal gain data common to all inputs."""

    _dataset_spec: ClassVar = {
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


class GainData(FreqContainer, TODContainer, GainDataBase):
    """Parallel container for holding gain data."""

    _axes = ("input",)

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "input", "time"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "input", "time"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "update_id": {
            "axes": ["time"],
            "dtype": np.dtype("<U64"),
            "initialise": False,
            "distributed": False,
        },
    }

    @property
    def update_id(self):
        """Get the update id dataset if it exists."""
        try:
            return self.datasets["update_id"]
        except KeyError:
            return None

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]


class SiderealGainData(FreqContainer, SiderealContainer, GainDataBase):
    """Parallel container for holding sidereal gain data."""

    _axes = ("input",)

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]


class StaticGainData(FreqContainer, GainDataBase):
    """Parallel container for holding static gain data (i.e. non time varying)."""

    _axes = ("input",)

    _dataset_spec: ClassVar = {
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
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]


class DelayCutoff(ContainerBase):
    """Container for a delay cutoff."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "cutoff": {
            "axes": ["pol", "el"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "el",
        }
    }

    @property
    def cutoff(self):
        """Get the cutoff dataset."""
        return self.datasets["cutoff"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]


class DelayContainer(ContainerBase):
    """A container with a delay axis."""

    _axes = ("delay",)

    @property
    def delay(self) -> np.ndarray:
        """The delay axis in microseconds."""
        return self.index_map["delay"]


class DelaySpectrum(DelayContainer):
    """Container for a delay power spectrum.

    Notes
    -----
    A note about definitions: for a dataset with a frequency axis, the corresponding
    delay spectrum is the result of Fourier transforming in frequency, while the delay
    power spectrum is obtained by taking the squared magnitude of each element of the
    delay spectrum, and then usually averaging over some other axis. Our unfortunate
    convention is to store a delay power spectrum in a `DelaySpectrum` container, and
    store a delay spectrum in a :py:class:`~draco.core.containers.DelayTransform`
    container.
    """

    _axes = ("baseline", "sample")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["baseline", "delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "spectrum_samples": {
            "axes": ["sample", "baseline", "delay"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "spectrum_mask": {
            "axes": ["baseline"],
            "dtype": bool,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }

    def __init__(self, *args, weight_boost=1.0, sample=1, **kwargs):
        super().__init__(*args, sample=sample, **kwargs)
        self.attrs["weight_boost"] = weight_boost

    @property
    def spectrum(self):
        """Get the spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def weight_boost(self):
        """Get the weight boost factor.

        If set, this factor was used to set the assumed noise when computing the
        spectrum.
        """
        return self.attrs["weight_boost"]

    @property
    def freq(self):
        """Get the frequency axis of the input data."""
        return self.attrs["freq"]


class DelayTransform(DelayContainer):
    """Container for a delay spectrum.

    Notes
    -----
    See the docstring for :py:class:`~draco.core.containers.DelaySpectrum` for a
    description of the difference between `DelayTransform` and `DelaySpectrum`.
    """

    _axes = ("baseline", "sample")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["baseline", "sample", "delay"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (512, 2048, 32),
            "truncate": True,
        },
        "weight": {
            "axes": ["baseline", "sample", "delay"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (512, 2048, 32),
            "truncate": True,
        },
        "spectrum_mask": {
            "axes": ["baseline", "sample"],
            "dtype": bool,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }

    def __init__(self, weight_boost=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs["weight_boost"] = weight_boost

    @property
    def spectrum(self):
        """Get the spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def weight(self):
        """Get the spectrum dataset."""
        return self.datasets["weight"]

    @property
    def weight_boost(self):
        """Get the weight boost factor.

        If set, this factor was used to set the assumed noise when computing the
        spectrum.
        """
        return self.attrs["weight_boost"]

    @property
    def freq(self):
        """Get the frequency axis of the input data."""
        return self.attrs["freq"]


class DelayTransformOperator(DelayContainer, FreqContainer, SiderealContainer):
    """Wiener filter that transforms each pixel from frequency to delay."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "filter": {
            "axes": ["pol", "ra", "el", "delay", "freq"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        }
    }

    @property
    def filter(self):
        """Get the filter dataset."""
        return self.datasets["filter"]


class CosmologyContainer(ContainerBase):
    """A baseclass for a container that is referenced to a background Cosmology.

    Parameters
    ----------
    cosmology
        An explicit cosmology instance or dict representation. If not set, the cosmology
        *must* get set via `attrs_from`.
    """

    def __init__(self, cosmology: Cosmology | dict | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cosmo_dict = self._resolve_args(cosmology, **kwargs)
        self.attrs["cosmology"] = cosmo_dict

    @staticmethod
    def _resolve_args(
        cosmology: Cosmology | dict | None = None,
        attrs_from: ContainerBase | None = None,
        **kwargs,
    ):
        """Try and extract a Cosmology dict representation from the parameters.

        Useful as subclasses sometimes need access *before* the full class is setup.
        """
        # Insert the Cosmological parameters
        if cosmology is None:
            if attrs_from is not None and "cosmology" in attrs_from.attrs:
                cosmology = attrs_from.attrs["cosmology"]
            else:
                raise ValueError("A cosmology must be supplied.")
        elif not isinstance(cosmology, (Cosmology | dict)):
            raise TypeError("cosmology argument must be a Cosmology instance.")

        if isinstance(cosmology, Cosmology):
            cosmology = cosmology.to_dict()

        return cosmology

    _cosmology_instance = None

    @property
    def cosmology(self):
        """The background cosmology."""
        if self._cosmology_instance is None:
            self._cosmology_instance = Cosmology(**self.attrs["cosmology"])

        return self._cosmology_instance


class Fourier3DContainer(CosmologyContainer, DelayContainer):
    """A base container with Fourier axes, (pol,delay,u,v)."""

    _axes = ("pol", "u", "v")

    _dataset_spec: ClassVar = {
        "kx": {
            "axes": ["u"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "ky": {
            "axes": ["v"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "kpara": {
            "axes": ["delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "uv_mask": {
            "axes": ["u", "v"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def kx(self):
        """Get the kx axis."""
        return self.datasets["kx"]

    @property
    def ky(self):
        """Get the ky axis."""
        return self.datasets["ky"]

    @property
    def kpara(self):
        """Get the k_parallel axis."""
        return self.datasets["kpara"]

    @property
    def uv_mask(self):
        """Get the uv-domain mask."""
        return self.datasets["uv_mask"]

    @property
    def redshift(self):
        """Get the redshift attrs."""
        return self.attrs["redshift"]

    @property
    def freq_center(self):
        """Get the central frequency attrs."""
        return self.attrs["freq_center"]


class SpatialDelayCube(Fourier3DContainer):
    """Container for a data in (pol,delays,u,v) domain."""

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["pol", "delay", "u", "v"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        },
    }

    @property
    def vis(self):
        """Get the spatial data cube."""
        return self.datasets["vis"]


class PowerSpectrum3D(Fourier3DContainer):
    """Container for a 3D power spectrum."""

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["pol", "delay", "u", "v"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        }
    }

    @property
    def spectrum(self):
        """Get the 3D power spectrum."""
        return self.datasets["spectrum"]

    @property
    def ps_norm(self):
        """Get the power spectrum normalizaiton attrs."""
        return self.attrs["ps_norm"]


class PowerSpectrum2D(CosmologyContainer):
    """Container for a 2D cylindrically averaged  power spectrum."""

    _axes = ("pol", "delay", "uv_dist")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        },
        "weight": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "neff": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        },
        "mask": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
        },
        "kpara": {
            "axes": ["delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "kperp": {
            "axes": ["uv_dist"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def spectrum(self):
        """Get the 2D power spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def weight(self):
        """Get the 2D weight dataset."""
        return self.datasets["weight"]

    @property
    def neff(self):
        """Get the effective number of modes dataset."""
        return self.datasets["neff"]

    @property
    def mask(self):
        """Get the 2D signal window dataset."""
        return self.datasets["mask"]

    @property
    def kpara(self):
        """Get the k_parallel axis."""
        return self.datasets["kpara"]

    @property
    def kperp(self):
        """Get the kprep axis."""
        return self.datasets["kperp"]

    @property
    def delay_cut(self):
        """Get the delay cutoff value."""
        return self.attrs["delay_cut"]


class PowerSpectrum1D(CosmologyContainer):
    """Container for a 1D power spectrum."""

    _axes = ("pol", "k")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["pol", "k"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
        },
        "samp_var": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "var": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "neff": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "k1D": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
    }

    @property
    def spectrum(self):
        """Get the 1D power spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def samp_var(self):
        """Get the 1D power spectrum error dataset."""
        return self.datasets["samp_var"]

    @property
    def var(self):
        """Get the 1D power spectrum var dataset."""
        return self.datasets["var"]

    @property
    def neff(self):
        """Get the 1D power spectrum var dataset."""
        return self.datasets["neff"]

    @property
    def k1D(self):
        """Get the k1D dataset."""
        return self.datasets["k1D"]


class WaveletSpectrum(FreqContainer, DelayContainer, DataWeightContainer):
    """Container for a wavelet power spectrum."""

    _axes = ("baseline",)

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["baseline", "delay", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "weight": {
            "axes": ["baseline", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }
    _data_dset_name = "spectrum"
    _weight_dset_name = "weight"

    @property
    def spectrum(self):
        """The wavelet spectrum."""
        return self.datasets["spectrum"]


class DelayCrossSpectrum(DelaySpectrum):
    """Container for a delay cross power spectra."""

    _axes = ("dataset",)

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["dataset", "dataset", "baseline", "delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "spectrum_samples": {
            "axes": ["sample", "dataset", "dataset", "baseline", "delay"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }

    @property
    def spectrum(self):
        """Get the spectrum dataset."""
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

    _dataset_spec: ClassVar = {
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

        super().__init__(*args, **kwargs)

    @property
    def powerspectrum(self):
        """Get the powerspectrum dataset."""
        return self.datasets["powerspectrum"]

    @property
    def C_inv(self):
        """Get the C inverse dataset."""
        return self.datasets["C_inv"]


class SVDSpectrum(ContainerBase):
    """Container for an m-mode SVD spectrum."""

    _axes = ("m", "singularvalue")

    _dataset_spec: ClassVar = {
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
        """Get the spectrum dataset."""
        return self.datasets["spectrum"]


class FrequencyStack(FreqContainer, DataWeightContainer):
    """Container for a frequency stack.

    In general used to hold the product of `draco.analysis.SourceStack`
    The stacked signal of frequency slices of the data in the direction
    of sources of interest.
    """

    _dataset_spec: ClassVar = {
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

    _data_dset_name = "stack"
    _weight_dset_name = "weight"

    @property
    def stack(self):
        """Get the stack dataset."""
        return self.datasets["stack"]


class FrequencyStackByPol(FrequencyStack):
    """Container for a frequency stack split by polarisation."""

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]


class MockFrequencyStack(FrequencyStack):
    """Container for holding a frequency stack for multiple mock catalogs.

    Adds a `mock` axis as the first dimension of each dataset.
    """

    _axes = ("mock",)

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["mock", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["mock", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }


class MockFrequencyStackByPol(FrequencyStackByPol):
    """Container for holding a frequency stack split by pol for multiple mock catalogs.

    Adds a `mock` axis as the first dimension of each dataset.
    """

    _axes = ("mock",)

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["mock", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["mock", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }


class Stack3D(FreqContainer, DataWeightContainer):
    """Container for a 3D frequency stack."""

    _axes = ("pol", "delta_ra", "delta_dec")

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    _data_dset_name = "stack"
    _weight_dset_name = "weight"

    @property
    def stack(self):
        """Get the stack dataset."""
        return self.datasets["stack"]


class SourceCatalog(TableBase):
    """A basic container for holding astronomical source catalogs.

    Notes
    -----
    The `ra` and `dec` coordinates should be ICRS.
    """

    _table_spec: ClassVar = {
        "position": {
            "columns": [["ra", np.float64], ["dec", np.float64]],
            "axis": "object_id",
        }
    }


class SpectroscopicCatalog(SourceCatalog):
    """A container for spectroscopic catalogs."""

    _table_spec: ClassVar = {
        "redshift": {
            "columns": [["z", np.float64], ["z_error", np.float64]],
            "axis": "object_id",
        }
    }


class FormedBeam(FreqContainer, DataWeightContainer):
    """Container for formed beams."""

    _axes = ("object_id", "pol")

    _dataset_spec: ClassVar = {
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
            "initialise": False,
            "distributed": False,
        },
    }

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def position(self):
        """Get the position dataset."""
        return self.datasets["position"]

    @property
    def redshift(self):
        """Get the redshift dataset."""
        if "redshift" in self.datasets:
            return self.datasets["redshift"]

        raise KeyError("Dataset 'redshift' not initialised.")

    @property
    def frequency(self):
        """Get the frequency axis."""
        return self.index_map["freq"]

    @property
    def id(self):
        """Get the object id axis."""
        return self.index_map["object_id"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]


class FormedBeamHA(FormedBeam):
    """Container for formed beams.

    These have not been collapsed in the hour angle (HA) axis.
    """

    _axes = ("ha",)

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (32, 4, 128, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (32, 4, 128, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
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
        """Get the hour angle dataset."""
        return self.datasets["object_ha"]


class FormedBeamHAEW(FormedBeamHA):
    """Container for formed beams constructed from a HybridVisStream.

    These have not been collapsed along the hour angle (ha) or
    east west baseline (ew) axis.
    """

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "ew", "ha"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (8, 4, 128, 4, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "ew", "ha"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (8, 4, 128, 4, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "object_ha": {
            "axes": ["object_id", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def ew(self):
        """Get the ew index map."""
        return self.index_map["ew"]


class FitFormedBeam(FormedBeam):
    """Container for formed beams fit to a primary beam model versus hour angle."""

    _dataset_spec: ClassVar = {
        "background": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight_background": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "corr_background_beam": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def background(self):
        """Get the background dataset."""
        return self.datasets["background"]

    @property
    def weight_background(self):
        """Get the weight_background dataset."""
        return self.datasets["weight_background"]

    @property
    def corr_background_beam(self):
        """Get the corr_background_beam dataset."""
        return self.datasets["corr_background_beam"]


class FitFormedBeamEW(FitFormedBeam):
    """Container for formed beams fit to a primary beam model versus hour angle.

    These have not been collapsed along the east west baseline (ew) axis.
    """

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "background": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight_background": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "corr_background_beam": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def ew(self):
        """Get the ew index map."""
        return self.index_map["ew"]


class FormedBeamMask(FreqContainer):
    """Mask bad formed beams."""

    _axes = ("object_id", "pol")

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]


class FormedBeamHAMask(FormedBeamMask):
    """Mask bad formed beams as a function of hour angle."""

    _axes = ("ha",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }


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

    raise RuntimeError(
        f"I don't know how to deal with data type {obj.__class__.__name__}"
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


def copy_datasets_filter(
    source: ContainerBase,
    dest: ContainerBase,
    axis: str | list | tuple = [],
    selection: np.ndarray | list | slice | dict = {},
    exclude_axes: list[str] | None = None,
    copy_without_selection: bool = False,
):
    """Copy datasets while filtering a given axis.

    By default, only datasets containing the axis to be filtered will be copied.

    Parameters
    ----------
    source
        Source container
    dest
        Destination container. The axes in this container should reflect the
        selections being made to the source.
    axis
        Name of the axes to filter. These must match the axes in `selection`,
        unless selection is a single item. This is partially here for legacy
        reasons, as the selections can be fully specified by `selection`
    selection
        A filtering selection to be applied to each axis.
    exclude_axes
        An optional set of axes that if a dataset contains one means it will
        not be copied.
    copy_without_selection
        If set to True, then datasets that do not have an axis appearing in
        selection will still be copied over in full.  Default is False.
    """
    exclude_axes_set = set(exclude_axes) if exclude_axes else set()
    if isinstance(axis, str):
        axis = [axis]
    axis = set(axis)

    # Resolve the selections and axes, removing any that aren't needed
    if not isinstance(selection, dict):
        # Assume we just want to apply this selection to all listed axes
        selection = dict.fromkeys(axis, selection)

    if not axis:
        axis = set(selection.keys())
    # Make sure that all axis keys are present in selection
    elif not all(ax in selection for ax in axis):
        raise ValueError(
            f"Mismatch between axis and selection. Got {axis} "
            f"but selections for {list(selection.keys())}."
        )

    # Try to clean up selections
    for ax in list(selection):
        sel = selection[ax]
        # Remove any unnecessary slices
        if sel == slice(None):
            del selection[ax]
        # Convert any indexed selections to slices where possible
        elif type(sel) in {list, tuple, np.ndarray}:
            if list(sel) == list(range(sel[0], sel[-1])):
                selection[ax] = slice(sel[0], sel[-1])

    stack = [source]

    while stack:
        item = stack.pop()

        if memh5.is_group(item):
            stack += list(item.values())
            continue

        item_axes = list(item.attrs.get("axis", ()))

        # Do not copy datasets that contain excluded axes
        if not exclude_axes_set.isdisjoint(item_axes):
            continue

        # Unless requested, do not copy datasets that do not contain selected axes
        if not copy_without_selection and not axis.intersection(item_axes):
            continue

        if item.name not in dest:
            dest.add_dataset(item.name)

        dest_dset = dest[item.name]

        # Make sure that both datasets are distributed to the same axis
        if isinstance(item, memh5.MemDatasetDistributed):
            if not isinstance(dest_dset, memh5.MemDatasetDistributed):
                raise ValueError(
                    "Cannot filter a distributed dataset into a non-distributed "
                    "dataset using this method."
                )

            # Choose the best possible axis to distribute over. Try
            # to avoid redistributing if possible
            original_ax_id = item.distributed_axis
            # If no selections are being made or the slection is not over the
            # current axis, so no need to redistribute
            if not selection or item_axes[original_ax_id] not in selection:
                new_ax_id = original_ax_id
            else:
                # Find the largest axis available
                ax_priority = [
                    x for _, x in sorted(zip(item.shape, item_axes)) if x not in axis
                ]
                if not ax_priority:
                    raise ValueError(
                        "Could not find a valid axis to redistribute. At least one "
                        "axis must be omitted from filtering."
                    )
                new_ax_id = item_axes.index(ax_priority[-1])

            # Make sure both datasets are distributed to the same axis.
            item.redistribute(new_ax_id)
            dest_dset.redistribute(new_ax_id)

        # Apply the selections
        arr = item[:].view(np.ndarray)

        for ax, sel in selection.items():
            try:
                ax_ind = item_axes.index(ax)
            except ValueError:
                continue
            arr = mpiarray._apply_sel(arr, sel, ax_ind)

        dest_dset[:] = arr[:]

        # also copy attritutes
        memh5.copyattrs(item.attrs, dest_dset.attrs)

        if isinstance(dest_dset, memh5.MemDatasetDistributed):
            # Redistribute back to the original axis
            dest_dset.redistribute(original_ax_id)


class LocalizedRFIMask(FreqContainer, TODContainer):
    """Container for an RFI mask for each freq, el, and time sample.

    The data frac_rfi stores information about the proportion of subdata
    that detected RFI, which is used to generate the mask.
    """

    _axes = ("el",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "el", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "frac_rfi": {
            "axes": ["freq", "el", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 128, 512),
            "truncate": True,
        },
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def frac_rfi(self):
        """Get the frac_rfi dataset."""
        return self.datasets["frac_rfi"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]


class LocalizedSiderealRFIMask(FreqContainer, SiderealContainer):
    """Container for an RFI mask for each freq, ra, and el sample.

    The data frac_rfi stores information about the proportion of subdata
    that detected RFI, which is used to generate the mask.
    """

    _axes = ("el",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "ra", "el"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "frac_rfi": {
            "axes": ["freq", "ra", "el"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 512, 128),
            "truncate": True,
        },
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def frac_rfi(self):
        """Get the frac_rfi dataset."""
        return self.datasets["frac_rfi"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]


class VisBandpassWindow(FreqContainer):
    """Container for bandpass gains and their window estimated by running bandpass HyFoReS on hybrid beam-formed visibilities."""

    _axes = ("pol",)

    # TODO: check if np.complex128 is required
    _dataset_spec: ClassVar = {
        "bandpass": {
            "axes": ["pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "window": {
            "axes": ["pol", "freq", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def bandpass(self):
        """Get the bandpass dataset."""
        return self.datasets["bandpass"]

    @property
    def window(self):
        """Get the window dataset."""
        return self.datasets["window"]


class VisBandpassCompensate(FreqContainer):
    """Container for window-compensated bandpass gains."""

    _axes = ("pol",)

    # TODO: redefine the second axis for sval
    _dataset_spec: ClassVar = {
        "comp_bandpass": {
            "axes": ["pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "sval": {
            "axes": ["pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def comp_bandpass(self):
        """Get the comp_bandpass dataset."""
        return self.datasets["comp_bandpass"]

    @property
    def sval(self):
        """Get the sval dataset."""
        return self.datasets["sval"]


class VisBandpassWindowBaseline(VisBandpassWindow):
    """Container for bandpass gains and their window estimated by running bandpass HyFoReS on hybrid beam-formed visibilities."""

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "bandpass": {
            "axes": ["pol", "ew", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "window": {
            "axes": ["pol", "ew", "freq", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def bandpass(self):
        """Get the bandpass dataset."""
        return self.datasets["bandpass"]

    @property
    def window(self):
        """Get the window dataset."""
        return self.datasets["window"]


class VisBandpassCompensateBaseline(VisBandpassCompensate):
    """Container for window-compensated bandpass gains."""

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "comp_bandpass": {
            "axes": ["pol", "ew", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "sval": {
            "axes": ["pol", "ew", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def comp_bandpass(self):
        """Get the comp_bandpass dataset."""
        return self.datasets["comp_bandpass"]

    @property
    def sval(self):
        """Get the sval dataset."""
        return self.datasets["sval"]


class VisBandpassWindowBaselineRA(SiderealContainer, VisBandpassWindowBaseline):
    """Container for bandpass gains and their window estimated by running bandpass HyFoReS on hybrid beam-formed visibilities."""

    _dataset_spec: ClassVar = {
        "bandpass": {
            "axes": ["pol", "ew", "ra", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "window": {
            "axes": ["pol", "ew", "ra", "freq", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048, 32, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def bandpass(self):
        """Get the bandpass dataset."""
        return self.datasets["bandpass"]

    @property
    def window(self):
        """Get the window dataset."""
        return self.datasets["window"]


class VisBandpassCompensateBaselineRA(SiderealContainer, VisBandpassCompensateBaseline):
    """Container for window-compensated bandpass gains."""

    _dataset_spec: ClassVar = {
        "comp_bandpass": {
            "axes": ["pol", "ew", "ra", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "rank": {
            "axes": ["pol", "ew", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def comp_bandpass(self):
        """Get the comp_bandpass dataset."""
        return self.datasets["comp_bandpass"]

    @property
    def rank(self):
        """Get the sval dataset."""
        return self.datasets["rank"]


class HorizonLimit(ContainerBase):
    """Container holding the altitude of the horizon as a function of azimuth."""

    _axes = ("azimuth",)

    _dataset_spec: ClassVar = {
        "altitude": {
            "axes": ["azimuth"],
            "dtype": float,
            "initialise": True,
            "distributed": False,
        }
    }

    def get_horizon_limit(self, az):
        """Interpolate the horizon altitude at a given azimuth.

        Parameters
        ----------
        az : float or ndarray
            Azimuth angle in degrees at which to evaluate the horizon limit.

        Returns
        -------
        alt : float or ndarray
            Interpolated horizon altitude(s) in degrees.
        """
        return np.interp(az, self.azimuth, self.altitude, period=360.0)

    @property
    def azimuth(self):
        """Get the index map containing the azimuth angle (in degrees)."""
        return self.index_map["azimuth"]

    @property
    def altitude(self):
        """Get the dataset containing the altitude (in degrees)."""
        return self.datasets["altitude"]

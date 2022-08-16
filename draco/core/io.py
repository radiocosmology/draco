"""Tasks for reading and writing data.

File Groups
===========

Several tasks accept groups of files as arguments. These are specified in the YAML file as a dictionary like below.

.. code-block:: yaml

    list_of_file_groups:
        -   tag: first_group  # An optional tag naming the group
            files:
                -   'file1.h5'
                -   'file[3-4].h5'  # Globs are processed
                -   'file7.h5'

        -   files:  # No tag specified, implicitly gets the tag 'group_2'
                -   'another_file1.h5'
                -   'another_file2.h5'


    single_group:
        files: ['file1.h5', 'file2.h5']
"""
import os.path
import shutil
from typing import Union, Dict, List

import numpy as np
from yaml import dump as yamldump

from caput import pipeline, config, fileformats, memh5, truncate, mpiutil

from cora.util import units

from drift.core import telescope, manager, beamtransfer

from . import task
from ..util.exception import ConfigError


def _list_of_filelists(files: Union[List[str], List[List[str]]]) -> List[List[str]]:
    """Take in a list of lists/glob patterns of filenames

    Parameters
    ----------
    files
        A path or glob pattern (e.g. /my/data/*.h5) or a list of those (or a list of
        lists of those).

    Raises
    ------
    ConfigError
        If files has the wrong format or refers to a file that doesn't exist.

    Returns
    -------
    The input file list list. Any glob patterns will be flattened to file path string
    lists.
    """
    import glob

    f2 = []

    for filelist in files:

        if isinstance(filelist, str):
            if "*" not in filelist and not os.path.isfile(filelist):
                raise ConfigError("File not found: %s" % filelist)
            filelist = glob.glob(filelist)
        elif isinstance(filelist, list):
            for i in range(len(filelist)):
                filelist[i] = _list_or_glob(filelist[i])
        else:
            raise ConfigError("Must be list or glob pattern.")
        f2 = f2 + filelist

    return f2


def _list_or_glob(files: Union[str, List[str]]) -> List[str]:
    """Take in a list of lists/glob patterns of filenames

    Parameters
    ----------
    files
        A path or glob pattern (e.g. /my/data/*.h5) or a list of those

    Returns
    -------
    The input file list. Any glob patterns will be flattened to file path string lists.

    Raises
    ------
    ConfigError
        If files has the wrong type or if it refers to a file that doesn't exist.
    """
    import glob

    # If the input was a list, process each element and return as a single flat list
    if isinstance(files, list):
        parsed_files = []
        for f in files:
            parsed_files = parsed_files + _list_or_glob(f)
        return parsed_files

    # If it's a glob we need to expand the glob and then call again
    if isinstance(files, str) and "*" in files:
        return _list_or_glob(sorted(glob.glob(files)))

    # We presume a string is an actual path...
    if isinstance(files, str):

        # Check that it exists and is a file (or dir if zarr format)
        if files.endswith(".zarr"):
            if not os.path.isdir(files):
                raise ConfigError(
                    f"Expected a zarr directory store, but directory not found: {files}"
                )
            return [files]
        else:
            if not os.path.isfile(files):
                raise ConfigError("File not found: %s" % files)
            return [files]

    raise ConfigError(
        "Argument must be list, glob pattern, or file path, got %s" % repr(files)
    )


def _list_of_filegroups(groups: Union[List[Dict], Dict]) -> List[Dict]:
    """Process a file group/groups

    Parameters
    ----------
    groups
        Dicts should contain keys 'files': An iterable with file path or glob pattern
        strings, 'tag': the group tag str

    Returns
    -------
    The input groups. Any glob patterns in the 'files' list will be flattened to file
    path strings.

    Raises
    ------
    ConfigError
        If groups has the wrong format.
    """
    import glob

    # Convert to list if the group was not included in a list
    if not isinstance(groups, list):
        groups = [groups]

    # Iterate over groups, set the tag if needed, and process the file list
    # through glob
    for gi, group in enumerate(groups):

        try:
            files = group["files"]
        except KeyError:
            raise ConfigError("File group is missing key 'files'.")
        except TypeError:
            raise ConfigError(
                "Expected type dict in file groups (got {}).".format(type(group))
            )

        if "tag" not in group:
            group["tag"] = "group_%i" % gi

        flist = []

        for fname in files:
            if "*" not in fname and not os.path.isfile(fname):
                raise ConfigError("File not found: %s" % fname)
            flist += glob.glob(fname)

        if not len(flist):
            raise ConfigError("No files in group exist (%s)." % files)

        group["files"] = flist

    return groups


class LoadMaps(task.MPILoggedTask):
    """Load a series of maps from files given in the tasks parameters.

    Maps are given as one, or a list of `File Groups` (see
    :mod:`draco.core.io`). Maps within the same group are added together
    before being passed on.

    Attributes
    ----------
    maps : list or dict
        A dictionary specifying a file group, or a list of them.
    """

    maps = config.Property(proptype=_list_of_filegroups)

    def next(self):
        """Load the groups of maps from disk and pass them on.

        Returns
        -------
        map : :class:`containers.Map`
        """

        from . import containers

        # Exit this task if we have eaten all the file groups
        if len(self.maps) == 0:
            raise pipeline.PipelineStopIteration

        group = self.maps.pop(0)

        map_stack = None

        # Iterate over all the files in the group, load them into a Map
        # container and add them all together
        for mfile in group["files"]:

            self.log.debug("Loading file %s", mfile)

            current_map = containers.Map.from_file(mfile, distributed=True)
            current_map.redistribute("freq")

            # Start the stack if needed
            if map_stack is None:
                map_stack = current_map

            # Otherwise, check that the new map has consistent frequencies,
            # nside and pol and stack up.
            else:

                if (current_map.freq != map_stack.freq).all():
                    raise RuntimeError("Maps do not have consistent frequencies.")

                if (current_map.index_map["pol"] != map_stack.index_map["pol"]).all():
                    raise RuntimeError("Maps do not have the same polarisations.")

                if (
                    current_map.index_map["pixel"] != map_stack.index_map["pixel"]
                ).all():
                    raise RuntimeError("Maps do not have the same pixelisation.")

                map_stack.map[:] += current_map.map[:]

        # Assign a tag to the stack of maps
        map_stack.attrs["tag"] = group["tag"]

        return map_stack


class LoadFITSCatalog(task.SingleTask):
    """Load an SDSS-style FITS source catalog.

    Catalogs are given as one, or a list of `File Groups` (see
    :mod:`draco.core.io`). Catalogs within the same group are combined together
    before being passed on.

    Attributes
    ----------
    catalogs : list or dict
        A dictionary specifying a file group, or a list of them.
    z_range : list, optional
        Select only sources with a redshift within the given range.
    freq_range : list, optional
        Select only sources with a 21cm line freq within the given range. Overrides
        `z_range`.
    """

    catalogs = config.Property(proptype=_list_of_filegroups)
    z_range = config.list_type(type_=float, length=2, default=None)
    freq_range = config.list_type(type_=float, length=2, default=None)

    def process(self):
        """Load the groups of catalogs from disk, concatenate them and pass them on.

        Returns
        -------
        catalog : :class:`containers.SpectroscopicCatalog`
        """

        from astropy.io import fits
        from . import containers

        # Exit this task if we have eaten all the file groups
        if len(self.catalogs) == 0:
            raise pipeline.PipelineStopIteration

        group = self.catalogs.pop(0)

        # Set the redshift selection
        if self.freq_range:
            zl = units.nu21 / self.freq_range[1] - 1
            zh = units.nu21 / self.freq_range[0] - 1
            self.z_range = (zl, zh)

        if self.z_range:
            zl, zh = self.z_range
            self.log.info(f"Applying redshift selection {zl:.2f} <= z <= {zh:.2f}")

        # Load the data only on rank=0 and then broadcast
        if self.comm.rank == 0:
            # Iterate over all the files in the group, load them into a Map
            # container and add them all together
            catalog_stack = []
            for cfile in group["files"]:

                self.log.debug("Loading file %s", cfile)

                # TODO: read out the weights from the catalogs
                with fits.open(cfile, mode="readonly") as cat:
                    pos = np.array([cat[1].data[col] for col in ["RA", "DEC", "Z"]])

                # Apply any redshift selection to the objects
                if self.z_range:
                    zsel = (pos[2] >= self.z_range[0]) & (pos[2] <= self.z_range[1])
                    pos = pos[:, zsel]

                catalog_stack.append(pos)

            # NOTE: this one is tricky, for some reason the concatenate in here
            # produces a non C contiguous array, so we need to ensure that otherwise
            # the broadcasting will get very confused
            catalog_array = np.concatenate(catalog_stack, axis=-1).astype(np.float64)
            catalog_array = np.ascontiguousarray(catalog_array)
            num_objects = catalog_array.shape[-1]
        else:
            num_objects = None
            catalog_array = None

        # Broadcast the size of the catalog to all ranks, create the target array and
        # broadcast into it
        num_objects = self.comm.bcast(num_objects, root=0)
        self.log.debug(f"Constructing catalog with {num_objects} objects.")

        if self.comm.rank != 0:
            catalog_array = np.zeros((3, num_objects), dtype=np.float64)
        self.comm.Bcast(catalog_array, root=0)

        catalog = containers.SpectroscopicCatalog(object_id=num_objects)
        catalog["position"]["ra"] = catalog_array[0]
        catalog["position"]["dec"] = catalog_array[1]
        catalog["redshift"]["z"] = catalog_array[2]
        catalog["redshift"]["z_error"] = 0

        # Assign a tag to the stack of maps
        catalog.attrs["tag"] = group["tag"]

        return catalog


class LoadFITSDelta(task.SingleTask):
    """Load eBOSS Lyman-alpha delta files in FITS format.

    See https://data.sdss.org/datamodel/files/EBOSS_LYA/DELTA_LYAB/delta.html
    for data format specification.

    Attributes
    ----------
    delta_files : list or dict
        A dictionary specifying a file group, or a list of them.
    """

    delta_files = config.Property(proptype=_list_of_filegroups)

    def process(self):
        """Load spectra from all files and combine into a single
        container with a common frequency axis.

        Returns
        -------
        spectra : :class:`containers.FormedBeam`
        """

        from astropy.io import fits
        from . import containers
        from time import time

        # Exit this task if we have eaten all the file groups
        if len(self.delta_files) == 0:
            raise pipeline.PipelineStopIteration

        group = self.delta_files.pop(0)

        if self.comm.rank == 0:

            # lists to read spectra into
            delta = []
            weight = []
            pos = []
            z = []
            loglam = []
            obj_id = []

            for fname in group["files"]:

                # read FITS file
                with fits.open(fname, mode="readonly") as fh:
                    # read all data into memory
                    t0 = time()
                    self.log.debug(f"(rank {self.comm.rank}) reading {fname}...")
                    fh.readall()
                    self.log.debug(f"in memory ({time() - t0} s)")

                    # prepare arrays to copy into
                    n_rec = len(fh) - 1
                    this_pos = np.zeros((n_rec, 2), dtype=np.float64)
                    this_z = np.zeros(n_rec, dtype=np.float64)
                    this_id = np.zeros(n_rec, dtype=np.int32)

                    # loop through HDUs and read
                    for i, rec in enumerate(fh[1:]):
                        this_pos[i] = tuple([rec.header[c] for c in ["RA", "DEC"]])
                        this_z[i] = rec.header["Z"]
                        this_id[i] = rec.header["THING_ID"]
                        delta.append(np.ascontiguousarray(rec.data["DELTA"]))
                        weight.append(np.ascontiguousarray(rec.data["WEIGHT"]))
                        loglam.append(np.ascontiguousarray(rec.data["LOGLAM"]))

                    pos.append(this_pos)
                    z.append(this_z)
                    obj_id.append(this_id)

                    self.log.debug(f"{fname} done ({time() - t0} s)")

            # these have a single value per object
            pos = np.concatenate(pos, axis=0)
            z = np.concatenate(z)
            obj_id = np.concatenate(obj_id)

            # create common wavelength axis
            loglam_all = np.concatenate(loglam)
            llam_max = loglam_all.max()
            llam_min = loglam_all.min()
            dllam = np.min(np.abs(np.diff(loglam[0])))
            # N.B. all spectra must share a common resolution
            for ll in loglam:
                this_dllam = np.min(np.abs(np.diff(ll)))
                if np.abs(dllam - this_dllam) / dllam > 1e-3:
                    self.log.warn("Not all spectra use same gridding.")
            nlam = int(np.round((llam_max - llam_min) / dllam)) + 1
            lam = np.power(10, llam_min + np.arange(nlam) * dllam)

            # sort data into a single array
            delta_common = np.zeros((len(delta), nlam), dtype=delta[0].dtype)
            weight_common = np.zeros((len(weight), nlam), dtype=weight[0].dtype)
            t0 = time()
            for i in range(len(delta)):
                ind = np.argmin(
                    np.abs(np.power(10, loglam[i])[:, np.newaxis] - lam), axis=1
                )
                delta_common[i, ind] = delta[i]
                weight_common[i, ind] = weight[i]
            self.log.debug(f"Done sorting spectra. ({time() - t0:.1f} s)")
            delta = delta_common
            weight = weight_common

            # weights are inverse variance estimate with a redshift-dependent bias correction
            # (eq. 7 in du Mas des Bourboux et al)

            # can't just concatenate because lam axis is different
            num_objects = len(pos)
            num_pix = len(lam)

            self.log.debug("rank 0 done reading files")
            self.log.debug(f"delta shape {delta.shape}")

        else:
            num_objects = None
            num_pix = None
            delta = None
            weight = None

        self.comm.Barrier()

        # Broadcast the size of the catalog to all ranks
        num_objects = self.comm.bcast(num_objects, root=0)
        num_pix = self.comm.bcast(num_pix, root=0)
        self.log.debug(f"Reading spectra for {num_objects} objects.")

        # broadcast the axes
        if self.comm.rank != 0:
            pos = np.zeros((num_objects, 2), dtype=np.float64)
            z = np.zeros(num_objects, dtype=np.float64)
            lam = np.zeros(num_pix, dtype=np.float64)
            obj_id = np.zeros(num_objects, dtype=np.int32)
        self.comm.Bcast(pos, root=0)
        self.comm.Bcast(z, root=0)
        self.comm.Bcast(lam, root=0)
        self.comm.Bcast(obj_id, root=0)
        self.log.debug(f"(rank {self.comm.rank}) got axes.")

        # convert wavelength (A) to frequency (MHz)
        freq = units.c / lam * 1e4

        # scatter the distributed datasets over frequency
        def scatter_array(array, root=0, axis=0):
            # split distributed axis
            if self.comm.rank == root:
                shape = array.shape
                n = shape[axis]
                dtype = array.dtype
            else:
                shape = None
                n = None
                dtype = None
            shape = self.comm.bcast(shape, root=root)
            dtype = self.comm.bcast(dtype, root=root)
            mpitype = mpiutil.typemap(dtype)
            n = self.comm.bcast(n, root=root)
            n_all, s_all, e_all = mpiutil.split_all(n, comm=self.comm)
            slices = []
            for r in range(self.comm.size):
                this_slice = [slice(None)] * len(shape)
                this_slice[axis] = slice(s_all[r], e_all[r])
                slices.append(tuple(this_slice))

            # scatter array from source rank
            if self.comm.rank == root:
                local = array[slices[root]]
                # send to each rank
                sends = []
                for r in range(self.comm.size):
                    if r == root:
                        continue
                    to_send = array[slices[r]].flatten()
                    sends.append(self.comm.Isend([to_send, mpitype], dest=r))
                for req in sends:
                    req.wait()
            else:
                local_shape = list(shape)
                local_shape[axis] = n_all[self.comm.rank]
                local = np.zeros(local_shape, dtype=dtype)
                self.comm.Recv([local, mpitype], source=root)

            self.comm.Barrier()

            return local

        self.log.debug(f"(rank {self.comm.rank}) scattering arrays...")
        delta_local = scatter_array(delta, root=0, axis=1)
        weight_local = scatter_array(weight, root=0, axis=1)
        self.log.debug(f"(rank {self.comm.rank}) done scattering arrays.")

        # construct FormedBeam container
        deltas = containers.FormedBeam(
            freq=freq, object_id=obj_id, pol=np.array(["0"]), comm=self.comm
        )
        # DELTA files use radians
        deltas["position"]["ra"] = np.degrees(pos[:, 0])
        deltas["position"]["dec"] = np.degrees(pos[:, 1])
        deltas["redshift"]["z"] = z
        deltas["redshift"]["z_error"] = 0

        # write to the distributed array
        deltas.redistribute("freq")
        deltas["beam"][:] = delta_local[:, np.newaxis, :]
        deltas["weight"][:] = weight_local[:, np.newaxis, :]

        # assign tag
        deltas.attrs["tag"] = group["tag"]

        return deltas


class BaseLoadFiles(task.SingleTask):
    """Base class for loading containers from a file on disk.

    Provides the capability to make selections along axes.

    Attributes
    ----------
    distributed : bool, optional
        Whether the file should be loaded distributed across ranks.
    convert_strings : bool, optional
        Convert strings to unicode when loading.
    selections : dict, optional
        A dictionary of axis selections. See the section below for details.
    redistribute : str, optional
        An optional axis name to redistribute the container over after it has
        been read.

    Selections
    ----------
    Selections can be given to limit the data read to specified subsets. They can be
    given for any named axis in the container.

    Selections can be given as a slice with an `<axis name>_range` key with either
    `[start, stop]` or `[start, stop, step]` as the value. Alternatively a list of
    explicit indices to extract can be given with the `<axis name>_index` key, and
    the value is a list of the indices. If both `<axis name>_range` and `<axis
    name>_index` keys are given the former will take precedence, but you should
    clearly avoid doing this.

    Additionally index based selections currently don't work for distributed reads.

    Here's an example in the YAML format that the pipeline uses:

    .. code-block:: yaml

        selections:
            freq_range: [256, 512, 4]  # A strided slice
            stack_index: [1, 2, 4, 9, 16, 25, 36, 49, 64]  # A sparse selection
            stack_range: [1, 14]  # Will override the selection above
    """

    distributed = config.Property(proptype=bool, default=True)
    convert_strings = config.Property(proptype=bool, default=True)
    selections = config.Property(proptype=dict, default=None)
    redistribute = config.Property(proptype=str, default=None)

    def setup(self):
        """Resolve the selections."""
        self._sel = self._resolve_sel()

    def _load_file(self, filename, extra_message=""):
        # Load the file into the relevant container

        if not os.path.exists(filename):
            raise RuntimeError(f"File does not exist: {filename}")

        self.log.info(f"Loading file {filename} {extra_message}")
        self.log.debug(f"Reading with selections: {self._sel}")

        # If we are applying selections we need to dispatch the `from_file` via the
        # correct subclass, rather than relying on the internal detection of the
        # subclass. To minimise the number of files being opened this is only done on
        # rank=0 and is then broadcast
        if self._sel:
            if self.comm.rank == 0:
                with fileformats.guess_file_format(filename).open(filename, "r") as fh:
                    clspath = memh5.MemDiskGroup._detect_subclass_path(fh)
            else:
                clspath = None
            clspath = self.comm.bcast(clspath, root=0)
            new_cls = memh5.MemDiskGroup._resolve_subclass(clspath)
        else:
            new_cls = memh5.BasicCont

        cont = new_cls.from_file(
            filename,
            distributed=self.distributed,
            comm=self.comm,
            convert_attribute_strings=self.convert_strings,
            convert_dataset_strings=self.convert_strings,
            **self._sel,
        )

        if self.redistribute is not None:
            cont.redistribute(self.redistribute)

        return cont

    def _resolve_sel(self):
        # Turn the selection parameters into actual selectable types

        sel = {}

        sel_parsers = {"range": self._parse_range, "index": self._parse_index}

        # To enforce the precedence of range vs index selections, we rely on the fact
        # that a sort will place the axis_range keys after axis_index keys
        for k in sorted(self.selections or []):

            # Parse the key to get the axis name and type, accounting for the fact the
            # axis name may contain an underscore
            *axis, type_ = k.split("_")
            axis_name = "_".join(axis)

            if type_ not in sel_parsers:
                raise ValueError(
                    f'Unsupported selection type "{type_}", or invalid key "{k}"'
                )

            sel[f"{axis_name}_sel"] = sel_parsers[type_](self.selections[k])

        return sel

    def _parse_range(self, x):
        # Parse and validate a range type selection

        if not isinstance(x, (list, tuple)) or len(x) > 3 or len(x) < 2:
            raise ValueError(
                f"Range spec must be a length 2 or 3 list or tuple. Got {x}."
            )

        for v in x:
            if not isinstance(v, int):
                raise ValueError(f"All elements of range spec must be ints. Got {x}")

        return slice(*x)

    def _parse_index(self, x):
        # Parse and validate an index type selection

        if not isinstance(x, (list, tuple)) or len(x) == 0:
            raise ValueError(f"Index spec must be a non-empty list or tuple. Got {x}.")

        for v in x:
            if not isinstance(v, int):
                raise ValueError(f"All elements of index spec must be ints. Got {x}")

        return list(x)


class LoadFilesFromParams(BaseLoadFiles):
    """Load data from files given in the tasks parameters.

    Attributes
    ----------
    files : glob pattern, or list
        Can either be a glob pattern, or lists of actual files.
    """

    files = config.Property(proptype=_list_or_glob)

    _file_ind = 0

    def process(self):
        """Load the given files in turn and pass on.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
        """
        # Garbage collect to workaround leaking memory from containers.
        # TODO: find actual source of leak
        import gc

        gc.collect()

        if self._file_ind == len(self.files):
            raise pipeline.PipelineStopIteration

        # Fetch and remove the first item in the list
        file_ = self.files[self._file_ind]

        # Load into a container
        nfiles_str = str(len(self.files))
        message = f"[{self._file_ind + 1: {len(nfiles_str)}}/{nfiles_str}]"
        cont = self._load_file(file_, extra_message=message)

        if "tag" not in cont.attrs:
            # Get the first part of the actual filename and use it as the tag
            tag = os.path.splitext(os.path.basename(file_))[0]

            cont.attrs["tag"] = tag

        self._file_ind += 1

        return cont


# Define alias for old code
LoadBasicCont = LoadFilesFromParams


class FindFiles(pipeline.TaskBase):
    """Take a glob or list of files specified as a parameter in the
    configuration file and pass on to other tasks.

    Parameters
    ----------
    files : list or glob
    """

    files = config.Property(proptype=_list_or_glob)

    def setup(self):
        """Return list of files specified in the parameters."""
        if not isinstance(self.files, (list, tuple)):
            raise RuntimeError("Argument must be list of files.")

        return self.files


class LoadFiles(LoadFilesFromParams):
    """Load data from files passed into the setup routine.

    File must be a serialised subclass of :class:`memh5.BasicCont`.
    """

    files = None

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """

        # Call the baseclass setup to resolve any selections
        super().setup()

        if not isinstance(files, (list, tuple)):
            raise RuntimeError(f'Argument must be list of files. Got "{files}"')

        self.files = files


class Save(pipeline.TaskBase):
    """Save out the input, and pass it on.

    Assumes that the input has a `to_hdf5` method. Appends a *tag* if there is
    a `tag` entry in the attributes, otherwise just uses a count.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)

    count = 0

    def next(self, data):
        """Write out the data file.

        Assumes it has an MPIDataset interface.

        Parameters
        ----------
        data : mpidataset.MPIDataset
            Data to write out.
        """

        if "tag" not in data.attrs:
            tag = self.count
            self.count += 1
        else:
            tag = data.attrs["tag"]

        fname = "%s_%s.h5" % (self.root, str(tag))

        data.to_hdf5(fname)

        return data


class Print(pipeline.TaskBase):
    """Stupid module which just prints whatever it gets. Good for debugging."""

    def next(self, input_):

        print(input_)

        return input_


class LoadBeamTransfer(pipeline.TaskBase):
    """Loads a beam transfer manager from disk.

    Attributes
    ----------
    product_directory : str
        Path to the saved Beam Transfer products.
    """

    product_directory = config.Property(proptype=str)

    def setup(self):
        """Load the beam transfer matrices.

        Returns
        -------
        tel : TransitTelescope
            Object describing the telescope.
        bt : BeamTransfer
            BeamTransfer manager.
        feed_info : list, optional
            Optional list providing additional information about each feed.
        """

        import os

        from drift.core import beamtransfer

        if not os.path.exists(self.product_directory):
            raise RuntimeError("BeamTransfers do not exist.")

        bt = beamtransfer.BeamTransfer(self.product_directory)

        tel = bt.telescope

        try:
            return tel, bt, tel.feeds
        except AttributeError:
            return tel, bt


class LoadProductManager(pipeline.TaskBase):
    """Loads a driftscan product manager from disk.

    Attributes
    ----------
    product_directory : str
        Path to the root of the products. This is the same as the output
        directory used by ``drift-makeproducts``.
    """

    product_directory = config.Property(proptype=str)

    def setup(self):
        """Load the beam transfer matrices.

        Returns
        -------
        manager : ProductManager
            Object describing the telescope.
        """

        import os

        from drift.core import manager

        if not os.path.exists(self.product_directory):
            raise RuntimeError("Products do not exist.")

        # Load ProductManager and Timestream
        pm = manager.ProductManager.from_config(self.product_directory)

        return pm


class Truncate(task.SingleTask):
    """Precision truncate data prior to saving with bitshuffle compression.

    If no configuration is provided, will look for preset values for the
    input container. Any properties defined in the config will override the
    presets.

    If available, each specified dataset will be truncated relative to a
    (specified) weight dataset with the truncation increasing the variance up
    to the specified maximum in `variance_increase`. If there is no specified
    weight dataset then the truncation falls back to using the
    `fixed_precision`.

    Attributes
    ----------
    dataset : dict
        Datasets to be truncated as keys. Possible values are:
        - bool : Whether or not to truncate, using default fixed precision.
        - float : Truncate to this relative precision.
        - dict : Specify values for `weight_dataset`, `fixed_precision`, `variance_increase`.
    ensure_chunked : bool
        If True, ensure datasets are chunked according to their dataset_spec.
    """

    dataset = config.Property(proptype=dict, default=None)
    ensure_chunked = config.Property(proptype=bool, default=True)

    default_params = {
        "weight_dataset": None,
        "fixed_precision": 1e-4,
        "variance_increase": 1e-3,
    }

    def _get_params(self, container, dset):
        """
        Load truncation parameters for a dataset from config or container defaults.

        Parameters
        ----------
        container
            Container class.
        dset : str
            Dataset name

        Returns
        -------
        Dict or None
            Returns `None` if the dataset shouldn't get truncated.
        """
        # Check if dataset should get truncated at all
        if (self.dataset is None) or (dset not in self.dataset):
            if dset not in container._dataset_spec or not container._dataset_spec[
                dset
            ].get("truncate", False):
                self.log.debug(f"Not truncating dataset '{dset}' in {container}.")
                return None
            # Use the dataset spec if nothing specified in config
            given_params = container._dataset_spec[dset].get("truncate", False)
        else:
            given_params = self.dataset[dset]

        # Parse config parameters
        params = self.default_params.copy()
        if type(given_params) is dict:
            params.update(given_params)
        elif type(given_params) is float:
            params["fixed_precision"] = given_params
        elif not given_params:
            self.log.debug(f"Not truncating dataset '{dset}' in {container}.")
            return None

        # Factor of 3 for variance over uniform distribution of truncation errors
        if params["variance_increase"] is not None:
            params["variance_increase"] *= 3

        return params

    def process(self, data):
        """Truncate the incoming data.

        The truncation is done *in place*.

        Parameters
        ----------
        data : containers.ContainerBase
            Data to truncate.

        Returns
        -------
        truncated_data : containers.ContainerBase
            Truncated data.

        Raises
        ------
        `caput.pipeline.PipelineRuntimeError`
            If input data has mismatching dataset and weight array shapes.
        `config.CaputConfigError`
             If the input data container has no preset values and `fixed_precision` or `variance_increase` are not set
             in the config.
        """

        if self.ensure_chunked:
            data._ensure_chunked()

        for dset in data._dataset_spec:
            # get truncation parameters from config or container defaults
            specs = self._get_params(type(data), dset)

            if (specs is None) or (dset not in data):
                # Don't truncate this dataset
                continue

            old_shape = data[dset][:].shape
            # np.ndarray.reshape must be used with ndarrays
            # MPIArrays use MPIArray.reshape()
            val = np.ndarray.reshape(data[dset][:].view(np.ndarray), data[dset][:].size)
            if specs["weight_dataset"] is None:
                if np.iscomplexobj(data[dset]):
                    data[dset][:].real = truncate.bit_truncate_relative(
                        val.real, specs["fixed_precision"]
                    ).reshape(old_shape)
                    data[dset][:].imag = truncate.bit_truncate_relative(
                        val.imag, specs["fixed_precision"]
                    ).reshape(old_shape)
                else:
                    data[dset][:] = truncate.bit_truncate_relative(
                        val, specs["fixed_precision"]
                    ).reshape(old_shape)
            else:
                invvar = (
                    np.broadcast_to(
                        data[specs["weight_dataset"]][:], data[dset][:].shape
                    )
                    .copy()
                    .reshape(-1)
                )
                invvar *= (2.0 if np.iscomplexobj(data[dset]) else 1.0) / specs[
                    "variance_increase"
                ]
                if np.iscomplexobj(data[dset]):
                    data[dset][:].real = truncate.bit_truncate_weights(
                        val.real,
                        invvar,
                        specs["fixed_precision"],
                    ).reshape(old_shape)
                    data[dset][:].imag = truncate.bit_truncate_weights(
                        val.imag,
                        invvar,
                        specs["fixed_precision"],
                    ).reshape(old_shape)
                else:
                    data[dset][:] = truncate.bit_truncate_weights(
                        val,
                        invvar,
                        specs["fixed_precision"],
                    ).reshape(old_shape)

        return data


class ZipZarrContainers(task.SingleTask):
    """Zip up a Zarr container into a single file.

    This is useful to save on file quota and speed up IO by combining the chunk
    data into a single file. Note that the file cannot really be updated after
    this process has been performed.

    As this process is IO limited in most cases, it will attempt to parallelise
    the compression across different distinct nodes. That means at most only
    one rank per node will participate.

    Attributes
    ----------
    containers : list
        The names of the Zarr containers to compress. The zipped files will
        have the same names with `.zip` appended.
    remove : bool
        Remove the original data when finished. Defaults to True.
    """

    containers = config.Property(proptype=list)
    remove = config.Property(proptype=bool, default=True)

    _host_rank = None

    def setup(self, _=None):
        """Setup the task.

        This routine does nothing at all with the input, but it means the
        process won't run until the (optional) requirement is received. This
        can be used to delay evaluation until you know that all the files are
        available.
        """
        import socket

        # See if we can find 7z
        path_7z = shutil.which("7z")
        if path_7z is None:
            raise RuntimeError("Could not find 7z on the PATH")
        self._path_7z = path_7z

        # Get the rank -> hostname mapping for all ranks
        my_host = socket.gethostname()
        my_rank = self.comm.rank
        all_host_ranks = self.comm.allgather((my_host, my_rank))

        # Identify the lowest rank running on each node
        unique_hosts = {}
        for host, rank in all_host_ranks:
            if host not in unique_hosts:
                unique_hosts[host] = rank
            else:
                if unique_hosts[host] > rank:
                    unique_hosts[host] = rank

        self._num_hosts = len(unique_hosts)

        # Figure out if this rank is one that needs to do anything
        if unique_hosts[my_host] != my_rank:
            # This is not the lowest rank on the host, so we don't do anything
            self._host_rank = None
        else:
            # This is the lowest rank, so find where we are in the sorted list of all hosts
            self._host_rank = sorted(unique_hosts).index(my_host)
            self.log.debug(f"Lowest rank on {my_host}")

    def process(self):
        """Compress the listed zarr containers.

        Only the lowest rank on each node will participate.
        """

        if self._host_rank is not None:
            import subprocess

            # Get the set of containers this rank is responsible for compressing
            my_containers = self.containers[self._host_rank :: self._num_hosts]

            for container in my_containers:

                self.log.info(f"Zipping {container}")

                if not container.endswith(".zarr") or not os.path.isdir(container):
                    raise ValueError(f"{container} is not a valid .zarr directory")

                # Run 7z to zip up the file
                dest_file = container + ".zip"
                src_dir = container + "/."
                command = [self._path_7z, "a", "-tzip", "-mx=0", dest_file, src_dir]
                status = subprocess.run(command, capture_output=True)

                if status.returncode != 0:
                    self.log.debug("Error occurred while zipping. Debug logs follow...")
                    self.log.debug(f"stdout={status.stdout}")
                    self.log.debug(f"stderr={status.stderr}")
                    raise RuntimeError(f"Error occurred while zipping {container}.")

                self.log.info(f"Done zipping. Generated {dest_file}.")

                if self.remove:
                    shutil.rmtree(container)
                    self.log.info(f"Removed original container {container}.")

        self.comm.Barrier()

        raise pipeline.PipelineStopIteration


class SaveModuleVersions(task.SingleTask):
    """Write module versions to a YAML file.

    The list of modules should be added to the configuration under key 'save_versions'.
    The version strings are written to a YAML file.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)

    done = True

    def setup(self):
        """Save module versions."""

        fname = "{}_versions.yml".format(self.root)
        f = open(fname, "w")
        f.write(yamldump(self.versions))
        f.close()
        self.done = True

    def process(self):
        """Do nothing."""
        self.done = True
        return


class SaveConfig(task.SingleTask):
    """Write pipeline config to a text file.

    Yaml configuration document is written to a text file.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)
    done = True

    def setup(self):
        """Save module versions."""

        fname = "{}_config.yml".format(self.root)
        f = open(fname, "w")
        f.write(yamldump(self.pipeline_config))
        f.close()
        self.done = True

    def process(self):
        """Do nothing."""
        self.done = True
        return


# Python types for objects convertible to beamtransfers or telescope instances
BeamTransferConvertible = Union[manager.ProductManager, beamtransfer.BeamTransfer]
TelescopeConvertible = Union[BeamTransferConvertible, telescope.TransitTelescope]


def get_telescope(obj):
    """Return a telescope object out of the input (either `ProductManager`,
    `BeamTransfer` or `TransitTelescope`).
    """
    try:
        return get_beamtransfer(obj).telescope
    except RuntimeError:
        if isinstance(obj, telescope.TransitTelescope):
            return obj

    raise RuntimeError("Could not get telescope instance out of %s" % repr(obj))


def get_beamtransfer(obj):
    """Return a BeamTransfer object out of the input (either `ProductManager`,
    `BeamTransfer`).
    """
    if isinstance(obj, beamtransfer.BeamTransfer):
        return obj

    if isinstance(obj, manager.ProductManager):
        return obj.beamtransfer

    raise RuntimeError("Could not get BeamTransfer instance out of %s" % repr(obj))

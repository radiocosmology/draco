"""Tasks for reading and writing data.

Tasks
=====

.. autosummary::
    :toctree:

    LoadFiles
    LoadMaps
    LoadFilesFromParams
    Save
    Print
    LoadBeamTransfer

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
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
from past.builtins import basestring

# === End Python 2/3 compatibility

import os.path
import numpy as np
from yaml import dump as yamldump

from caput import pipeline
from caput import config

from . import task
from ..util.truncate import bit_truncate_weights, bit_truncate_fixed
from .containers import SiderealStream, TimeStream, TrackBeam


TRUNC_SPEC = {
    SiderealStream: {
        "dataset": ["vis", "vis_weight"],
        "weight_dataset": ["vis_weight", None],
        "fixed_precision": 1e-4,
        "variance_increase": 1e-3,
    },
    TimeStream: {
        "dataset": ["vis", "vis_weight"],
        "weight_dataset": ["vis_weight", None],
        "fixed_precision": 1e-4,
        "variance_increase": 1e-3,
    },
    TrackBeam: {
        "dataset": ["beam", "weight"],
        "weight_dataset": ["weight", None],
        "fixed_precision": 1e-4,
        "variance_increase": 1e-3,
    },
}


def _list_of_filelists(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    f2 = []

    for filelist in files:

        if isinstance(filelist, basestring):
            filelist = glob.glob(filelist)
        elif isinstance(filelist, list):
            pass
        else:
            raise Exception("Must be list or glob pattern.")
        f2.append(filelist)

    return f2


def _list_or_glob(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    if isinstance(files, basestring):
        files = sorted(glob.glob(files))
    elif isinstance(files, list):
        pass
    else:
        raise ValueError("Argument must be list or glob pattern, got %s" % repr(files))

    return files


def _list_of_filegroups(groups):
    # Process a file group/groups
    import glob

    # Convert to list if the group was not included in a list
    if not isinstance(groups, list):
        groups = [groups]

    # Iterate over groups, set the tag if needed, and process the file list
    # through glob
    for gi, group in enumerate(groups):

        files = group["files"]

        if "tag" not in group:
            group["tag"] = "group_%i" % gi

        flist = []

        for fname in files:
            flist += glob.glob(fname)

        if not len(flist):
            raise RuntimeError("No files in group exist (%s)." % files)

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


class LoadFilesFromParams(task.SingleTask):
    """Load data from files given in the tasks parameters.

    Attributes
    ----------
    files : glob pattern, or list
        Can either be a glob pattern, or lists of actual files.
    distributed : bool, optional
        Whether the file should be loaded distributed across ranks.
    convert_strings : bool. optional
        Convert strings to unicode when loading.
    """

    files = config.Property(proptype=_list_or_glob)
    distributed = config.Property(proptype=bool, default=True)
    convert_strings = config.Property(proptype=bool, default=True)

    def process(self):
        """Load the given files in turn and pass on.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
        """

        from caput import memh5

        # Garbage collect to workaround leaking memory from containers.
        # TODO: find actual source of leak
        import gc

        gc.collect()

        if len(self.files) == 0:
            raise pipeline.PipelineStopIteration

        # Fetch and remove the first item in the list
        file_ = self.files.pop(0)

        self.log.info("Loading file %s" % file_)

        cont = memh5.BasicCont.from_file(
            file_,
            distributed=self.distributed,
            comm=self.comm,
            convert_attribute_strings=self.convert_strings,
            convert_dataset_strings=self.convert_strings,
        )

        if "tag" not in cont.attrs:
            # Get the first part of the actual filename and use it as the tag
            tag = os.path.splitext(os.path.basename(file_))[0]

            cont.attrs["tag"] = tag

        return cont


# Define alias for old code
LoadBasicCont = LoadFilesFromParams


class FindFiles(pipeline.TaskBase):
    """ Take a glob or list of files specified as a parameter in the
    configuration file and pass on to other tasks.

    Parameters
    ----------
    files : list or glob
    """

    files = config.Property(proptype=_list_or_glob)

    def setup(self):
        """ Return list of files specified in the parameters.
        """
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
        if not isinstance(files, (list, tuple)):
            raise RuntimeError("Argument must be list of files.")

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
    """Stupid module which just prints whatever it gets. Good for debugging.
    """

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
    dataset : list of str
        Datasets to truncate.
    weight_dataset : list of str
        Datasets to use as inverse variance for truncation precision.
    fixed_precision : float
        Relative precision to truncate to (default 1e-4).
    variance_increase : float
        Maximum fractional increase in variance from numerical truncation.
    """

    dataset = config.Property(proptype=list, default=None)
    weight_dataset = config.Property(proptype=list, default=None)
    fixed_precision = config.Property(proptype=float, default=None)
    variance_increase = config.Property(proptype=float, default=None)

    def _get_params(self, container):
        """Load truncation parameters from config or container defaults."""
        if container in TRUNC_SPEC:
            self.log.info("Truncating from preset for container {}".format(container))
            for key in [
                "dataset",
                "weight_dataset",
                "fixed_precision",
                "variance_increase",
            ]:
                attr = getattr(self, key)
                if attr is None:
                    setattr(self, key, TRUNC_SPEC[container][key])
                else:
                    self.log.info("Overriding container default for '{}'.".format(key))
        else:
            if (
                self.dataset is None
                or self.fixed_precision is None
                or self.variance_increase is None
            ):
                raise pipeline.PipelineConfigError(
                    "Container {} has no preset values. You must define all of 'dataset', "
                    "'fixed_precision', and 'variance_increase' properties.".format(
                        container
                    )
                )
        # Factor of 3 for variance over uniform distribution of truncation errors
        self.variance_increase *= 3

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
        """
        # get truncation parameters from config or container defaults
        self._get_params(type(data))

        if self.weight_dataset is None:
            self.weight_dataset = [None] * len(self.dataset)

        for dset, wgt in zip(self.dataset, self.weight_dataset):
            old_shape = data[dset].local_shape
            val = np.ndarray.reshape(data[dset][:], data[dset][:].size)
            if wgt is None:
                if np.iscomplexobj(data[dset]):
                    data[dset][:].real = bit_truncate_fixed(
                        val.real, self.fixed_precision
                    ).reshape(old_shape)
                    data[dset][:].imag = bit_truncate_fixed(
                        val.imag, self.fixed_precision
                    ).reshape(old_shape)
                else:
                    data[dset][:] = bit_truncate_fixed(
                        val, self.fixed_precision
                    ).reshape(old_shape)
            else:
                if data[dset][:].shape != data[wgt][:].shape:
                    raise pipeline.PipelineRuntimeError(
                        "Dataset and weight arrays must have same shape ({} != {})".format(
                            data[dset].shape, data[wgt].shape
                        )
                    )
                invvar = np.ndarray.reshape(data[wgt][:], data[dset][:].size)
                if np.iscomplexobj(data[dset]):
                    data[dset][:].real = bit_truncate_weights(
                        val.real,
                        invvar * 2.0 / self.variance_increase,
                        self.fixed_precision,
                    ).reshape(old_shape)
                    data[dset][:].imag = bit_truncate_weights(
                        val.imag,
                        invvar * 2.0 / self.variance_increase,
                        self.fixed_precision,
                    ).reshape(old_shape)
                else:
                    data[dset][:] = bit_truncate_weights(
                        val, invvar / self.variance_increase, self.fixed_precision
                    ).reshape(old_shape)

        return data


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


def get_telescope(obj):
    """Return a telescope object out of the input (either `ProductManager`,
    `BeamTransfer` or `TransitTelescope`).
    """
    from drift.core import telescope

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
    from drift.core import manager, beamtransfer

    if isinstance(obj, beamtransfer.BeamTransfer):
        return obj

    if isinstance(obj, manager.ProductManager):
        return obj.beamtransfer

    raise RuntimeError("Could not get BeamTransfer instance out of %s" % repr(obj))

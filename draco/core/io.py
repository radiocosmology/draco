"""Tasks for reading and writing specific container types."""

import numpy as np
from caput import config, pipeline, task, units
from caput.task.io import list_of_filegroups
from drift.core import beamtransfer, manager, telescope


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

    maps = config.Property(proptype=list_of_filegroups)

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

    catalogs = config.Property(proptype=list_of_filegroups)
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
        return manager.ProductManager.from_config(self.product_directory)


# Python types for objects convertible to beamtransfers or telescope instances
BeamTransferConvertible = manager.ProductManager | beamtransfer.BeamTransfer
TelescopeConvertible = BeamTransferConvertible | telescope.TransitTelescope


def get_telescope(obj):
    """Return a telescope object out of the input.

    Either `ProductManager`, `BeamTransfer`, or `TransitTelescope`.
    """
    try:
        return get_beamtransfer(obj).telescope
    except RuntimeError:
        if isinstance(obj, telescope.TransitTelescope):
            return obj

    raise RuntimeError(f"Could not get telescope instance out of {obj!r}")


def get_beamtransfer(obj):
    """Return a BeamTransfer object out of the input.

    Either `ProductManager` or `BeamTransfer`.
    """
    if isinstance(obj, beamtransfer.BeamTransfer):
        return obj

    if isinstance(obj, manager.ProductManager):
        return obj.beamtransfer

    raise RuntimeError(f"Could not get BeamTransfer instance out of {obj!r}")

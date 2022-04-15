"""Tasks for foreground filtering data."""


import numpy as np

from caput import config
from ..core import task, containers, io


class _ProjectFilterBase(task.SingleTask):
    """A base class for projecting data to/from a different basis.

    Attributes
    ----------
    mode : string
        Which projection to perform. Into the new basis (forward), out of the
        new basis (backward), and forward then backward in order to filter the
        data through the basis (filter).
    """

    mode = config.enum(["forward", "backward", "filter"], default="forward")

    def process(self, inp):
        """Project or filter the input data.

        Parameters
        ----------
        inp : memh5.BasicCont
            Data to process.

        Returns
        -------
        output : memh5.BasicCont
        """

        if self.mode == "forward":
            return self._forward(inp)

        if self.mode == "backward":
            return self._backward(inp)

        if self.mode == "filter":
            return self._backward(self._forward(inp))

    def _forward(self, inp):
        pass

    def _backward(self, inp):
        pass


class SVDModeProject(_ProjectFilterBase):
    """SVD projection between the raw m-modes and the reduced degrees of freedom.

    Note that this produces the packed SVD modes, with the modes from each
    frequency concatenated.

    Attributes
    ----------
    svcut : float, optional
        The relative precision below the maximum singular value (at each m separately)
        to exclude low-sensitivity SVD modes. If not specified, the value set in the
        BeamTransfer instance is used. Default: None.
    """

    svcut = config.Property(proptype=float, default=None)

    def setup(self, bt):
        """Set the beamtransfer instance.

        Parameters
        ----------
        bt : BeamTransfer
            This can also take a ProductManager instance.
        """
        self.beamtransfer = io.get_beamtransfer(bt)
        if self.svcut is None:
            self.svcut = self.beamtransfer.svcut

    def _forward(self, mmodes):
        # Forward transform into SVD basis

        bt = self.beamtransfer
        tel = bt.telescope

        svdmodes = containers.SVDModes(
            mode=bt.ndofmax(), axes_from=mmodes, attrs_from=mmodes, svcut=self.svcut
        )
        svdmodes.vis[:] = 0.0

        mmodes.redistribute("m")
        svdmodes.redistribute("m")

        # Iterate over local m's, project mode and save to disk.
        for lm, mi in mmodes.vis[:].enumerate(axis=0):

            tm = mmodes.vis[mi].transpose((1, 0, 2)).reshape(tel.nfreq, -1)
            svdm = bt.project_vector_telescope_to_svd(mi, tm, svcut=self.svcut)

            svdmodes.nmode[mi] = len(svdm)
            svdmodes.vis[mi, : svdmodes.nmode[mi]] = svdm

            # TODO: apply transform correctly to weights. For now just crudely
            # transfer over the weights, only really good for determining
            # whether an m-mode should be masked comoletely
            svdmodes.weight[mi] = np.median(mmodes.weight[mi])

        # Save input and stack axes from input m-modes, so that they can be restored
        # after a backwards SVD projection (prodmap can be restored purely from input
        # axis, plus it can run into HDF5 attribute size limits if we try to save it)
        svdmodes.attrs["input"] = mmodes.input
        svdmodes.attrs["stack"] = mmodes.stack

        return svdmodes

    def _backward(self, svdmodes):
        # Backward transform from SVD basis into the m-modes

        bt = self.beamtransfer
        tel = bt.telescope

        # Try to fetch the feed index and info from the SVDModes container, them from
        # the telescope object.
        try:
            input = svdmodes.attrs["input"]
        except:
            try:
                input = tel.input_index
            except AttributeError:
                input = tel.nfeed

        # Try to fetch the stack info from the SVDModes container, then from the
        # telescope object
        try:
            stack = svdmodes.attrs["stack"]
        except:
            stack = None

        # Construct frequency index map
        freqmap = np.zeros(
            len(tel.frequencies), dtype=[("centre", np.float64), ("width", np.float64)]
        )
        freqmap["centre"][:] = tel.frequencies
        freqmap["width"][:] = np.abs(np.diff(tel.frequencies)[0])

        # Verify that input SVDModes container has svcut attribute
        if svdmodes.svcut is None:
            raise AttributeError("Input SVDModes container is missing svcut attribute!")

        # Construct the new m-mode container
        mmodes = containers.MModes(
            freq=freqmap,
            input=input,
            stack=stack,
            attrs_from=svdmodes,
            axes_from=svdmodes,
        )
        mmodes.redistribute("m")
        svdmodes.redistribute("m")

        # Iterate over local m's, project mode and save to disk.
        for lm, mi in mmodes.vis[:].enumerate(axis=0):

            svdm = svdmodes.vis[mi]
            tm = bt.project_vector_svd_to_telescope(mi, svdm, svcut=svdmodes.svcut)

            svdmodes.nmode[mi] = len(svdm)
            mmodes.vis[mi] = tm.transpose((1, 0, 2))

            # TODO: apply transform correctly to weights. For now just crudely
            # transfer over the weights, only really good for determining
            # whether an m-mode should be masked comoletely
            mmodes.weight[mi] = np.median(svdmodes.weight[mi])

        return mmodes


class KLModeProject(_ProjectFilterBase):
    """Project between the SVD and KL basis.

    Attributes
    ----------
    threshold : float, optional
        KL mode threshold.
    klname : str
        Name of filter to use.
    mode : string
        Which projection to perform. Into the KL basis (forward), out of the
        KL basis (backward), and forward then backward in order to KL filter the
        data through the basis (filter).
    """

    threshold = config.Property(proptype=float, default=None)
    klname = config.Property(proptype=str)

    def setup(self, manager):
        """Set the product manager that holds the saved KL modes."""
        self.product_manager = manager

    def _forward(self, svdmodes):
        # Forward transform into the KL modes

        bt = self.product_manager.beamtransfer

        # Check and set the KL basis we are using
        if self.klname not in self.product_manager.kltransforms:
            raise RuntimeError(
                "Requested KL basis %s not available (options are %s)"
                % (self.klname, repr(list(self.product_manager.kltransforms.items())))
            )
        kl = self.product_manager.kltransforms[self.klname]

        # Construct the container and redistribute
        klmodes = containers.KLModes(
            mode=bt.ndofmax(), axes_from=svdmodes, attrs_from=svdmodes
        )

        klmodes.vis[:] = 0.0

        klmodes.redistribute("m")
        svdmodes.redistribute("m")

        # Iterate over local m's and project mode into KL basis
        for lm, mi in svdmodes.vis[:].enumerate(axis=0):

            sm = svdmodes.vis[mi][: svdmodes.nmode[mi]]
            klm = kl.project_vector_svd_to_kl(mi, sm, threshold=self.threshold)

            klmodes.nmode[mi] = len(klm)
            klmodes.vis[mi, : klmodes.nmode[mi]] = klm

            # TODO: apply transform correctly to weights. For now just crudely
            # transfer over the weights, only really good for determining
            # whether an m-mode should be masked comoletely
            klmodes.weight[mi] = np.median(svdmodes.weight[mi])

        return klmodes

    def _backward(self, klmodes):
        # Backward transform from the KL modes into the SVD modes

        bt = self.product_manager.beamtransfer

        # Check and set the KL basis we are using
        if self.klname not in self.product_manager.kltransforms:
            raise RuntimeError(
                "Requested KL basis %s not available (options are %s)"
                % (self.klname, repr(list(self.product_manager.kltransforms.items())))
            )
        kl = self.product_manager.kltransforms[self.klname]

        # Construct the container and redistribute

        svdmodes = containers.SVDModes(
            mode=bt.ndofmax(), axes_from=klmodes, attrs_from=klmodes
        )
        klmodes.redistribute("m")
        svdmodes.redistribute("m")

        # Iterate over local m's and project mode into KL basis
        for lm, mi in klmodes.vis[:].enumerate(axis=0):

            klm = klmodes.vis[mi][: klmodes.nmode[mi]]
            sm = kl.project_vector_kl_to_svd(mi, klm, threshold=self.threshold)

            svdmodes.nmode[mi] = len(sm)
            svdmodes.vis[mi, : svdmodes.nmode[mi]] = sm

            # TODO: apply transform correctly to weights. For now just crudely
            # transfer over the weights, only really good for determining
            # whether an m-mode should be masked comoletely
            svdmodes.weight[mi] = np.median(klmodes.weight[mi])

        return svdmodes

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
    """

    def setup(self, bt):
        """Set the beamtransfer instance.

        Parameters
        ----------
        bt : BeamTransfer
            This can also take a ProductManager instance.
        """
        self.beamtransfer = io.get_beamtransfer(bt)

    def _forward(self, mmodes):
        # Forward transform into SVD basis

        bt = self.beamtransfer
        tel = bt.telescope

        svdmodes = containers.SVDModes(
            mode=bt.ndofmax, axes_from=mmodes, attrs_from=mmodes
        )
        svdmodes.vis[:] = 0.0

        mmodes.redistribute("m")
        svdmodes.redistribute("m")

        # Iterate over local m's, project mode and save to disk.
        for lm, mi in mmodes.vis[:].enumerate(axis=0):
            tm = mmodes.vis[mi].transpose((1, 0, 2)).reshape(tel.nfreq, 2 * tel.npairs)
            svdm = bt.project_vector_telescope_to_svd(mi, tm)

            svdmodes.nmode[mi] = len(svdm)
            svdmodes.vis[mi, : svdmodes.nmode[mi]] = svdm

            # TODO: apply transform correctly to weights. For now just crudely
            # transfer over the weights, only really good for determining
            # whether an m-mode should be masked comoletely
            svdmodes.weight[mi] = np.median(mmodes.weight[mi])

        return svdmodes

    def _backward(self, svdmodes):
        # Backward transform from SVD basis into the m-modes

        bt = self.beamtransfer
        tel = bt.telescope

        # Try and fetch out the feed index and info from the telescope object.
        try:
            feed_index = tel.input_index
        except AttributeError:
            feed_index = tel.nfeed

        # Construct frequency index map
        freqmap = np.zeros(
            len(tel.frequencies), dtype=[("centre", np.float64), ("width", np.float64)]
        )
        freqmap["centre"][:] = tel.frequencies
        freqmap["width"][:] = np.abs(np.diff(tel.frequencies)[0])

        # Construct the new m-mode container
        mmodes = containers.MModes(
            freq=freqmap,
            prod=tel.uniquepairs,
            input=feed_index,
            attrs_from=svdmodes,
            axes_from=svdmodes,
        )
        mmodes.redistribute("m")
        svdmodes.redistribute("m")

        # Iterate over local m's, project mode and save to disk.
        for lm, mi in mmodes.vis[:].enumerate(axis=0):
            svdm = svdmodes.vis[mi]
            tm = bt.project_vector_svd_to_telescope(mi, svdm)

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
            mode=bt.ndofmax, axes_from=svdmodes, attrs_from=svdmodes
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
            mode=bt.ndofmax, axes_from=klmodes, attrs_from=klmodes
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

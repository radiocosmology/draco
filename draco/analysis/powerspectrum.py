"""Power spectrum estimation code."""

import numpy as np
from caput import config

from ..core import containers, task


class QuadraticPSEstimation(task.SingleTask):
    """Estimate a power spectrum from a set of KLModes.

    Attributes
    ----------
    psname : str
        Name of power spectrum to use. Must be precalculated in the driftscan
        products.
    pstype : str
        Type of power spectrum estimate to calculate. One of 'unwindowed',
        'minimum_variance' or 'uncorrelated'.
    """

    psname = config.Property(proptype=str)

    pstype = config.enum(
        ["unwindowed", "minimum_variance", "uncorrelated"], default="unwindowed"
    )

    def setup(self, manager):
        """Set the ProductManager instance to use.

        Parameters
        ----------
        manager : ProductManager
            Manager object to use
        """
        self.manager = manager

    def process(self, klmodes):
        """Estimate the power spectrum from the given data.

        Parameters
        ----------
        klmodes : containers.KLModes
            KLModes for which to estimate the power spectrum

        Returns
        -------
        ps : containers.PowerSpectrum
        """
        import scipy.linalg as la

        if not isinstance(klmodes, containers.KLModes):
            raise ValueError(
                "Input container must be instance of "
                f"KLModes (received {klmodes.__class__!s})"
            )

        klmodes.redistribute("m")

        pse = self.manager.psestimators[self.psname]
        pse.genbands()

        q_list = []

        for mi, m in klmodes.vis[:].enumerate(axis=0):
            ps_single = pse.q_estimator(m, klmodes.vis[m, : klmodes.nmode[m]])
            q_list.append(ps_single)

        q = klmodes.comm.allgather(np.array(q_list).sum(axis=0))
        q = np.array(q).sum(axis=0)

        # reading from directory
        fisher, bias = pse.fisher_bias()

        ps = containers.Powerspectrum2D(
            kperp_edges=pse.kperp_bands, kpar_edges=pse.kpar_bands
        )

        npar = len(ps.index_map["kpar"])
        nperp = len(ps.index_map["kperp"])

        # Calculate the right unmixing matrix for each ps type
        if self.pstype == "unwindowed":
            M = la.pinv(fisher, rcond=1e-8)
        elif self.pstype == "uncorrelated":
            Fh = la.cholesky(fisher)
            M = la.inv(Fh) / Fh.sum(axis=1)[:, np.newaxis]
        elif self.pstype == "minimum_variance":
            M = np.diag(fisher.sum(axis=1) ** -1)

        ps.powerspectrum[:] = np.dot(M, q - bias).reshape(nperp, npar)
        ps.C_inv[:] = fisher.reshape(nperp, npar, nperp, npar)

        return ps

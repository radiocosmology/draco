"""Power spectrum estimation code."""


import numpy as np
import time
from mpi4py import MPI

from caput import config, mpiarray, mpiutil
from ..core import task, containers


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
        """
        self.manager = manager

    def process(self, klmodes):
        """Estimate the power spectrum from the given data.

        Parameters
        ----------
        klmodes : containers.KLModes

        Returns
        -------
        ps : containers.PowerSpectrum
        """

        import scipy.linalg as la

        if not isinstance(klmodes, containers.KLModes):
            raise ValueError(
                "Input container must be instance of "
                "KLModes (received %s)" % klmodes.__class__
            )

        klmodes.redistribute("m")

        pse = self.manager.psestimators[self.psname]
        pse.genbands()

        q_list = []

        st = time.time()

        for mi, m in klmodes.vis[:].enumerate(axis=0):
            ps_single = pse.q_estimator(m, klmodes.vis[m, : klmodes.nmode[m]])
            q_list.append(ps_single)

        q = klmodes.comm.allgather(np.array(q_list).sum(axis=0))
        q = np.array(q).sum(axis=0)

        et = time.time()
        if mpiutil.rank0:
            print("m, time needed for quadratic pse", m, et - st)

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


class QuadraticPSEstimationLarge(QuadraticPSEstimation):
    """Quadratic PS estimation for large arrays"""

    def process(self, klmodes):
        """Estimate the power spectrum from the given data.

        Parameters
        ----------
        klmodes : containers.KLModes

        Returns
        -------
        ps : containers.PowerSpectrum
        """

        import scipy.linalg as la

        if not isinstance(klmodes, containers.KLModes):
            raise ValueError(
                "Input container must be instance of "
                "KLModes (received %s)" % klmodes.__class__
            )

        klmodes.redistribute("m")

        pse = self.manager.psestimators[self.psname]
        pse.genbands()

        tel = self.manager.telescope

        self.sky_array = mpiarray.MPIArray(
            (tel.mmax + 1, tel.nfreq, tel.lmax + 1),
            axis=0,
            comm=MPI.COMM_WORLD,
            dtype=np.complex128,
        )

        self.sky_array[:] = 0.0

        for mi, m in klmodes.vis[:].enumerate(axis=0):
            sky_vec1 = pse.project_vector_kl_to_sky(
                m,
                klmodes.vis[m, :klmodes.nmode[m]]
            )
            if sky_vec1.shape[0] != 0:
                self.sky_array[mi] = sky_vec1

        self.sky_array = self.sky_array.allgather()

        pse.qa = mpiarray.MPIArray(
            (tel.mmax + 1, pse.nbands, 1), axis=1, comm=MPI.COMM_WORLD, dtype=np.float64
        )

        pse.qa[:] = 0.0

        for m in range(tel.mmax + 1):
            if np.sum(self.sky_array[m]) != 0:
                sky_m = self.sky_array[m].reshape((tel.nfreq, tel.lmax + 1, 1))
                pse.q_estimator(m, sky_m)

        # Redistribute over m's and have bands not distributed
        pse.qa = pse.qa.redistribute(axis=0)
        # Sum over local m's for all bands
        q_local = np.sum(np.array(pse.qa[:]), axis=0)
        # Collect all m on each rank
        q = mpiutil.allreduce(q_local[:, 0], op=MPI.SUM)

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

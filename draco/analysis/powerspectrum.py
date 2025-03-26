"""Power spectrum estimation code."""

import numpy as np
from caput import config
from caput.tools import invert_no_zero

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


class EstimateVariance(task.SingleTask):

    axis = config.Property(proptype=str, default="freq")
    dataset = config.Property(proptype=str, default="map")
    expand_pol = config.Property(proptype=bool, default=True)

    def process(self, data0)#, data1):

        data1 = None

        # make sure we are not distributed over the axis to measure variance on
        if data0.distributed and data0[self.dataset].distributed_axis != self.axis:
            data0.redistribute(np.argmax(data0[self.dataset].shape))

        # get a reference to the first dataset and its weights
        m0 = data0[self.dataset][:].local_array
        w0 = data0.weight[:].local_array

        if data1 is None:
            m1, w1 = m0, w0
        else:
            if data1.distributed and data1[self.dataset].distributed_axis != self.axis:
                data1.redistribute(np.argmax(data0[self.dataset].shape))
            m1 = data1[self.dataset][:].local_array
            w1 = data1.weight[:].local_array

        if m0.shape != w0.shape:
            # hopefully this works in most cases...
            m0, m1 = np.squeeze(m0), np.squeeze(m1)

        # get axis index to take sums over
        ax = list(data0.weight.attrs["axis"]).index(self.axis)

        # first estimate the mean map in each partition
        mu0, mu1 = np.sum(m0 * w0, axis=ax) * invert_no_zero(np.sum(w0, axis=ax)), np.sum(m1 * w1, axis=ax) * invert_no_zero(np.sum(w1, axis=ax))
        # then estimate the cross-variance
        wx = np.sqrt(w0 * w1)
        wx *= np.expand_dims(invert_no_zero(np.sum(wx, axis=ax)), axis=ax)
        var = np.sum(m0 * m1 * wx, axis=ax) - np.sum(m0 * wx, axis=ax) * mu1 - np.sum(m1 * wx, axis=ax) * mu0 + mu0 * mu1

        pax = None
        if self.expand_pol:
            try:
                pax = list(data0.weight.attrs["axis"]).index("pol")
            except ValueError:
                self.log.warning("Estimating the variance across polarisations was requested but there is no polarisation axis. Skipping.")
        if pax is not None:
            # reverse the pol axis to take cross-variance
            rs = [slice(None, None, None)] * len(w0.shape)
            rs[pax] = slice(None, None, -1)
            rs = tuple(rs)
            rs_nof = rs[:ax] + rs[ax+1:]

            # compute the cross-pol variance
            wx = np.sqrt(w0 * w1[rs])
            wx *= np.expand_dims(invert_no_zero(np.sum(wx, axis=ax)), axis=ax)
            var_xpol = np.sum(m0 * m1[rs] * wx, axis=ax) - np.sum(m0 * wx, axis=ax) * mu1[rs_nof] - np.sum(m1[rs] * wx, axis=ax) * mu0 + mu0 * mu1[rs_nof]

        # save both polarisations separately
        new_ax = {self.axis: np.array([np.mean(data0.freq)])}
        if pax is not None:
            new_ax["pol"] = np.array([p + p for p in data0.pol] + [data0.pol[0] + data0.pol[1], data0.pol[1] + data0.pol[0]])
        new_cont = containers.empty_like(data0, **new_ax)
        if pax is not None:
            new_cont[self.dataset][:] = np.concatenate((var, var_xpol), axis=pax).reshape(new_cont[self.dataset].local_shape)
        else:
            new_cont[self.dataset][:] = var.reshape(new_cont[self.dataset].local_shape)

        return new_cont

"""Routines to estimate systematics like crosstalk and gains."""

import numpy as np
from caput import config

from ..core import containers, task


class ExtractCrosstalkGain(task.SingleTask):
    """Estimate the cross talk and gain relative to an input stack.

    Attributes
    ----------
    nra : int
        The number of output RA bins.
    """

    nra = config.Property(proptype=int, default=64)

    def setup(self, sstack: containers.SiderealStream):
        """Set the reference sidereal stack."""
        self.sstack = sstack

        if len(sstack.ra) % self.nra != 0:
            raise RuntimeError(
                f"The number of RA bins ({self.nra}) must evenly divide the stack "
                f"length ({len(sstack.ra)})."
            )
        self.sstack.redistribute("freq")

    def process(self, ss: containers.SiderealStream) -> containers.VisCrosstalkGain:
        """For each input stream assess the cross talk and gain."""
        if np.any(self.sstack.ra != ss.ra):
            raise RuntimeError(
                "The RA bins in the input must match those in the stack."
            )

        if ss.vis[:].shape != self.sstack.vis[:].shape:
            raise RuntimeError(
                "The shape of the input vis dataset must match that in the stack."
            )

        ss.redistribute("freq")

        # Figure out which container to output to
        if isinstance(ss, containers.VisGridStream):
            outcont = containers.VisCrosstalkGainGrid
            slshape = (ss.vis.shape[0], ss.vis.shape[2], ss.vis.shape[3])
            freq_axis = 1
        else:
            outcont = containers.VisCrosstalkGain
            slshape = (ss.vis.shape[1],)
            freq_axis = 0

        ra_bin = ss.ra.reshape(-1, len(ss.ra) // self.nra).mean(axis=-1)

        ss_db = outcont(ra=ra_bin, axes_from=ss, attrs_from=ss)

        ssiv = ss.vis[:].local_array
        ssiw = ss.weight[:].local_array

        sssv = self.sstack.vis[:].local_array

        ssbg = ss_db["gain"][:].local_array
        ssbx = ss_db["crosstalk"][:].local_array
        ssbgw = ss_db["gain_weight"][:].local_array
        ssbxw = ss_db["crosstalk_weight"][:].local_array

        for lfi in range(ssiv.shape[freq_axis]):

            fslice = (slice(None),) * freq_axis + (lfi,)
            newshape = (*slshape, self.nra, -1)

            # Construct the dirty estimator for the gain and cross talk
            gx_dirty = np.zeros((*slshape, self.nra, 2), dtype=np.complex64)
            gx_dirty[..., 0] = (
                (sssv[fslice].conj() * ssiw[fslice] * ssiv[fslice])
                .reshape(newshape)
                .sum(axis=-1)
            )
            gx_dirty[..., 1] = (
                (ssiw[fslice] * ssiv[fslice]).reshape(newshape).sum(axis=-1)
            )

            # Construct the inverse covariance
            mNi = np.zeros((*slshape, self.nra, 2, 2), dtype=np.complex64)
            mNi[..., 0, 0] = (
                (np.abs(sssv[fslice]) ** 2 * ssiw[fslice])
                .reshape(newshape)
                .sum(axis=-1)
            )
            mNi[..., 0, 1] = (
                (sssv[fslice].conj() * ssiw[fslice]).reshape(newshape).sum(axis=-1)
            )
            mNi[..., 1, 1] = ssiw[fslice].reshape(newshape).sum(axis=-1)
            mNi[..., 1, 0] = mNi[..., 0, 1].conj()

            # Assign weights before adding in prior term
            ssbgw[fslice] = mNi[..., 0, 0].real
            ssbxw[fslice] = mNi[..., 1, 1].real

            # TODO: don't hardcode prior/regularisation
            mNi += np.array([[1e-6, 0], [0, 1e-8]])

            gxhat = np.linalg.solve(mNi, gx_dirty)

            ssbg[fslice] = gxhat[..., 0]
            ssbx[fslice] = gxhat[..., 1]

        return ss_db

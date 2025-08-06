"""Routines to estimate systematics like crosstalk and gains."""

from typing import overload

import numpy as np
from caput import config, tools

from ..core import containers, task


class ExtractCrosstalkGain(task.SingleTask):
    """Estimate the cross talk and gain relative to an input stack.

    Attributes
    ----------
    nra : int
        The number of output RA bins.
    """

    nra = config.Property(proptype=int, default=64)

    def setup(self, sstack: containers.SiderealStream | containers.VisGridStream):
        """Set the reference sidereal stack.

        Parameters
        ----------
        sstack
            Reference dataset
        """
        self.sstack = sstack

        if len(sstack.ra) % self.nra != 0:
            raise RuntimeError(
                f"The number of RA bins ({self.nra}) must evenly divide the stack "
                f"length ({len(sstack.ra)})."
            )
        self.sstack.redistribute("freq")

    @overload
    def process(
        self, ss: containers.VisGridStream
    ) -> containers.VisCrosstalkGainGrid: ...
    @overload
    def process(self, ss: containers.SiderealStream) -> containers.VisCrosstalkGain: ...
    def process(self, ss):
        """For each input stream assess the cross talk and gain.

        Parameters
        ----------
        ss
            Calculate gains and crosstalk for this dataset

        Returns
        -------
        ss_db
            Container with gain and crosstalk estimates
        """
        # Verify that these datasets are compatible
        if not isinstance(ss, self.sstack.__class__):
            raise RuntimeError(
                f"Both datasets must be the same type. Got {type(ss)} and {type(self.sstack)}."
            )

        if np.any(self.sstack.ra != ss.ra):
            raise RuntimeError(
                "The RA bins in the input must match those in the stack."
            )

        if ss.vis[:].global_shape != self.sstack.vis[:].global_shape:
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

            # Construct the covariance. Because we just have a batch of
            # 2x2 matrices to invert, instead of constructing the inverse
            # covariance and using `solve`, we can make the covariance
            # directly and multiply
            mN = np.zeros((*slshape, self.nra, 2, 2), dtype=np.complex64)
            mN[..., 1, 1] = (
                (np.abs(sssv[fslice]) ** 2 * ssiw[fslice])
                .reshape(newshape)
                .sum(axis=-1)
            )
            mN[..., 0, 0] = ssiw[fslice].reshape(newshape).sum(axis=-1)
            mN[..., 0, 1] = -1.0 * (sssv[fslice].conj() * ssiw[fslice]).reshape(
                newshape
            ).sum(axis=-1)
            mN[..., 1, 0] = -1.0 * mN[..., 0, 1].conj()

            # Assign weights before including determinant and
            # prior terms. Note that these are swapped because
            # the weights should be _inverse_ variance
            ssbgw[fslice] = mN[..., 1, 1].real
            ssbxw[fslice] = mN[..., 0, 0].real

            # Divide by the determinant to get the covariance
            detmN = mN[..., 1, 1] * mN[..., 0, 0] - mN[..., 0, 1] * mN[..., 1, 0]
            mN *= tools.invert_no_zero(detmN)[..., np.newaxis, np.newaxis]

            # For a N-D matrix where N>2, matmul treats the array as a
            # stack of 2D matrices residing in the last two axes, which
            # is the situation we are in. Need to add the extra axis
            # to make this work properly
            # TODO: it should be really easy to write a cython function
            # to do this, but it might not be necessary
            gxhat = mN @ gx_dirty[..., np.newaxis]

            ssbg[fslice] = gxhat[..., 0, 0]
            ssbx[fslice] = gxhat[..., 1, 0]

        return ss_db

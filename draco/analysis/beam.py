"""
======================================================
Beam model related tasks (:mod:`~draco.analysis.beam`)
======================================================

.. currentmodule:: draco.analysis.beam

Tools that enable generation and deconvolution of beam models.

Tasks
=====

.. autosummary::
    :toctree: generated/

    CreateBeamStream
"""

import numpy as np
import scipy.constants

from caput import config, interferometry

from ..core import task
from ..core import io
from ..core import containers
from ..util import tools


class CreateBeamStream(task.SingleTask):
    """Convert a GridBeam to a HybridVisStream that can be used for ringmap maker deconvolution."""

    telescope = None

    def setup(self, telescope: io.TelescopeConvertible):
        """Set the telescope object.

        Parameters
        ----------
        telescope
            The telescope object to use.
        """
        self.telescope = io.get_telescope(telescope)

        self.log.info(
            "Using telescope at latitude %0.4f deg with rotation angle %0.4f deg."
            % (self.telescope.latitude, self.telescope.rotation_angle)
        )

    def process(self, data, beam):
        """Convert the beam model into a format that can be applied to the data.

        Parameters
        ----------
        data : containers.HybridVisStream
            Data to be de-convolved.
        beam : containers.CommonModeGridBeam
            Model for the beam.

        Returns
        -------
        out : containers.HybridVisStream
            Effective beam transfer function.
        """
        beam.redistribute("freq")

        # Determine local frequencies
        nfreq = beam.beam.local_shape[0]
        fstart = beam.beam.local_offset[0]
        fstop = fstart + nfreq

        freq = beam.freq[fstart:fstop]

        # Make sure the beam is in celestial coordinates
        if beam.coords != "celestial":
            raise RuntimeError(
                "Beam must be converted to celestial coordinates prior to generating a HybridVisStream."
            )

        # Check that el matches
        dec = beam.theta

        el_beam = np.sin(np.radians(dec - self.telescope.latitude))
        el_data = data.index_map["el"]

        if not np.allclose(el_beam, el_data):
            raise RuntimeError("The el axis for the beam and data do not match.")

        # Map the RAs
        ha = beam.phi
        ra_beam = (ha + 360.0) % 360.0
        ra_data = data.index_map["ra"]

        map_ra = np.searchsorted(ra_data, ra_beam)

        if not np.allclose(ra_data[map_ra], ra_beam):
            # This checks fails for rev_00 due to the shifted RA axis.
            # Once we move on to processing later revisisions we can
            # raise an error here.
            self.log.info(
                "RA axis of beam and data differ at most by %0.6f deg"
                % np.max(ra_data[map_ra] - ra_beam)
            )

        # Determine other axes
        x = data.index_map["ew"][:]

        arr_ha = np.radians(ha[np.newaxis, np.newaxis, np.newaxis, :])
        arr_dec = np.radians(dec[np.newaxis, np.newaxis, :, np.newaxis])

        # Determine baseline distances
        lmbda = scipy.constants.c * 1e-6 / freq
        u = x[np.newaxis, :] / lmbda[:, np.newaxis]
        u = u[:, :, np.newaxis, np.newaxis]

        # Rotate the baseline distances by the telescope's rotation angle.
        # This assumes that the baseline distances used to beamform in the
        # NS direction were NOT rotated, and hence the phase due to that rotation
        # should be corrected by the beam.  Note that we can only partially
        # correct for the rotation in this way, since we have already collapsed
        # over NS baselines. However, this partial correction should be pretty good
        # for small rotation angles and for sources near meridian.
        rot = np.radians(self.telescope.rotation_angle)
        v = np.sin(rot) * u
        u = np.cos(rot) * u

        # Calculate the phase
        phi = interferometry.fringestop_phase(
            arr_ha, np.radians(self.telescope.latitude), arr_dec, u, v
        ).conj()

        # Reshape the beam datasets to match the output container.
        # The output weight dataset does not have an el axis, use the
        # average non-zero value of the weight along the el direction.
        bweight = beam.weight[:]
        bweight = np.sum(bweight, axis=-2) * tools.invert_no_zero(
            np.sum(bweight > 0, axis=-2, dtype=np.float32)
        )

        # Transpose the first two dimensions from (freq, pol) to (pol, freq)
        bweight = bweight.swapaxes(0, 1)
        bvis = beam.beam[:].swapaxes(0, 1)

        # Create output container
        out = containers.HybridVisStream(
            axes_from=data,
            attrs_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )
        out.redistribute("freq")

        for dset in out.datasets.values():
            dset[:] = 0.0

        out.weight[:][..., map_ra] = bweight
        out.vis[:][..., map_ra] = bvis * phi[np.newaxis, ...]

        return out

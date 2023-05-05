"""Beam model related tasks (:mod:`~draco.analysis.beam`).

.. currentmodule:: draco.analysis.beam

Tools that enable generation and deconvolution of beam models.

Tasks
=====

.. autosummary::
    :toctree: generated/

    CreateBeamStream
    CreateBeamStreamFromTelescope
"""

import numpy as np
import scipy.constants
from caput import interferometry

from ..core import containers, io, task
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
            f"Using telescope at latitude {self.telescope.latitude:.4f} "
            f"deg with rotation angle {self.telescope.rotation_angle:.4f} deg."
        )

    def process(self, data, beam):
        """Convert the beam model into a format that can be deconvolved from data.

        Parameters
        ----------
        data : containers.HybridVisStream
            Data to be de-convolved.
        beam : containers.GridBeam
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
                "Beam must be converted to celestial coordinates prior to generating "
                "a HybridVisStream."
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
        nra = int(round(360.0 / np.abs(ha[1] - ha[0])))
        delta_ra = 360.0 / nra

        map_ra = np.rint(ra_beam / delta_ra).astype(int)

        # Test that the positions of the original beam samples are close enough to exact
        # grid locations in the output grid. This 1e-4 number tolerance is just a guess
        # as to what is reasonable.
        if not np.allclose(ra_beam / delta_ra, map_ra, atol=1e-4):
            raise ValueError(
                "Input beam cannot be placed on an grid between 0 and 360 degrees."
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
            ra=nra,
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


class CreateBeamStreamFromTelescope(CreateBeamStream):
    """Create a HybridVisStream from a telescope instance."""

    def process(self, data):
        """Convert a telescope's beam into a format that can be deconvolved from data.

        Parameters
        ----------
        data : containers.HybridVisStream
            Data to be de-convolved.

        Returns
        -------
        out : containers.HybridVisStream
            Effective beam transfer function.
        """
        beam = self._evaluate_beam(data)

        return super().process(data, beam)

    def _evaluate_beam(self, data):
        """Evaluate the beam model at the coordinates in the data container."""
        # Create the beam container
        inputs = np.array(["common-mode"])
        ha = (data.ra + 180.0) % 360.0 - 180.0
        dec = np.degrees(np.arcsin(data.index_map["el"])) + self.telescope.latitude

        out = containers.GridBeam(
            theta=dec,
            phi=ha,
            input=inputs,
            axes_from=data,
            attrs_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )

        out.redistribute("freq")
        out.beam[:] = 0.0
        out.weight[:] = 1.0

        # Dereference datasets
        beam = out.beam[:].view(np.ndarray)
        weight = out.weight[:].view(np.ndarray)

        # Extract polarisations pairs.  For each polarisation,
        # find a corresponding feed in the telescope instance.
        pol_pairs = out.index_map["pol"]
        unique_pol = list({p for pp in pol_pairs for p in pp})
        map_pol_to_feed = {
            pol: list(self.telescope.polarisation).index(pol) for pol in unique_pol
        }

        # Determine local frequencies
        nfreq = out.beam.local_shape[0]
        fstart = out.beam.local_offset[0]
        fstop = fstart + nfreq

        local_freq = data.index_map["freq"][fstart:fstop]

        # Find the index of the frequency in the telescope instance
        local_freq_index = np.array(
            [
                np.argmin(np.abs(nu - self.telescope.frequencies))
                for nu in local_freq["centre"]
            ]
        )

        local_freq_flag = np.abs(
            local_freq["centre"] - self.telescope.frequencies[local_freq_index]
        ) <= (0.5 * local_freq["width"])

        # Construct a vector that contains the coordinates in the format
        # required for the beam method of the telescope class
        angpos = np.meshgrid(
            0.5 * np.pi - np.radians(dec), np.radians(ha), indexing="ij"
        )
        angpos = np.hstack([ap.reshape(ap.size, 1) for ap in angpos])

        shp = (dec.size, ha.size)

        # Loop over local frequencies and polarisations and evaluate the beam
        # by calling the telescopes beam method.
        for ff, freq in enumerate(local_freq_index):
            if not local_freq_flag[ff]:
                weight[ff] = 0.0
                continue

            for pp, pol in enumerate(pol_pairs):
                bii = self.telescope.beam(map_pol_to_feed[pol[0]], freq, angpos)

                if pol[0] != pol[1]:
                    bjj = self.telescope.beam(map_pol_to_feed[pol[1]], freq, angpos)
                else:
                    bjj = bii

                beam[ff, pp, 0] = np.sum(bii * bjj.conjugate(), axis=1).reshape(shp)

        return out

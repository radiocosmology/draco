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
import scipy.interpolate

from caput import interferometry

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

        self.latitude = np.radians(self.telescope.latitude)
        self.rotation_angle = np.radians(self.telescope.rotation_angle)

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
        # Ensure the polarisations and EW baseline distances match for data and beam
        if not np.array_equal(beam.pol, data.index_map["pol"]):
            raise RuntimeError("Polarisation axis differs for data and beam.")

        ew = data.index_map["ew"][:]
        if (beam.input.size == 1) and (beam.input[0] == "common-mode"):
            ew_match = np.zeros(ew.size, dtype=int)
        else:
            ew_match = np.array([np.argmin(np.abs(b - beam.input)) for b in ew])
            if np.any(np.abs(ew - beam.input[ew_match]) > 0.1):
                raise RuntimeError("EW (input) axis differs for data (beam).")

        # Redistribute over frequency
        beam.redistribute("freq")

        # Determine local frequencies
        nfreq = beam.beam.local_shape[0]
        fstart = beam.beam.local_offset[0]
        fstop = fstart + nfreq

        freq = beam.freq[fstart:fstop]

        # Extract the coordinates
        el_data = data.index_map["el"]

        if beam.coords == "celestial":
            dec = np.radians(beam.theta)
            el_beam = np.sin(dec - self.latitude)

            ha = beam.phi
            ra_beam = (ha + 360.0) % 360.0
            nra = int(round(360.0 / np.abs(ha[1] - ha[0])))
            delta_ra = 360.0 / nra
            ra = np.arange(nra) * delta_ra

            map_ra = np.rint(ra_beam / delta_ra).astype(int)

            # Test that the positions of the original beam samples are close enough to exact
            # grid locations in the output grid. This 1e-4 number tolerance is just a guess
            # as to what is reasonable.
            if not np.allclose(ra_beam / delta_ra, map_ra, atol=1e-4):
                raise ValueError(
                    "Input beam cannot be placed on an grid between 0 and 360 degrees."
                )

            ha = np.radians(ha)

        elif beam.coords == "telescope":
            el_beam = beam.theta
            dec = np.arcsin(el_beam) + self.latitude

            ra = data.ra
            ha = np.radians(((ra + 180.0) % 360.0) - 180.0)

        else:
            raise RuntimeError(f"Do not recognize {beam.coords} coordinate system.")

        # Make sure that el matches
        if not np.allclose(el_beam, el_data):
            raise RuntimeError("The el axis for the beam and data do not match.")

        # Determine baseline distances
        lmbda = scipy.constants.c * 1e-6 / freq
        u = ew[np.newaxis, :] / lmbda[:, np.newaxis]

        # Rotate the baseline distances by the telescope's rotation angle.
        # This assumes that the baseline distances used to beamform in the
        # NS direction were NOT rotated, and hence the phase due to that rotation
        # should be corrected by the beam.  Note that we can only partially
        # correct for the rotation in this way, since we have already collapsed
        # over NS baselines. However, this partial correction should be pretty good
        # for small rotation angles and for sources near meridian.
        v = np.sin(self.rotation_angle) * u
        u = np.cos(self.rotation_angle) * u

        # Reshape the beam datasets to match the output container.
        # The output weight dataset does not have an el axis, use the
        # average non-zero value of the weight along the el direction.
        bweight = beam.weight[:].local_array
        bweight = np.sum(bweight, axis=-2) * tools.invert_no_zero(
            np.sum(bweight > 0, axis=-2, dtype=np.float32)
        )

        # Transpose the first two dimensions from (freq, pol) to (pol, freq)
        bweight = bweight.swapaxes(0, 1)
        bvis = beam.beam[:].local_array.real.swapaxes(0, 1)

        self.log.info("Using the real component.")

        # Create output container
        out = containers.HybridVisStream(
            ra=ra,
            axes_from=data,
            attrs_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )
        out.redistribute("freq")

        for dset in out.datasets.values():
            dset[:] = 0.0

        oweight = out.weight[:].local_array
        ovis = out.vis[:].local_array

        # Use a different procedure for a beam in celestial
        # versus telescope coordinates.
        if beam.coords == "celestial":
            # Calculate the phase
            phi = interferometry.fringestop_phase(
                ha[np.newaxis, np.newaxis, np.newaxis, :],
                self.latitude,
                dec[np.newaxis, np.newaxis, :, np.newaxis],
                u[:, :, np.newaxis, np.newaxis],
                v[:, :, np.newaxis, np.newaxis],
            ).conj()

            # Save the beam times the phase to the output stream
            # at the appropriate right ascensions
            oweight[..., map_ra] = bweight
            ovis[..., map_ra] = bvis * phi[np.newaxis, ...]

        else:  # beam.coords == "telescope"
            # Extract telescope x for the beam
            bx = beam.phi
            span_bx = np.percentile(bx, [0, 100])

            # The x coordinate must be monotonically increasing
            # in order to create a spline.
            if bx[0] > bx[1]:
                bx = bx[::-1]
                bvis = bvis[..., ::-1]

            # Set the weight for the stream equal to the average non-zero weight over x.
            # We will be interpolating over this dimension.
            oweight[:] = np.sum(bweight, axis=-1, keepdims=True) * tools.invert_no_zero(
                np.sum(bweight > 0, axis=-1, keepdims=True, dtype=np.float32)
            )

            # Calculate the x and y coordinate of the stream
            arr_ha, arr_dec = np.meshgrid(ha, dec, copy=True, indexing="xy")
            dx, dy, dz = interferometry.sph_to_ground(arr_ha, self.latitude, arr_dec)

            # Only bother to generate the stream if we are above the horizon and
            # the x coordinate is within the range spanned by the beam
            valid = np.nonzero(
                (dz > 0.0)
                & (dx >= span_bx[0])
                & (dx <= span_bx[1])
                & (np.abs(arr_ha) < (0.5 * np.pi))
            )

            arr_ha = arr_ha[valid]
            arr_dec = arr_dec[valid]

            dx = dx[valid]
            dy = dy[valid]

            # Loop over frequencies and ew baseline distances.
            # This is necessary because we can only create one
            # 2D interpolator at a time.
            for ff in range(nfreq):

                for de, be in enumerate(ew_match):

                    # Calculate the phase
                    phi = interferometry.fringestop_phase(
                        arr_ha, self.latitude, arr_dec, u[ff, de], v[ff, de]
                    ).conj()

                    # Loop over polarisations
                    for pp in range(ovis.shape[0]):
                        # Create a RectBivariateSpline interpolator for the beam at this frequency
                        # as a function of the telescope (x,y) coordinates
                        binterpolator = scipy.interpolate.RectBivariateSpline(
                            el_beam, bx, bvis[pp, ff, be, :, :], kx=3, ky=3, s=0
                        )

                        # Interpolate the beam to the valid x,y coordinates in the data container
                        # and multiply by the phase
                        binterp = binterpolator(dy, dx, grid=False)

                        ovis[pp, ff, de][valid] = binterp * phi

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

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
            ra = (ha + 360.0) % 360.0
            nra = int(round(360.0 / np.abs(ha[1] - ha[0])))
            delta_ra = 360.0 / nra

            map_ra = np.rint(ra / delta_ra).astype(int)

            # Test that the positions of the original beam samples are close enough to exact
            # grid locations in the output grid. This 1e-4 number tolerance is just a guess
            # as to what is reasonable.
            if not np.allclose(ra / delta_ra, map_ra, atol=1e-4):
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
        x = data.index_map["ew"][:]

        lmbda = scipy.constants.c * 1e-6 / freq
        u = x[np.newaxis, :] / lmbda[:, np.newaxis]
        u = u[:, :, np.newaxis]

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
        bvis = beam.beam[:].local_array.swapaxes(0, 1)

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
                u[:, :, np.newaxis, :],
                v[:, :, np.newaxis, :],
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
            # in order to create a CubicSpline.
            if bx[0] > bx[1]:
                bx = bx[::-1]
                bvis = bvis[..., ::-1]

            # Set the weight for the stream equal to the average non-zero weight over x
            # We will be interpolating over this dimension.
            bweight = np.sum(bweight, axis=-1, keepdims=True) * tools.invert_no_zero(
                np.sum(bweight > 0, axis=-1, keepdims=True, dtype=np.float32)
            )

            # Loop over declinations
            for dd, dc in enumerate(dec):
                # Create a CubicSpline interpolator for the beam at this declination
                # as a function of the telescope x coordinate
                binterpolator = scipy.interpolate.CubicSpline(
                    bx, bvis[..., dd, :], axis=-1, bc_type="clamped", extrapolate=True
                )

                # Calculate the x coordinate of the stream
                dx = -1 * np.cos(dc) * np.sin(ha)

                # Only bother to generate the stream if the x coordinate
                # is within the range spanned by the beam and the hour
                # angle is less than 90 degrees (to avoid evaluating
                # the antipodal transit).
                valid = np.flatnonzero(
                    (dx >= span_bx[0])
                    & (dx <= span_bx[1])
                    & (np.abs(ha) < (0.5 * np.pi))
                )

                # Calculate the phase
                phi = interferometry.fringestop_phase(
                    ha[np.newaxis, np.newaxis, valid], self.latitude, dc, u, v
                ).conj()

                # Interpolate the beam to the x coordinates at this declination
                # and then multiply by the phase
                ovis[..., dd, valid] = binterpolator(dx[valid]) * phi[np.newaxis, ...]
                oweight[..., valid] = bweight

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

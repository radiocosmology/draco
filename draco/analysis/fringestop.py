"""Tasks for fringestopping visibilities to enable further downsampling."""

import numpy as np
import scipy.constants
from caput.pipeline import tasklib

from ..core import io


class Mix(tasklib.base.ContainerTask):
    r"""Baseclass for down-mixing or up-mixing visibilities as a function of time.

    Also known as fringestopping.  This task multiplies the visibilities by a
    complex sinusoid in local earth rotation angle :math:`{\phi}`:

    .. math::
        V_{ij}^{downmixed} = V_{ij} * exp(j \omega \phi)

    where the angular frequency :math:`{\omega}` is set by instantaneous fringe rate
    at the centre of the field of view:

    .. math::
        \omega = 2 \pi b_{ew} cos(\delta) / \lambda

    where :math:`b_{ew}` is the east-west baseline distance, :math:`{\delta}`
    is the declination of the centre of the field of view, and :math:`{\lambda}`
    is the wavelength.  This will remove the fringe pattern that would be observed
    in the signal from a point source located at the centre of the field of view.
    This will enable further downsampling in time, assuming the field of view is
    sufficiently small.

    Can also be thought of as applying a baseline, declination, and frequency
    dependent shift in m-mode space.
    """

    def setup(self, manager):
        """Set the local observers position and the telescope baseline distances.

        Parameters
        ----------
        manager : caput.astro.observer.Observer or draco.core.io.TelescopeConvertible
            If the data is already beamformed, then this can be an Observer object
            holding the geographic location of the telescope.  Otherwise this should
            be a TelescopeConvertible that also contains the baseline distribution.
        """
        self.telescope = io.get_telescope(manager)

    def process(self, stream):
        """Multiply visibilities by a complex sinusoid in local earth rotation angle.

        Parameters
        ----------
        stream : TimeStream, SiderealStream, or HybridVisStream
            Visibilites or hybrid beamformed visibilities to fringestop.

        Returns
        -------
        stream : TimeStream, SiderealStream, or HybridVisStream
            Fringestopped visibilites.  Note the input container is modified in place.
        """
        sign = -1.0 if self.conjugate else 1.0

        # Distribute over frequencies
        stream.redistribute("freq")
        freq = stream.freq[stream.vis[:].local_bounds]

        # Dereference visibilities
        vis = stream.vis[:].local_array
        weight = stream.weight[:].local_array

        # Determine the east-west baseline distance
        if "ew" in stream.index_map:
            x = stream.index_map["ew"][:, np.newaxis]

        else:
            prod = stream.prodstack

            aa, bb = prod["input_a"], prod["input_b"]

            x = (
                self.telescope.feedpositions[aa, 0]
                - self.telescope.feedpositions[bb, 0]
            )

            # Determine if any baselines contains masked feeds
            # These baselines will be flagged since they do not
            # have valid baseline distances.
            mask = self.telescope.feedmask[(aa, bb)][np.newaxis, :, np.newaxis].astype(
                float
            )

            vis *= mask
            weight *= mask

        # Determine the local earth rotation angle based on input container type.
        if "ra" in stream.index_map:
            dphi = np.radians(stream.ra)
        else:
            dphi = np.radians(self.telescope.lsa(stream.time))

        # Determine declination
        if "el" in stream.index_map:
            # Handles the case where we have hybrid beamformed visibilities
            cos_dec = np.cos(
                np.arcsin(stream.index_map["el"][np.newaxis, :])
                + np.radians(self.telescope.latitude)
            )
        else:
            # Handles the case where we have visibilities from a telescope
            # pointed at some declination
            pointing = getattr(self.telescope, "elevation_pointing_offset", 0.0)
            cos_dec = np.cos(np.radians(self.telescope.latitude + pointing))

        # Loop over local frequencies
        for ff, nu in enumerate(freq):
            lmbda = scipy.constants.c / (nu * 1e6)

            omega = 2.0 * np.pi * x * cos_dec / lmbda

            for rr, dp in enumerate(dphi):
                ind = np.s_[..., ff] + (np.s_[:],) * omega.ndim + (rr,)
                vis[ind] *= np.exp(1.0j * sign * omega * dp)

        # Describe what has been done
        stream.attrs["fringestopped"] = not self.conjugate

        return stream


class DownMix(Mix):
    """Down-mix the visibilities."""

    conjugate = False


class UpMix(Mix):
    """Up-mix the visibilities."""

    conjugate = True

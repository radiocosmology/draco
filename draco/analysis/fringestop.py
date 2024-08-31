"""Tasks for fringestopping visibilities to enable further downsampling."""

import numpy as np

from caput import config
import scipy.constants

from ..core import task
from ..core import io
from ..util import tools
from ..core import containers


class Mix(task.SingleTask):
    """Baseclass for down-mixing or up-mixing visibilities as a function of time.

    Also called fringestopping.  This task multiplies the visibilities by a
    complex sinusoid in local earth rotation angle \phi:

        V_{ij}^{downmixed} = V_{ij} * exp(j \omega \phi)

    where the angular frequency \omega is set by instantaneous fringe rate at the
    centre of the field of view:

        \omega = 2 \pi b_{ew} cos(\delta) / \lambda

    where b_{ew} is the east-west baseline distance, \delta is the declination
    at the centre of the field of view, and \lambda is the wavelength.  This will
    remove the fringe pattern that would be observed in the signal from a
    point source located at \delta.  This will enable further downsampling in time,
    assuming the field of view is sufficiently small.

    Can also be thought of as applying a baseline, declination, and
    frequency dependent shift in m-mode space.
    """

    def setup(self, manager):
        """Set the local observers position.

        Parameters
        ----------
        observer : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        # Need an Observer object holding the geographic location of the telescope.
        self.observer = io.get_telescope(manager)

    def process(self, stream):
        """Multiply visibilities by a complex sinusoid in local earth rotation angle.

        Parameters
        ----------
        stream : TimeStream, SiderealStream, or HybridVisStream

        Returns
        -------
        stream : TimeStream, SiderealStream, or HybridVisStream
            Note this modifies the input container in place.
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

            x = self.observer.feedpositions[aa, 0] - self.observer.feedpositions[bb, 0]

            # Determine if any baselines contains masked feeds
            # These baselines will be flagged since they do not
            # have valid baseline distances.
            mask = self.observer.feedmask[(aa, bb)][np.newaxis, :, np.newaxis].astype(
                float
            )

            vis *= mask
            weight *= mask

        # Determine the local earth rotation angle based on input container type.
        if "ra" in stream.index_map:
            dphi = np.radians(stream.ra)
        else:
            dphi = np.radians(self.observer.lsa(stream.time))

        # Determine declination
        if "el" in stream.index_map:
            # Handles the case where we have hybrid beamformed visibilities
            cos_dec = np.cos(
                np.arcsin(stream.index_map["el"][np.newaxis, :])
                + np.radians(self.observer.latitude)
            )
        else:
            # Handles the case where we have visibilities from a telescope
            # pointed at some declination
            pointing = getattr(self.observer, "elevation_pointing_offset", 0.0)
            cos_dec = np.cos(np.radians(self.observer.latitude + pointing))

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

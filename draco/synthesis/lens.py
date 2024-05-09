"""Tasks for generating synthetic CMB lensing maps."""
import numpy as np

from caput import config
from cora.util import units

from ..core.task import SingleTask
from ..core.containers import Map


class MakeLensingMap(SingleTask):
    """Generate a CMB lensing map by integrating the given LSS map against redshift.

    Will use an analytic expression for the lensing kernel.
    """

    def _kernel(self, lss):
        """Eq 3 from http://arxiv.org/abs/2304.05203"""

        # get a cosmology from the LSS sim
        z = lss.redshift[:]
        cosmo = lss.cosmology
        cosmo.units = "si"
        chi = cosmo.comoving_distance

        # get into si units
        H0 = cosmo.H0 * 1000.0 / units.mega_parsec

        # redshift to CMB
        z0 = 1100

        return (
            1.5 * cosmo.omega_m * H0**2 * (1 + z) / cosmo.H(z) * chi(z)
            / units.c * (chi(z0) - chi(z)) / chi(z0)
        )

    def process(self, lss):
        """Integrate the LSS simulation to obtain the lensing map.

        Parameters
        ----------
        lss : Map
            Map of matter density fluctuations.

        Returns
        -------
        lens : Map
            Map of CMB lensing.
        """
        lss.redistribute("pixel")

        # create a new container
        lens = Map(axes_from=lss, attrs_from=lss, freq=1, polarisation=False, distributed=True)
        lens.redistribute("pixel")

        kern = self._kernel(lss)

        # calculate redshift bin widths
        z = lss.redshift[:]
        z_bnd = 0.5 * (z[:-1] + z[1:])
        z_bnd = np.concatenate(
            ([z[0] - z_bnd[0] + z[0]], z_bnd, [z[-1] + z[-1] - z_bnd[-1]])
        )
        z_bins = np.diff(z_bnd)

        kern *= z_bins

        lens.map[:] = np.sum(kern[:, np.newaxis, np.newaxis] * lss.map[:], axis=0)[np.newaxis, ...]

        return lens

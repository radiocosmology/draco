"""Tasks for generating synthetic Lyman-alpha forest data."""
import numpy as np

from caput import config

from ..core.task import SingleTask
from ..core.containers import Map


def _fgpa(delta, z):
    """Fluctuating Gunn-Peterson Approximation.

    See arXiv:1201.0594, arXiv:1509.07875, and https://iopscience.iop.org/article/10.1086/305289/pdf
    """
    A = 0.3 * ((1 + z) / 3.4) ** 4.5  # arXiv:1509.07875 sec. 3
    alpha = 1.6
    # Fbar = 0.64  # empirical mean transmission from Croft et al.

    F = np.exp(-A[:, np.newaxis] * (1 + delta) ** alpha)
    F /= F.mean(axis=1)[:, np.newaxis]

    return F


class GenerateFluxTransmission(SingleTask):
    """Generate a Lyman-alpha flux transmission field from an LSS simulation.

    Requires a model that maps matter density fluctuations to Lyman-alpha optical
    depth. This task produces a map of Lyman-alpha flux transmission relative fluctuations
    (i.e. the delta quantity provided by the eBOSS catalogues).

    TODO: should this use the HI density or matter density as an input?

    Attributes
    ----------
    model : string
        Which model to use. Only `"FGPA"` is available right now.
    """

    model = config.enum(["FGPA"], default="FGPA")

    _model_dict = {
        "FGPA": _fgpa,
    }

    def setup(self):
        """Get the optical depth model."""
        self._tx_model = self._model_dict[self.model]

    def process(self, lss):
        """Transform the given LSS simulation to flux transmission.

        Parameters
        ----------
        lss : Map
            Map of matter density fluctuations.

        Returns
        -------
        tx : Map
            Map of Lyman-alpha flux transmission relative fluctuations.
        """
        lss.redistribute("chi")

        # create a new container
        tx = Map(axes_from=lss, attrs_from=lss, polarisation=False)

        offset = lss.delta.local_offset[0]
        loc = slice(offset, offset + lss.delta.local_shape[0])

        tx.map[:] = self._get_tx(lss.delta[:], lss.redshift[loc])[:, np.newaxis, :] - 1

        return tx

    def _get_tx(self, delta, z):
        return self._tx_model(delta, z)

import numpy as np

from caput import config
from cora.signal.lssutil import lognormal_transform

from ..core.task import SingleTask
from ..core.containers import Map


def _fgpa(delta, z):
    """Fluctuating Gunn-Peterson Approximation.
    See arXiv:1201.0594, arXiv:1509.07875, and https://iopscience.iop.org/article/10.1086/305289/pdf
    """

    A = 0.3 * ((1 + z) / 3.4) ** 4.5  # arXiv:1509.07875 sec. 3
    alpha = 1.6
    # Fbar = 0.64  # empirical mean transmission from Croft et al.

    # take lognormal transform to avoid negative optical depths
    delta = lognormal_transform(delta, axis=1)

    F = np.exp(-A[:, np.newaxis] * (1 + delta) ** alpha)
    F /= F.mean(axis=1)[:, np.newaxis]

    return F


class GenerateFluxTransmission(SingleTask):
    # TODO: should this use the HI density or matter density as an input?

    model = config.enum(["FGPA"], default="FGPA")

    _model_dict = {
        "FGPA": _fgpa,
    }

    def setup(self):
        self._tx_model = self._model_dict[self.model]

    def process(self, lss):

        lss.redistribute("chi")

        # create a new container
        tx = Map(axes_from=lss, attrs_from=lss, polarisation=False)

        offset = lss.delta.local_offset[0]
        loc = slice(offset, offset + lss.delta.local_shape[0])

        tx.map[:] = self._get_tx(lss.delta[:], lss.redshift[loc])[:, np.newaxis, :] - 1

        return tx

    def _get_tx(self, delta, z):
        return self._tx_model(delta, z)

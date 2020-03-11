# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
from scipy import constants

from caput import config

from ..core import task
from ..util.tools import invert_no_zero
from ..core.containers import ApertureBeam


class ApertureTransform(task.SingleTask):

    max_ha = config.Property(proptype=float, default=0.0)
    apod = config.enum(["hanning", "uniform"], default="uniform")

    def process(self, transit):

        transit.redistribute("freq")
        freq = transit.freq[:]

        # get direction axes and truncate if specified
        ha = transit.pix["phi"][:]
        ha_ind = (
            np.where(np.abs(ha) <= self.max_ha)[0]
            if self.max_ha > 0.0
            else np.arange(ha.shape[0])
        )
        ha = ha[ha_ind]
        za = transit.pix["theta"][:][ha_ind]

        # direction cosines to source during transit
        # conjugate to aperture coordinates
        k = (
            np.cos(np.radians(za))
            * np.sin(np.radians(ha))
            * 2
            * np.pi
            * freq[:, np.newaxis]
            * 1e6
            / constants.c
        )

        # aperture coordinates
        # pick resolution for highest frequency
        max_fi = np.argmax(freq)
        k_span = k[max_fi, -1] - k[max_fi, 0]
        x = (np.arange(k.shape[1], dtype=np.float32) - k.shape[1] // 2) / k_span

        # apodisation
        apod_map = {
            "uniform": np.ones,
            "hanning": np.hanning,
        }
        apod = apod_map[self.apod](k.shape[1])
        apod /= apod.sum()

        # initialise output container
        apert = ApertureBeam(
            axes_from=transit, attrs_from=transit, x=x, comm=transit.comm
        )

        # dereference datasets
        beam = transit.beam[:]
        weight = transit.weight[:]
        abeam = apert.beam[:]
        aweight = apert.weight[:]

        # transform one frequency at a time
        for lfi, fi in beam.enumerate(axis=0):
            # make the DFT matrix
            dft_weights = np.exp(1j * k[fi][np.newaxis, :] * x[:, np.newaxis])

            # normalize by response at transit
            inorm = beam[lfi, :, :, transit.beam.shape[-1] // 2]
            inorm = invert_no_zero(inorm)[..., np.newaxis]

            # apply DFT
            abeam[lfi] = np.matmul(
                dft_weights,
                (apod * beam[lfi, :, :][..., ha_ind] * inorm)[..., np.newaxis],
            )[..., 0]
            w = np.sum(
                invert_no_zero(weight[lfi, :, :][..., ha_ind])
                * (apod * np.abs(inorm)) ** 2,
                axis=-1,
            )
            aweight[lfi] = invert_no_zero(w)[..., np.newaxis]

        return apert

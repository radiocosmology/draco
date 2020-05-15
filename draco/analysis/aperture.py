""" Tasks for beam measurement processing.

    Tasks
    =====

    .. autosummary::
        :toctree:

        ApertureTransform

"""

import numpy as np
from scipy import constants

from caput import config

from ..core import task, containers
from ..util import tools


class ApertureTransform(task.SingleTask):

    apod = config.enum(["hanning", "uniform"], default="uniform")
    max_ha = config.Property(proptype=float, default=180.0)
    res_mult = config.Property(proptype=int, default=1)
    transit_norm = config.Property(proptype=bool, default=True)

    def process(self, transit):

        ha = transit.pix["phi"][:]
        dec = transit.pix["theta"][:]
        freq = transit.freq[:]

        ha_ind = np.where(np.abs(ha) <= self.max_ha)[0]
        ha_ind = slice(ha_ind[0], ha_ind[-1] + 1)

        transit.redistribute("freq")

        # direction cosines to source during transit
        k = (
            np.cos(np.radians(dec[ha_ind]))
            * np.sin(np.radians(ha[ha_ind]))
            * 2
            * np.pi
            * freq[:, np.newaxis]
            * 1e6
            / constants.c
        )

        # aperture coordinates
        # pick resolution for highest frequency
        f_max = np.argmax(freq)
        k_span = k[f_max].max() - k[f_max].min()
        nx = k.shape[1] * self.res_mult
        x = (np.arange(nx, dtype=np.float32) - nx // 2) / k_span

        # apodisation
        apod_map = {
            "uniform": np.ones,
            "hanning": np.hanning,
        }
        apod = apod_map[self.apod](k.shape[1])
        apod /= apod.sum()

        # output container
        apert = containers.ApertureBeam(
            x=x, axes_from=transit, attrs_from=transit, comm=transit.comm
        )
        apert.redistribute("freq")

        # dereference datasets
        b = transit.beam[:]
        bw = transit.weight[:]
        a = apert.beam[:]
        aw = apert.weight[:]

        for lfi, fi in b.enumerate(0):

            # DFT matrix
            dft_weights = np.exp(1j * np.outer(x, k[fi]))
            dft_weights *= apod

            # normalize by response at transit
            if self.transit_norm:
                norm = b[lfi, :, :, np.argmin(np.abs(ha))]
                norm = np.where(norm == 0.0, 1.0, norm)[..., np.newaxis]
            else:
                norm = 1.0

            # perform transform
            a[lfi] = np.matmul(
                dft_weights, (b[lfi][..., ha_ind] / norm)[..., np.newaxis]
            )[..., 0]

            # sum over weights
            aw[lfi] = tools.invert_no_zero(
                (np.abs(dft_weights[0]) / np.abs(norm) * tools.invert_no_zero(bw[lfi][..., ha_ind])).sum(axis=-1)
            )[..., np.newaxis]

        return apert

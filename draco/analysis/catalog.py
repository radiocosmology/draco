import numpy as np

from ..core import containers, io, task
from ch_util import ephemeris


class CatalogPixelization(task.SingleTask):

    def setup(self, telescope):

        self.telescope = telescope

    def process(self, rm_empty, mock_cat):

        rm_empty.redistribute("freq")

        #pols = rm_empty.index_map["pol"]
        #if ("XY" in pols) or ("YX" in pols):
        #    if ("XY" in pols) ^ ("YX" in pols):
        #        raise ValueError(
        #            "If cross-pols exist both XY and YX must be present." f"Got {pols}."
        #        )
        #    dpol = ["reXY", "imXY"]
        #else:
        #    dpol = []

        #if "XX" in pols:
        #    dpol = ["XX", *dpol]

        #if "YY" in pols:
        #    dpol = dpol.append("YY")

        #dpol = np.array(dpol, dtype="U4")

        rm_out = containers.RingMap(
            attrs_from=rm_empty,
            axes_from=rm_empty,
        )

        rm_out.redistribute("freq")

        rm_freq = rm_empty.index_map["freq"][:]

        freq_axis_index = list(rm_empty.map.attrs["axis"]).index("freq")
        nfreq_local = rm_empty.map.local_shape[freq_axis_index]
        freq_local_offset = rm_empty.map.local_offset[freq_axis_index]

        local_freq_slice = slice(
            freq_local_offset, freq_local_offset + nfreq_local
        )

        rm_ra = rm_empty.index_map["ra"][:]
        rm_el = rm_empty.index_map["el"][:]

        pos_arr = np.array(mock_cat["position"])
        dec_arr = pos_arr["dec"]
        ra_arr = pos_arr["ra"]

        freq_arr = 1420 / (1 + mock_cat["redshift"]["z"])

        dra = np.median(np.abs(np.diff(rm_ra)))
        max_ra_ind = len(rm_ra) - 1
        dza = np.median(np.abs(np.diff(rm_el)))
        za_min = rm_el.min()

        for ff, (nu, width) in enumerate(rm_freq[local_freq_slice]):
            lb = nu - width/2
            ub = nu + width/2

            in_band = (freq_arr < ub) & (freq_arr > lb)

            ra_ib = ra_arr[in_band]
            dec_ib = dec_arr[in_band]

            ra_ind = (np.rint(ra_ib) / dra % max_ra_ind).astype(np.int64)

            za_ind = np.rint(
                (np.sin(np.radians(dec_ib - self.telescope.latitude)) - za_min) / dza
            ).astype(np.int64)

            rm_out.map[:, :, ff, ra_ind, za_ind] += 1

        return rm_out

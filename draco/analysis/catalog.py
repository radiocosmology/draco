import numpy as np

from ..core import containers, io, task
from ch_util import ephemeris


class CatalogPixelization(task.SingleTask):

    def process(self, rm_empty, mock_cat):

        pols = rm_empty.index_map["pol"]
        if ("XY" in pols) or ("YX" in pols):
            if ("XY" in pols) ^ ("YX" in pols):
                raise ValueError(
                    "If cross-pols exist both XY and YX must be present." f"Got {pols}."
                )
            dpol = ["reXY", "imXY"]
        else:
            dpol = []

        if "XX" in pols:
            dpol = ["XX", *dpol]

        if "YY" in pols:
            dpol = dpol.append("YY")

        dpol = np.array(dpol, dtype="U4")

        nbeam = len(rm_empty.index_map["beam"])

        rm_store = containers.RingMap(
            beam=nbeam,
            pol=dpol,
            axes_from=rm_empty,
        )

        rm_store.redistribute("freq")

        freq_axis_ind = list(rm_store.map.attrs["axis"]).index("freq")
        offset_ind = rm_store.map.local_offset[freq_axis_ind]

        rm_freq = rm_empty.index_map["freq"]["centre"][:]
        freq_width = rm_empty.index_map["freq"]["width"][:]
        nfreq = len(rm_freq)

        rm_ra = rm_empty.index_map["ra"][:]
        rm_el = rm_empty.index_map["el"][:]

        nbeam = len(rm_empty.index_map["beam"][:])
        npol = len(rm_empty.index_map["pol"][:])

        pos_arr = np.array(mock_cat["position"])
        dec_arr = pos_arr["dec"]
        ra_arr = pos_arr["ra"]

        freq_arr = 1420 / (1 + mock_cat["redshift"]["z"])

        dra = np.median(np.abs(np.diff(rm_ra)))
        dza = np.median(np.abs(np.diff(rm_el)))
        za_min = rm_el.min()

        full_ind = []

        for i in range(nfreq):
            index_arr = np.where(
                (rm_freq[i] - freq_width[i] / 2 < freq_arr)
                & (freq_arr <= rm_freq[i] + freq_width[i] / 2)
            )

            ra_bin = ra_arr[index_arr]
            dec_bin = dec_arr[index_arr]

            max_ra_ind = len(rm_ra) - 1
            ra_ind = (np.rint(ra_bin) / dra % max_ra_ind).astype(np.int64)

            za_ind = np.rint(
                (np.sin(np.radians(dec_bin - 49.0)) - za_min) / dza
            ).astype(np.int64)

            ind_stack = np.vstack((ra_ind, za_ind))
            full_ind.append(ind_stack)

        rm_store = np.zeros((np.shape(rm_empty["map"][:])))
        for b in range(nbeam):
            for p in range(npol):
                for freq in range(np.shape(rm_store)[2]):
                    freq_offset = freq + offset_ind
                    ra_ind = full_ind[freq_offset][0]
                    el_ind = full_ind[freq_offset][1]
                    for i in range(len(ra_ind)):
                        rm_store[b][p][freq][ra_ind[i]][el_ind[i]] += 1

        rm_empty["map"][:] = rm_store
        return rm_store

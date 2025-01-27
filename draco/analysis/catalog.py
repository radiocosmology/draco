import numpy as np

from ..core import containers, io, task
from ch_util import ephemeris


class CatalogPixelization(task.SingleTask):

    def process(self, rm_empty, mock_cat):

        pols = rm_empty.index_map["pol"]
        if ("XY" in pols) or ("YX" in pols):
            if ("XY" in pols) ^ ("YX" in pols):
                raise ValueError("If cross-pols exist both XY and YX must be present." \
                                 f"Got {pols}.")
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

        offset_ind = rm_store.map.local_offset[2]

        rm_v = rm_empty.index_map["freq"][:]
        rm_ra = rm_empty.index_map["ra"][:]
        rm_el = rm_empty.index_map["el"][:]

        beam = len(rm_empty.index_map["beam"][:])
        pol = len(rm_empty.index_map["pol"][:])
        rm_freq = np.zeros(len(rm_v))
        freq_width = np.zeros(len(rm_v))

        for i in range(len(rm_v)):
            rm_freq[i] = rm_v[i][0]
            freq_width[i] = rm_v[i][1]

        pos_arr = np.array(mock_cat['position'])
        z_arr = np.array(mock_cat['redshift'])

        dec_arr = np.zeros(len(pos_arr))
        for i in range(len(pos_arr)):
            dec_arr[i] = pos_arr[i][1]

        ra_arr = np.zeros(len(pos_arr))
        for i in range(len(pos_arr)):
            ra_arr[i] = pos_arr[i][0]

        freq_arr = np.zeros(len(z_arr))
        for i in range(len(z_arr)):
            freq_arr[i] = 1420/(1+z_arr[i][0])

        dra = np.median(np.abs(np.diff(rm_ra)))
        dza = np.median(np.abs(np.diff(rm_el)))
        za_min = rm_el.min()

        full_ind = []

        for i in range(len(rm_freq)):
            index_arr = np.where((rm_freq[i]-freq_width[i]/2
                                  < freq_arr) & (freq_arr <= rm_freq[i]+freq_width[i]/2))

            ra_bin = ra_arr[index_arr]
            dec_bin = dec_arr[index_arr]

            max_ra_ind = len(rm_ra) - 1
            ra_ind = (np.rint(ra_bin) / dra % max_ra_ind).astype(np.int64)

            za_ind = np.rint((np.sin(np.radians(dec_bin - 49)) - za_min) / dza).astype(np.int64)

            ind_stack = np.vstack((ra_ind, za_ind))
            full_ind.append(ind_stack)

        rm_store = np.zeros((np.shape(rm_empty["map"][:])))
        for b in range(beam):
            for p in range(pol):
                for freq in range(np.shape(rm_store)[2]):
                    freq_offset = freq + offset_ind
                    ra_ind = full_ind[freq_offset][0]
                    el_ind = full_ind[freq_offset][1]
                    for i in range(len(ra_ind)):
                        rm_store[b][p][freq][ra_ind[i]][el_ind[i]] += 1

        rm_empty["map"][:] = rm_store
        return rm_empty

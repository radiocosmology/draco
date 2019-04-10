import h5py
import numpy as np
import healpy as hp
from caput import mpiarray, config, mpiutil, pipeline
from ..core import task, containers, io
from cora.util import units
from ch_util import tools, andata, ephemeris

from caput import pipeline

# Constants
NU21 = units.nu21
C = units.c


class QuasarStack(task.SingleTask):
    """Blah.

    Attributes
    ----------
    qcat_path : str
            Full path to quasar catalog to stack on.



    """

    # Full path to quasar catalog to stack on
    qcat_path = config.Property(proptype=str)

    # Number of frequencies to keep on each side of quasar RA
    # Pick only frequencies around the quasar (50 on each side)
    freqside = config.Property(proptype=int, default=50)

    def setup(self, manager):
        """Load quasar catalog and initialize the stack array.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.

        """
        # Load base quasar catalog from file (Not distributed)
        self._qcat = containers.SpectroscopicCatalog.from_file(
                                                    self.qcat_path)
        self.nqso = len(self._qcat['position'])

        # Size of quasar stack array
        self.nstack = 2 * self.freqside + 1

        # Quasar stack array.
        self.quasar_stack = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype='complex64'), axis=0)
        # Keep track of number of quasars added to each frequency bin
        # in the quasar stack array
        self.quasar_weight = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype='complex64'), axis=0)

        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)

    # TODO: Should data be an argument to process or init?
    # In other words: will we receive a sideral stream a single time
    # or multiple times to add up in the stack?
    def process(self, data):
        """

        Parameters
        ----------
        data : :class:`containers.SiderealStream`

        Returns
        -------
        """
        # Ensure data is distributed in vis-stack axis
        data.redistribute(1)
        nfreq = len(data.index_map['freq'])
        nvis = len(data.index_map['stack'])

        # Find where each Quasar falls in the RA axis
        # Assume equal spacing in RA axis.
        ra_width = np.mean(data.index_map['ra'][1:] -
                           data.index_map['ra'][:-1])
        # Normaly I would set the RAs as the center of the bins.
        # But these start at 0 and end at 360 - ra_width.
        # So they are the left edge of the bin here...
        ra_bins = np.insert(arr=data.index_map['ra'][:] + ra_width,
                            values=0., obj=0)
        # Bin Quasars in RA axis. Need -1 due to the way np.digitize works.
        qso_ra_indices = np.digitize(self._qcat['position']['ra'], ra_bins) - 1
        if not ((qso_ra_indices >= 0) &
                (qso_ra_indices < len(data.index_map['ra']))).all():
            # TODO: raise an error?
            pass

        # Find which quasars are in the frequency range of the data.
        # Frequency of quasars
        qso_freq = NU21/(self._qcat['redshift']['z'] + 1.)  # MHz.
        # Sidereal stream frequency bin edges
        freqbins = (data.index_map['freq']['centre'] +
                    0.5*data.index_map['freq']['width'])
        freqbins = np.append(freqbins, data.index_map['freq']['centre'][-1] -
                             0.5*data.index_map['freq']['width'][-1])
        # Frequency index of quasars (-1 due to np.digitize behaviour)
        qso_findex = np.digitize(qso_freq, freqbins) - 1
        # Only quasars in the frequency range of the data.
        qso_selection = np.where((qso_findex >= 0) & (qso_findex < nfreq))[0]

        # Compute all baseline vectors.
        # Baseline vectors in meters. Mpiarray is created distributed in the
        # 0th axis by default. Argument is global shape.
        bvec_m = mpiarray.MPIArray((nvis, 2), dtype=np.float64)
        copol_indices = []  # Indices (local) of co-pol products
        for lvi, gvi in bvec_m.enumerate(axis=0):

            gpi = data.index_map['stack'][gvi][0]  # Global product index
            # Inputs that go into this product
            ipt0 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][gpi][0]]
            ipt1 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][gpi][1]]

            if self._is_array_copol(ipt0, ipt1):
                copol_indices.append(lvi)

                # Beseline vector in meters
                # I am only computing the baseline vector
                # for co-pol products, but the array has the full shape.
                # Cross-pol entries should be junk.
                unique_index = self.telescope.feedmap[ipt0, ipt1]
                bvec_m[lvi] = self.telescope.baselines[unique_index]
                if self.telescope.feedconj[ipt0, ipt1]:
                    bvec_m[lvi] *= -1.

        copol_indices = np.array(copol_indices, dtype=int)

        # For each quasar in the frequency range of the data
        for qq in qso_selection:

            dec = self._qcat['position']['dec'][qq]
            ra_index = qso_ra_indices[qq]

            # Pick only frequencies around the quasar (50 on each side)
            # Indices to be processed in full frequency axis
            lowindex = np.amax((0, qso_findex[qq] - self.freqside))
            upindex = np.amin((nfreq, qso_findex[qq] + self.freqside + 1))
            f_slice = np.s_[lowindex:upindex]
            # Corresponding indices in quasar stack array
            lowindex = lowindex - qso_findex[qq] + self.freqside
            upindex = upindex - qso_findex[qq] + self.freqside
            qs_slice = np.s_[lowindex:upindex]

            nu = data.index_map['freq']['centre'][f_slice]
            # Baseline vectors in wavelengths. Shape (nstack, nvis_local, 2)
            bvec = bvec_m[np.newaxis, :, :] * nu[:, np.newaxis, np.newaxis] * 1E6 / C
            
            # Complex corrections. Multiply by visibilities to make them real.
            correc = tools.fringestop_phase(
                                ha=0.,
                                lat=np.deg2rad(ephemeris.CHIMELATITUDE),
                                dec=np.deg2rad(dec),
                                u=bvec[:, copol_indices, 0],
                                v=bvec[:, copol_indices, 1])

            # This is done in a slightly weird order: adding visibility subsets
            # for different quasars in each rank first and then co-adding
            # accross visibilities and finally taking the real part:
            # Real( Sum_j Sum_i [ qso_i_vissubset_j ] )
            # Notice that data['vis'] is distributed in axis=1 ('stack'),
            # while quasar_stack is distributed in axis=0 (it's a 1d array).

            # Multiply phase corrections by multiplicity to get
            # fringestopped, natural wheighted visibilities.
            # `correc` now encodes fringestopping corrections in the phase
            # and multiplicity corrections in the magnitude.
            gci = copol_indices + bvec_m.local_offset[0]  # Global copol index
            correc *= self.telescope.redundancy[gci][np.newaxis, :].astype(float)

            # Fringestop, apply weight and sum.
            self.quasar_stack[qs_slice] += np.sum(
              data.vis[f_slice][:, copol_indices, ra_index] * correc, axis=1)
            # Increment wheight for the appropriate quasar stack indices.
            self.quasar_weight[qs_slice] += np.sum(abs(correc), axis=1)

        # TODO: In what follows I gather to all ranks and do the summing
        # in each rank. I end up with the same stack in all ranks, but there
        # might be slight differences due to rounding errors. Should I gather
        # to rank 0, do the operations there, and then scatter to all ranks?

        # Gather quasar stack for all ranks. Each contains the sum
        # over a different subset of visibilities. Add them all to
        # get the beamformed values.
        quasar_stack_full = np.zeros(mpiutil.size*self.nstack,
                                     dtype=self.quasar_stack.dtype)
        # Gather all ranks
        mpiutil.world.Allgather(self.quasar_stack,
                                quasar_stack_full)
        # Construct frequency offset axis
        freq_offset = data.index_map['freq'][int(nfreq/2) - self.freqside:
                                             int(nfreq/2) + self.freqside + 1]
        freq_offset['centre'] = (freq_offset['centre'] -
                                 freq_offset['centre'][self.freqside])
        # Container to hold the stack
        qstack = containers.FrequencyStack(freq=freq_offset)
        # Sum across ranks and take real part to complete the FT
        qstack.stack[:] = np.sum(quasar_stack_full.reshape(
                                    mpiutil.size, self.nstack), axis=0).real
        qstack.weight[:] = self.quasar_weight  # The same for all ranks.

        return qstack

    def _is_array_copol(self, ipt0, ipt1):
        """
        """
        result = (tools.is_array_on(self.telescope._feeds[ipt0]) and
                  tools.is_array_on(self.telescope._feeds[ipt1]))
        if result:
            # Test for co-polarization
            is_copol = (((self.telescope._feeds[ipt0].pol in ['S', 'N']) and
                         (self.telescope._feeds[ipt1].pol in ['S', 'N'])) or
                        ((self.telescope._feeds[ipt0].pol in ['E', 'W']) and
                         (self.telescope._feeds[ipt1].pol in ['E', 'W'])))
            result = result and is_copol

        return result


class QuasarStackFromMaps(task.SingleTask):
    """Blah.

    Attributes
    ----------
    qcat_path : str
            Full path to quasar catalog to stack on.
    map_path : str
            Full path to map to stack on.

    """
    # TODO: Change the map from a config parameter to
    # an object loaded with another task and passed as an 
    # argument to setup.

    # Full path to quasar catalog to stack on
    qcat_path = config.Property(proptype=str)
    map_path = config.Property(proptype=str)

    # Number of frequencies to keep on each side of quasar RA
    # Pick only frequencies around the quasar (50 on each side)
    freqside = config.Property(proptype=int, default=50)

    def setup(self):
        """Load quasar catalog and initialize the stack array.

        """
        # Load base quasar catalog from file (Not distributed)
        self._qcat = containers.SpectroscopicCatalog.from_file(
                                                    self.qcat_path)
        self.nqso = len(self._qcat['position'])

        # Size of quasar stack array
        self.nstack = 2 * self.freqside + 1

        # Quasar stack array.
        self.quasar_stack = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype='complex64'), axis=0)
        # Keep track of number of quasars added to each frequency bin
        # in the quasar stack array
        self.quasar_weight = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype='complex64'), axis=0)

        # Load map
        self._map = h5py.File(self.map_path, 'r')

    # Blah
    def process(self):
        """
        """
        # Blah
        self.nside = hp.npix2nside(self._map['map'].shape[2])
        nfreq = len(self._map['index_map']['freq'])

        # Find which quasars are in the frequency range of the data.
        # Frequency of quasars
        qso_freq = NU21/(self._qcat['redshift']['z'] + 1.)  # MHz.
        # Sidereal stream frequency bin edges
        freqbins = (self._map['index_map']['freq']['centre'] +
                    0.5 * self._map['index_map']['freq']['width'])
        freqbins = np.append(freqbins, self._map['index_map']['freq']['centre'][-1] -
                             0.5 * self._map['index_map']['freq']['width'][-1])
        # Frequency index of quasars (-1 due to np.digitize behaviour)
        qso_findex = np.digitize(qso_freq, freqbins) - 1
        # Only quasars in the frequency range of the data.
        qso_selection = np.where((qso_findex >= 0) & (qso_findex < nfreq))[0]

        # Indices to distribute qso amongst ranks
        local_size, local_offset, _ = mpiutil.split_local(len(qso_selection))

        # For each quasar assigned to this rank
        for ii in range(local_size):

            #if ii%1000==0:
            #    print mpiutil.rank, ii, ii+local_offset
            qq = qso_selection[ii + local_offset]
            ra = self._qcat['position']['ra'][qq]
            dec = self._qcat['position']['dec'][qq]
            pix = self.radec2pix(ra, dec)

            # Pick only frequencies around the quasar (50 on each side)
            # Indices to be processed in full frequency axis
            lowindex = np.amax((0, qso_findex[qq] - self.freqside))
            upindex = np.amin((nfreq, qso_findex[qq] + self.freqside + 1))
            f_slice = np.s_[lowindex:upindex]
            # Corresponding indices in quasar stack array
            lowindex = lowindex - qso_findex[qq] + self.freqside
            upindex = upindex - qso_findex[qq] + self.freqside
            qs_slice = np.s_[lowindex:upindex]

            # Fringestop, apply weight and sum.
            self.quasar_stack[qs_slice] += self._map['map'][f_slice][:, 0, pix]
            # Increment wheight for the appropriate quasar stack indices.
            self.quasar_weight[qs_slice] += 1.

        # Gather quasar stack for all ranks. Each contains the sum
        # over a different subset of quasars.
        quasar_stack_full = np.zeros(mpiutil.size*self.nstack,
                                     dtype=self.quasar_stack.dtype)
        # Gather all ranks
        mpiutil.world.Allgather(self.quasar_stack,
                                quasar_stack_full)
        # Construct frequency offset axis
        freq_offset = self._map['index_map']['freq'][int(nfreq/2) - self.freqside:
                                             int(nfreq/2) + self.freqside + 1]
        freq_offset['centre'] = (freq_offset['centre'] -
                                 freq_offset['centre'][self.freqside])
        # Container to hold the stack
        qstack = containers.FrequencyStack(freq=freq_offset)
        # Sum across ranks and take real part to complete the FT
        qstack.stack[:] = np.sum(quasar_stack_full.reshape(
                                    mpiutil.size, self.nstack), axis=0).real
        qstack.weight[:] = self.quasar_weight  # The same for all ranks.

        # This is needed because there is no argument to proecess()
        # Once I implement the map as an argument this will not be necessary any more.
        self.done = True

        return qstack

    def pix2radec(self, index):
        theta, phi = hp.pixelfunc.pix2ang(self.nside,index)
        return np.degrees(pi*2.-phi), -np.degrees(theta-pi/2.)

    def radec2pix(self, ra, dec):
        return hp.pixelfunc.ang2pix(self.nside, np.radians(-dec+90.), np.radians(360.-ra))

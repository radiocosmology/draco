import h5py
import numpy as np
from caput import mpiarray, config, mpiutil
from ..core import task, containers
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

    # TODO: Should data be an argument to process or init?
    # In other words: will we receive a sideral stream a single time
    # or multiple times to add up in the stack?
    def process(self, data):
        """Smooth the weights with a median filter.

        Parameters
        ----------
        data : :class:`andata.CorrData` or :class:`containers.TimeStream`
            Data containing the weights to be smoothed

        Returns
        -------
        data : Same object as data
            Data object containing the same data as the input, but with the
            weights substituted by the smoothed ones.
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
            conj = data.index_map['stack'][gvi][1]  # Product conjugation
            # Inputs that go into this product
            ipt0 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][gpi][0]]
            ipt1 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][gpi][1]]

            # Get position and polarization of each input
            pos0, pol0 = self._pos_pol(ipt0)
            pos1, pol1 = self._pos_pol(ipt1)

            iscopol = (((pol0 == 0) and (pol1 == 0)) or
                       ((pol0 == 1) and (pol1 == 1)))
            if iscopol:
                copol_indices.append(lvi)

                # Beseline vector in meters
                # I am only computing the baseline vector
                # for co-pol products, but the array has the full shape.
                # Cross-pol entries should be junk.
                bvec_m[lvi] = self._baseline(pos0, pos1, conj=conj)

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

            # Fringestop and sum.
            # TODO: this corresponds to Uniform weighting, not Natural.
            # Need to figure out the multiplicity of each visibility stack.
            self.quasar_stack[qs_slice] += np.sum(
              data['vis'][f_slice][:, copol_indices, ra_index] * correc, axis=1)
            # Increment wheight for the appropriate quasar stack indices.
            self.quasar_weight[qs_slice] += 1.

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

    # TODO: the next two functions are temporary hacks. The information
    # should either be obtained from a TransitTelescope object in the
    # pipeline. Also these functions, if still useful, should be moved to
    # some appropriate place like ch_util.tools?

    # TODO: This is a temporary hack.
    def _pos_pol(self, chan_id, nfeeds_percyl=64, ncylinders=2):
        """ This is a temporary hack. The pipeline will have a
        drift.core.telescope.TransitTelescope object with all of this
        information in it. I have to look up how to use it.

        Parameters
        ----------
        nfeeds_percyl : int
            Number of feeds per cylinder
        ncylinders : int
            Number of cylinders
        """

        cylpol = chan_id // nfeeds_percyl
        cyl = cylpol // ncylinders
        pol = cylpol % ncylinders
        pos = chan_id % nfeeds_percyl

        return (cyl, pos), pol

    # TODO: This is a temporary hack.
    def _baseline(self, pos0, pos1, nu=None, conj=1):
        """ Computes the vector sepparation between two positions
        given in cylinder index and feed position index.
        The vector goes from pos1 to pos0. This gives the right visibility
        phase for CHIME: phi_0_1 = 2pi baseline_0_1 * \hat{n}.

        +X is due East and +Y is due North.

        Parameters
        ----------
        pos0, pos1 : tuple or array-like (cyl, pos)
            cylinder number (W to E) and position in f.l. N to S.
        nu : float
            Frequency in MHz
        conj : int or bool
            If 1 (True) Multiply the final vector by -1
            to account for conjugation of the product.

        Returns
        -------
        Baseline vector in meters (if nu is None)
        or wavelengths (if nu is not None).
       """
        cylinder_sepparation = 22.  # In meters
        feed_sepparation = 0.3048  # In meters

        # -1 is to convert NS feed number (which runs South)
        # into Y coordinates (which point North)
        baseline_vec = np.array([cylinder_sepparation*float(pos0[0]-pos1[0]),
                                 feed_sepparation*float(pos0[1]-pos1[1])*(-1.)])

        if nu is not None:
            nu = float(nu)*1E6
            baseline_vec *= nu / C

        if conj:
            return (-1.)*baseline_vec
        else:
            return baseline_vec

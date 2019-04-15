import h5py
import numpy as np
import healpy as hp
from caput import mpiarray, config, mpiutil, pipeline
from ..core import task, containers, io
from cora.util import units
from ch_util import tools, ephemeris
from drift.telescope import cylbeam

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
    # I think data should be an argument to process() such that we could
    # keep stacking multiple data files in the same stack.
    # The final stack should be returned by the finish() method then,
    # But how does it work with the `save` parameter?
    def process(self, data):
        """

        Parameters
        ----------
        data : :class:`containers.SiderealStream`

        Returns
        -------
        """
        nfreq = len(data.index_map['freq'])

        # RA bins each quasar is closest to. 
        # Phasing will actually be done at quasar position.
        qso_ra_index = np.argmin(
            abs(data.index_map['ra'][:, np.newaxis] -
                self._qcat['position']['ra'][np.newaxis, :]),
            axis=0)

        # Number of RA bins to track the quasars at each side of transit 
        ha_side = self._ha_side(len(data.index_map['ra']))

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

        # Compute baselines and map visibilities polarization
        bvec_m, pol_array = self._bvec_m(data)

        # Ensure data is distributed in freq axis
        data.redistribute(0)
        # Reduced visibility data. Co-polarization only:
        # This increases the amount kept in memory by less than
        # a factor of 2 since it only copies co-pol visibilities.
        copol_vis = [data.vis[:, pol_array == 1, :],
                     data.vis[:, pol_array == 2, :]]  # [x-pol, y-pol]
        # List to store indices (in global full visibility axis)
        # of local visibility bins [x-pol, y-pol]
        full_pol_index = []
        # For each polarization
        for pp in range(2):
            # Wrap each polarization visibility in an MPI array
            # distributed in frequency (axis 0).
            copol_vis[pp] = mpiarray.MPIArray.wrap(copol_vis[pp], axis=0)
            # Re-distribute data in vis-stack axis
            copol_vis[pp] = copol_vis[pp].redistribute(axis=1)
            # Indices (in global full vis axis) of local vis:
            global_range = np.s_[copol_vis[pp].local_offset[1]:
                                 copol_vis[pp].local_offset[1] +
                                 copol_vis[pp].local_shape[1]]
            full_pol_index.append(np.where(pol_array == pp+1)[0][global_range])

        # Reduce bvec_m to local visibility indices [pol-x, pol-y]:
        bvec_m = [bvec_m[full_pol_index[0], :],
                  bvec_m[full_pol_index[1], :]]
        # Compute redundancy [pol-x, pol-y]:
        redundancy = [self.telescope.redundancy[full_pol_index[0]].
                      astype(np.float64),
                      self.telescope.redundancy[full_pol_index[1]].
                      astype(np.float64)]

        qcount = 0  # Quasar counter
        # For each quasar in the frequency range of the data
        for qq in qso_selection:
            qcount += 1
            # Declination of this quasar
            dec = self._qcat['position']['dec'][qq]

            # Compute hour angle array
            ha_array, ra_index_range = self._ha_array(
                                        data,
                                        qso_ra_index[qq],
                                        self._qcat['position']['ra'][qq],
                                        ha_side)

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
            # For each polarization
            for pp in range(2):
                # Baseline vectors in wavelengths. Shape:
                # (nstack (or this quasar's slice), nvis_local, 2)
                bvec = (bvec_m[pp][np.newaxis, :, :] *
                        nu[:, np.newaxis, np.newaxis] * 1E6 / C)
                # Complex corrections. Multiply by vis to make them real.
                correc = tools.fringestop_phase(
                                ha=ha_array,
                                lat=np.deg2rad(ephemeris.CHIMELATITUDE),
                                dec=np.deg2rad(dec),
                                # Broad-cast u, v, to get full result.
                                u=bvec[:, :, 0][:,:,np.newaxis],
                                v=bvec[:, :, 1][:,:,np.newaxis])
                # Beams to be used in the weighting
                # TODO: I could have a calculation here to have the array go 
                # until the beam drop to some fraction of the maximum value.
                beam = self._beamfunc(ha_array[np.newaxis, :], pp, 
                                      nu[:, np.newaxis], np.deg2rad(dec))

                # Multiply phase corrections by multiplicity and beam
                # to get fringestopped, natural wheighted visibilities.
                # `correc` now encodes fringestopping corrections in the phase
                # and multiplicity + beam corrections in the magnitude.
                visweight = (redundancy[pp][np.newaxis, :, np.newaxis] *
                             beam[:, np.newaxis, :])
                correc *= visweight

                # Fringestop, apply weight and sum.
                self.quasar_stack[qs_slice] += np.sum(
                    np.sum(copol_vis[pp][f_slice][:, :, ra_index_range] *
                           correc, axis=2), axis=1)

                # Increment wheight for the appropriate quasar stack indices.
                # The redundancy is the same for all frequencies and quasars,
                # so it amounts to an overall multiplication factor in the weight
                # that can be dropped. I add the beams to the weights because
                # they differ from quasar to quasar and frequency to frequency.
                self.quasar_weight[qs_slice] += np.sum(beam, axis=1)

        # TODO: In what follows I gather to all ranks and do the summing
        # in each rank. I end up with the same stack in all ranks, but there
        # might be slight differences due to rounding errors. Should I gather
        # to rank 0, do the operations there, and then scatter to all ranks?

        # Gather quasar stack for all ranks. Each contains the sum
        # over a different subset of visibilities. Add them all to
        # get the beamformed values.
        quasar_stack_full = np.zeros(mpiutil.size*self.nstack,
                                     dtype=self.quasar_stack.dtype)
        quasar_weight_full = np.zeros(mpiutil.size*self.nstack,
                                      dtype=self.quasar_weight.dtype)
        # Gather all ranks
        mpiutil.world.Allgather(self.quasar_stack,
                                quasar_stack_full)
        mpiutil.world.Allgather(self.quasar_weight,
                                quasar_weight_full)
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
        qstack.weight[:] = np.sum(quasar_weight_full.reshape(
                                    mpiutil.size, self.nstack), axis=0)

        print 'Number of quasars stacked: {0}'.format(qcount)
        return qstack

    def _get_pol(self, ipt0, ipt1):
        """ Returns 1 for co-pol X, 2 for co-pol Y, and 0 otherwise.
        """
        onfeeds = (tools.is_array_on(self.telescope._feeds[ipt0]) and
                   tools.is_array_on(self.telescope._feeds[ipt1]))
        if onfeeds:
            # Test for co-polarization
            xpol = (tools.is_array_x(self.telescope._feeds[ipt0]) and
                    tools.is_array_x(self.telescope._feeds[ipt1]))
            if xpol:
                return 1
            else:
                ypol = (tools.is_array_y(self.telescope._feeds[ipt0]) and
                        tools.is_array_y(self.telescope._feeds[ipt1]))
                if ypol:
                    return 2
        # Default, return 0. Cross-pol or Off.
        # Notice there is no else clause here.
        return 0

    def _bvec_m(self, data):
        """ Baseline vector in meters.
        """
        nvis = len(data.index_map['stack'])
        # pol_array: 1 for copol X, 2 for copol Y, 0 otherwise.
        pol_array = np.zeros(nvis, dtype=int)
        # Baseline vectors in meters.
        bvec_m = np.zeros((nvis, 2), dtype=np.float64)
        # Compute all baseline vectors.
        for vi in range(nvis):
            # Product index
            pi = data.index_map['stack'][vi][0]
            # Inputs that go into this product
            ipt0 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][pi][0]]
            ipt1 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][pi][1]]
            # Populate pol_array
            pol_array[vi] = self._get_pol(ipt0, ipt1)
            if pol_array[vi] > 0:
                # Beseline vector in meters
                # I am only computing the baseline vector
                # for co-pol products, but the array has the full shape.
                # Cross-pol entries should be junk.
                unique_index = self.telescope.feedmap[ipt0, ipt1]
                bvec_m[vi] = self.telescope.baselines[unique_index]
                if self.telescope.feedconj[ipt0, ipt1]:
                    bvec_m[vi] *= -1.

        return bvec_m, pol_array

    def _ha_side(self, nra, timetrack=900.):
        """
        Parameters
        ----------
        nra : int
            Number of RA bins in a sidereal day.
        timetrack : float
            Time in seconds to track at each side of transit.
            Default is 15 minutes.
        
        Returns
        -------
        ha_side : int
            Number of RA bins to track the source at each side of transit.
        """
        approx_time_perbin = 24. * 3600. / float(nra)  # In seconds
        # Track for 15 min at each side of transit
        return int(timetrack / approx_time_perbin)

    def _ha_array(self, data, qso_ra_index, qso_ra, ha_side):
        """

        Parameters
        ----------
        qso_ra_index : int
            Index in data.index_map['ra'] closest to qso_ra
        qso_ra : float
            RA of the quasar
        ha_side : int
            Number of RA/HA bins on each side of transit.

        Returns
        -------
        ha_array : np.ndarray
            Hour angle array in the range -180. to 180
        ra_index_range : np.ndarray of int
            Indices (in data.index_map['ra']) corresponding
            to ha_array.
        """
        # RA range to track this quasar through the beam.
        ra_index_range = np.arange(qso_ra_index - ha_side,
                                   qso_ra_index + ha_side + 1,
                                   dtype=int)
        # Number of RA/HA bins to track the source for
        nra = len(data.index_map['ra'])
        # Wrap RA indices around edges. Negative indices wrap nativelly.
        # Indices beiyond the upper edge need to be wraped manually.
        ra_index_range[ra_index_range >= nra] -= nra
        # Hour angle array (convert to radians)
        ha_array = np.deg2rad(data.index_map['ra'][ra_index_range] - qso_ra)
        # For later convenience it is better if `ha_array` is
        # in the range -pi to pi instead of 0 to 2pi.
        ha_array = (ha_array + np.pi) % (2.*np.pi) - np.pi

        return ha_array, ra_index_range

    def _beamfunc(self, ha, pol, freq, dec, zenith=0.70999994):
        """
        Parameters
        ----------
        ha : array or float
            Hour angle (in radians) to compute beam at.
        freq : array or float
            Frequency in MHz
        dec : array or float
            Declination in radians
        pp : int
            Polarization index. X : 0, Y :1
        zenith : float
            Polar angle of the telescope zenith in radians. 
            Equal to pi/2 - latitude
        """
        def _sig(pp, freq, dec):
            """
            """
            sig_amps = [14.71659607, 9.77384199]
            return sig_amps[pp]/freq/np.cos(dec)
    
        def _amp(pp, dec, zenith):
            """
            """
            def _flat_top_gauss6(x, A, sig, x0):
                """Flat-top gaussian. Power of 6."""
                return A*np.exp(-abs((x-x0)/sig)**6)
            def _flat_top_gauss4(x, A, sig, x0):
                """Flat-top gaussian. Power of 4."""
                return A*np.exp(-abs((x-x0)/sig)**3)
   
            prm_ns_x = np.array([9.97981768e-01, 1.29544939e+00, -2.97276017e-10])
            prm_ns_y = np.array([9.86421042e-01, 8.10213318e-01, 5.02940049e-08])

            if pp==0:
                return _flat_top_gauss6(dec - (0.5 * np.pi - zenith), *prm_ns_x)
            else:
                return _flat_top_gauss4(dec - (0.5 * np.pi - zenith), *prm_ns_y)
        
        ha0 = 0.
        return _amp(pol, dec, zenith)*np.exp(-((ha-ha0)/_sig(pol, freq, dec))**2)



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
        freqbins = np.append(freqbins,
                             self._map['index_map']['freq']['centre'][-1] -
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
        freq_offset = self._map['index_map']['freq'][
            int(nfreq/2) - self.freqside:int(nfreq/2) + self.freqside + 1]
        freq_offset['centre'] = (freq_offset['centre'] -
                                 freq_offset['centre'][self.freqside])
        # Container to hold the stack
        qstack = containers.FrequencyStack(freq=freq_offset)
        # Sum across ranks and take real part to complete the FT
        qstack.stack[:] = np.sum(quasar_stack_full.reshape(
                                    mpiutil.size, self.nstack), axis=0).real
        qstack.weight[:] = self.quasar_weight  # The same for all ranks.

        # This is needed because there is no argument to proecess()
        # Once I implement the map as an argument this will not be
        # necessary any more.
        self.done = True

        return qstack

    def pix2radec(self, index):
        theta, phi = hp.pixelfunc.pix2ang(self.nside, index)
        return np.rad2deg(np.pi * 2. - phi), -np.rad2deg(theta - np.pi / 2.)

    def radec2pix(self, ra, dec):
        return hp.pixelfunc.ang2pix(self.nside,
                                    np.deg2rad(-dec+90.),
                                    #np.deg2rad(360.-ra))
                                    np.deg2rad(ra))

import h5py
import numpy as np
import healpy as hp
from caput import mpiarray, config, mpiutil, pipeline
from ..core import task, containers, io
from cora.util import units
from ch_util import tools, ephemeris
from drift.telescope import cylbeam
from ..util._fast_tools import beamform

# Constants
NU21 = units.nu21
C = units.c


class QuasarStack(task.SingleTask):
    """

    Attributes
    ----------
    freqside : int
            Number of frequency bins to keep on each side of quasar
            when stacking.
    """

    # Number of frequencies to keep on each side of quasar RA
    # Pick only frequencies around the quasar (50 on each side)
    freqside = config.Property(proptype=int, default=50)

    def setup(self, manager, sstream):
        """Load quasar catalog and initialize the stack array.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.
        sstream : :class:`containers.SiderealStream`
            Data to stack on.

        """
        # Extract data info
        self.ra = sstream.index_map['ra']
        self.freq = sstream.index_map['freq']['centre']
        self.nfreq = len(self.freq)
        self.latitude = np.deg2rad(ephemeris.CHIMELATITUDE)
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)

        # Ensure data is distributed in something other than
        # frequency (0) to create bad frequency mask.
        sstream.redistribute(1)
        self.bad_freq_mask = np.invert(np.all(
                        sstream.vis[:] == 0., axis=(1, 2)))
        # Number of RA bins to track the quasars at each side of transit
        self.ha_side = self._ha_side(len(self.ra))

        # Compute baselines and map visibilities polarization
        self.bvec_m, pol_array = self._bvec_m(sstream)

        # Ensure data is distributed in freq axis, to extract
        # copol producs.
        sstream.redistribute(0)
        # Reduced visibility data. Co-polarization only:
        # This increases the amount kept in memory by less than
        # a factor of 2 since it only copies co-pol visibilities.
        self.copol_vis = [sstream.vis[:, pol_array == 1, :],
                          sstream.vis[:, pol_array == 2, :]]  # [x-pol, y-pol]
        # List to store indices (in global full visibility axis)
        # of local visibility bins [x-pol, y-pol]
        full_pol_index = []
        # For each polarization
        for pp in range(2):
            # Swap order of product(1) and RA(2) axes, to reduce striding
            # through memory later on.
            self.copol_vis[pp] = np.moveaxis(self.copol_vis[pp], 1, 2)
            # Wrap each polarization visibility in an MPI array
            # distributed in frequency (axis 0).
            self.copol_vis[pp] = mpiarray.MPIArray.wrap(
                                        self.copol_vis[pp], axis=0)
            # Re-distribute data in vis-stack (prod) axis (which is now #2!).
            self.copol_vis[pp] = self.copol_vis[pp].redistribute(axis=2)
            # Indices (in global full vis axis) of local vis:
            global_range = np.s_[self.copol_vis[pp].local_offset[2]:
                                 self.copol_vis[pp].local_offset[2] +
                                 self.copol_vis[pp].local_shape[2]]
            full_pol_index.append(np.where(pol_array == pp+1)[0][global_range])

        # Reduce bvec_m to local visibility indices [pol-x, pol-y]:
        self.bvec_m = [self.bvec_m[:, full_pol_index[0]],
                       self.bvec_m[:, full_pol_index[1]]]
        # Compute redundancy [pol-x, pol-y]:
        self.redundancy = [self.telescope.redundancy[full_pol_index[0]].
                           astype(np.float64),
                           self.telescope.redundancy[full_pol_index[1]].
                           astype(np.float64)]

        # Construct frequency offset axis (for qstack container)
        self.stack_axis = np.copy(sstream.index_map['freq'][
                int(self.nfreq/2)-self.freqside:
                int(self.nfreq/2)+self.freqside + 1])
        self.stack_axis['centre'] = (
                self.stack_axis['centre'] -
                self.stack_axis['centre'][self.freqside])

        # Size of quasar stack array
        self.nstack = 2 * self.freqside + 1

    def process(self, qcat):
        """

        Parameters
        ----------
        qcat : :class:`containers.SpectroscopicCatalog`
            Quasar catalog to stack on.

        Returns
        -------
        qstack : :class:`containers.FrequencyStack`
            Quasar redshift stack
        """
        # Quasars RA and Dec
        quasar_dec = qcat['position']['dec']
        quasar_ra = qcat['position']['ra']

        # Find which quasars are in the frequency range of the data.
        # Frequency of quasars
        qso_freq = NU21/(qcat['redshift']['z'] + 1.)  # MHz.
        # Get f_mask and qs_indices
        freqdiff = self.freq[np.newaxis, :] - qso_freq[:, np.newaxis]
        # Stack axis bin edges to digitize each quasar at.
        stackbins = (self.stack_axis['centre'] +
                     0.5 * self.stack_axis['width'])
        stackbins = np.append(stackbins, self.stack_axis['centre'][-1] -
                              0.5 * self.stack_axis['width'][-1])
        # Index of each frequency in stack axis, for each quasar
        qs_indices = np.digitize(freqdiff, stackbins) - 1
        # Indices to be processed in full frequency axis for each quasar
        f_mask = ((qs_indices >= 0) & (qs_indices < self.nstack))
        # Only quasars in the frequency range of the data.
        qso_selection = np.where(np.sum(f_mask, axis=1) > 0)[0]

        # Quasar stack array.
        quasar_stack = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype=np.float), axis=0)
        # Keep track of number of quasars added to each frequency bin
        # in the quasar stack array
        quasar_weight = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype=np.float), axis=0)

        qcount = 0  # Quasar counter
        # For each quasar in the frequency range of the data
        for qq in qso_selection:
            qcount += 1
            # Declination of this quasar
            dec = np.deg2rad(quasar_dec[qq])

            # RA bin this quasar is closest to.
            # Phasing will actually be done at quasar position.
            qso_ra_index = np.searchsorted(self.ra, quasar_ra[qq])

            # Compute hour angle array
            ha_array, ra_index_range = self._ha_array(
                    self.ra, qso_ra_index, quasar_ra[qq], self.ha_side)

            # Indices and slice for frequencies included in the stack.
            f_indices = np.arange(self.nfreq, dtype=np.int32)[f_mask[qq]]
            f_slice = np.s_[np.amin(f_indices):np.amax(f_indices)+1]
            nu = self.freq[f_slice]
            # For each polarization
            for pp in range(2):
                # Baseline vectors in wavelengths. Shape:
                # (2, nstack (or this quasar's slice), nvis_local)
                bvec = (self.bvec_m[pp][:, np.newaxis, :] *
                        nu[np.newaxis, :, np.newaxis] * 1E6 / C).copy()
                # Fringestop and sum over products
                formed_beam = beamform(
                        self.copol_vis[pp],
                        self.redundancy[pp], dec,
                        self.latitude, np.cos(ha_array), np.sin(ha_array),
                        bvec[0], bvec[1], f_indices, ra_index_range)

                # Beams to be used in the weighting
                # TODO: I could have a calculation here to have the array go
                # until the beam drop to some fraction of the maximum value.
                beam = self._beamfunc(ha_array[np.newaxis, :], pp,
                                      nu[:, np.newaxis], dec)
                # Sum over RA
                this_stack = np.sum(formed_beam * beam, axis=1)
                # Assign stack to bins
                this_stack = np.bincount(qs_indices[qq][f_slice],
                                         weights=this_stack,
                                         minlength=self.nstack)
                quasar_stack += this_stack

                # Increment wheight for the appropriate quasar stack indices.
                # The redundancy is the same for all frequencies and quasars,
                # so it amounts to an overall multiplication factor
                # that can be dropped. I add the beams to the weights because
                # they differ from quasar to quasar and frequency to frequency.
                this_qs_weight = (np.sum(beam, axis=1) *
                                  self.bad_freq_mask[f_slice].astype(np.float))
                quasar_weight += np.bincount(qs_indices[qq][f_slice],
                                             weights=this_qs_weight,
                                             minlength=self.nstack)

        # TODO: In what follows I gather to all ranks and do the summing
        # in each rank. I end up with the same stack in all ranks, but there
        # might be slight differences due to rounding errors. Should I gather
        # to rank 0, do the operations there, and then scatter to all ranks?

        # Gather quasar stack for all ranks. Each contains the sum
        # over a different subset of visibilities. Add them all to
        # get the beamformed values.
        quasar_stack_full = np.zeros(mpiutil.size*self.nstack,
                                     dtype=quasar_stack.dtype)
        quasar_weight_full = np.zeros(mpiutil.size*self.nstack,
                                      dtype=quasar_weight.dtype)
        # Gather all ranks
        mpiutil.world.Allgather(quasar_stack,
                                quasar_stack_full)
        mpiutil.world.Allgather(quasar_weight,
                                quasar_weight_full)

        # Container to hold the stack
        qstack = containers.FrequencyStack(freq=self.stack_axis)
        # Sum across ranks to complete the FT
        qstack.stack[:] = np.sum(quasar_stack_full.reshape(
                                 mpiutil.size, self.nstack), axis=0)
        qstack.weight[:] = np.sum(quasar_weight_full.reshape(
                                  mpiutil.size, self.nstack), axis=0)

        self.log.info("Number of quasars stacked: {0}".format(qcount))
        return qstack

    def _get_pol(self, ipt0, ipt1):
        """ Returns 1 for co-pol X, 2 for co-pol Y, and 0 otherwise.
        """
        onfeeds = (tools.is_array_on(self.telescope._feeds[ipt0]) and
                   tools.is_array_on(self.telescope._feeds[ipt1]))
        # Do not include autos
        if onfeeds and (ipt0 != ipt1):
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
        bvec_m = np.zeros((2, nvis), dtype=np.float64)
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
                bvec_m[:, vi] = self.telescope.baselines[unique_index]
                # No need to conjugate. Already done in telescope.baselines.
                #if self.telescope.feedconj[ipt0, ipt1]:
                #    bvec_m[:, vi] *= -1.

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

    def _ha_array(self, ra, qso_ra_index, qso_ra, ha_side):
        """

        Parameters
        ----------
        ra : array
            RA axis in the data
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
                                   dtype=np.int32)
        # Number of RA bins in data. For wrapping around.
        nra = len(ra)
        # Wrap RA indices around edges.
        ra_index_range[ra_index_range < 0] += nra
        ra_index_range[ra_index_range >= nra] -= nra
        # Hour angle array (convert to radians)
        ha_array = np.deg2rad(ra[ra_index_range] - qso_ra)
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
            sig_amps = [14.87857614, 9.95746878]
            return sig_amps[pp]/freq/np.cos(dec)

        def _amp(pp, dec, zenith):
            """
            """
            def _flat_top_gauss6(x, A, sig, x0):
                """Flat-top gaussian. Power of 6."""
                return A*np.exp(-abs((x-x0)/sig)**6)

            def _flat_top_gauss3(x, A, sig, x0):
                """Flat-top gaussian. Power of 3."""
                return A*np.exp(-abs((x-x0)/sig)**3)

            prm_ns_x = np.array([9.97981768e-01, 1.29544939e+00, 0.])
            prm_ns_y = np.array([9.86421047e-01, 8.10213326e-01, 0.])

            if pp == 0:
                return _flat_top_gauss6(dec - (0.5 * np.pi - zenith),
                                        *prm_ns_x)
            else:
                return _flat_top_gauss3(dec - (0.5 * np.pi - zenith),
                                        *prm_ns_y)

        ha0 = 0.
        return _amp(pol, dec, zenith)*np.exp(
                                    -((ha-ha0)/_sig(pol, freq, dec))**2)


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

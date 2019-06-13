import numpy as np
from caput import mpiarray, config, mpiutil, pipeline
from ..core import task, containers, io
from cora.util import units
from ch_util import tools, ephemeris
from drift.telescope import cylbeam
from ..util._fast_tools import beamform

# Constants
NU21 = units.nu21
C = units.c


class BeamFormBase(task.SingleTask):
    """ Base class for beam forming tasks.
        Only defines a few useful methods.
        Neither setup() nor process() are defined.
    """
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

        Returns
        -------
        beam : array or float
            The beam at the designated hhour angles, frequencies
            and declinations. This is the beam 'power', that is,
            voltage squared. To get the beam voltage, take the
            square root.
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


    def beamform_source(self, src, formed_beam):
        """
        """
            # Declination of this src
            dec = np.deg2rad(self.sdec[src])

            # RA bin this src is closest to.
            # Phasing will actually be done at src position.
            sra_index = np.searchsorted(self.ra, self.sra[src])

            # Compute hour angle array
            # TODO: I could have a calculation here to have the array go
            # until the beam drop to some fraction of the maximum value.
            ha_array, ra_index_range = self._ha_array(
                    self.ra, sra_index, self.sra[src], self.ha_side)

            # Indices of full frequency axis. Needed for the way beamform() works.
            f_indices = np.arange(self.nfreq, dtype=np.int32)
            # For each polarization
            for pol in range(self.npol):

                # Fringestop and sum over products
                this_formed_beam = beamform(
                        self.copol_vis[pol],
                        self.redundancy[pol],
                        dec, self.latitude,
                        np.cos(ha_array), np.sin(ha_array),
                        self.bvec[pol][0], self.bvec[pol][1],
                        f_indices, ra_index_range)

                # Beams to be used in the weighting
                # TODO: I could have a calculation here to have the array go
                # until the beam drop to some fraction of the maximum value.
                beam = self._beamfunc(ha_array[np.newaxis, :], pol,
                                      self.freq[:, np.newaxis], dec)
                # Sum over RA
                this_formed_beam = np.sum(this_formed_beam * beam, axis=1)

                # Gather all ranks. Each contains the sum over a 
                # different subset of visibilities. Add them all to
                # get the beamformed values.
                formed_beam_full = np.zeros(mpiutil.size*self.nfreq,
                                             dtype=float)
                # Gather all ranks
                mpiutil.world.Allgather(this_formed_beam,
                                        formed_beam_full)
                # Sum across ranks to complete the FT
                formed_beam.fbeam[src, pol, :] = np.sum(formed_beam_full.reshape(
                                         mpiutil.size, self.nfreq), axis=0)


class BeamForm(BeamFormBase):
    """ Version of BeamForm aimed at beamforming a single source catalog
        for a list of diffent datasets.

    """

    def setup(self, manager, source_cat):
        """

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.
        source_cat : :class:`containers.SourceCatalog`
            Catalog of points to beamform at.

        """
        self.latitude = np.deg2rad(ephemeris.CHIMELATITUDE)
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)

        #
        self.source_cat = source_cat
        # Number of poinytings
        self.nsource = len(self.source_cat['position'])
        # Pointings RA and Dec
        self.sdec = self.source_cat['position']['dec']
        self.sra = self.source_cat['position']['ra']
        # Number of polarizations. This should be generalized to be a parameter
        self.npol = 2

    def process(self, sstream):
        """

        Parameters
        ----------
        sstream : :class:`containers.SiderealStream`
            Data to beamform on.

        Returns
        -------
        """
        # Extract data info
        self.ra = sstream.index_map['ra']
        self.freq = sstream.index_map['freq']['centre']
        self.nfreq = len(self.freq)

        # Ensure data is distributed in something other than
        # frequency (0) to create bad frequency mask.
        sstream.redistribute(1)
        self.bad_freq_mask = np.invert(np.all(
                        sstream.vis[:] == 0., axis=(1, 2)))

        # Number of RA bins to track each source at each side of transit
        # TODO: Should make this declination dependent to ensure going 
        # well beyond the primary beam.
        self.ha_side = self._ha_side(len(self.ra))

        # Everything under this line needs to be generalysed for any
        # Particular polarization XX, YY, XY, YX.
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
        for pp in range(self.npol):
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

        # Reduce bvec_m to local visibility indices [pol-x, pol-y]
        # and multiply by freq to get vector inwavelengths.
        # Shape: (2, nfreq, nvis_local) for each pol.
        self.bvec = [(self.bvec_m[:, np.newaxis, full_pol_index[pol]] *
                      self.freq[np.newaxis, :, np.newaxis] * 1E6 / C).copy()
                     for pol in range(self.npol)]

        # Compute redundancy [pol-x, pol-y]:
        self.redundancy = [self.telescope.redundancy[full_pol_index[0]].
                           astype(np.float64),
                           self.telescope.redundancy[full_pol_index[1]].
                           astype(np.float64)]

        # Container to hold the formed beams
        #formed_beam = np.zeros((nsource, npol, self.nfreq), dtype=float)
        #pol = np.array(['I', 'Q', 'U', 'V']), pol = np.array(['I'])
        formed_beam = containers.FormedBeam(freq=self.freq, object_id=self.source_cat.index_map['object_id'], pol=np.array(['X','Y']))

        # For each source
        for src in range(self.nsource):
            self.beamform_source(src, formed_beam)

        return formed_beam



class BeamFormCat(BeamFormBase):
    """ Version of BeamForm aimed at beamforming at a list of different source
        catalogs in a single dataset.

    """

    def setup(self, manager, sstream):
        """

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.
        sstream : :class:`containers.SiderealStream`
            Data to beamform on.

        """
        # Extract data info
        self.ra = sstream.index_map['ra']
        self.freq = sstream.index_map['freq']['centre']
        self.nfreq = len(self.freq)
        self.npol = 2
        self.latitude = np.deg2rad(ephemeris.CHIMELATITUDE)
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)

        # Ensure data is distributed in something other than
        # frequency (0) to create bad frequency mask.
        sstream.redistribute(1)
        self.bad_freq_mask = np.invert(np.all(
                        sstream.vis[:] == 0., axis=(1, 2)))

        # Number of RA bins to track each source at each side of transit
        # TODO: Should make this declination dependent to ensure going 
        # well beyond the primary beam.
        self.ha_side = self._ha_side(len(self.ra))

        # Everything under this line needs to be generalysed for any
        # Particular polarization XX, YY, XY, YX.
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
        for pp in range(self.npol):
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

        # Reduce bvec_m to local visibility indices [pol-x, pol-y]
        # and multiply by freq to get vector inwavelengths.
        # Shape: (2, nfreq, nvis_local) for each pol.
        self.bvec = [(self.bvec_m[:, np.newaxis, full_pol_index[pol]] *
                      self.freq[np.newaxis, :, np.newaxis] * 1E6 / C).copy()
                     for pol in range(self.npol)]

        # Compute redundancy [pol-x, pol-y]:
        self.redundancy = [self.telescope.redundancy[full_pol_index[0]].
                           astype(np.float64),
                           self.telescope.redundancy[full_pol_index[1]].
                           astype(np.float64)]

    def process(self, source_cat):
        """

        Parameters
        ----------
        source_cat : :class:`containers.SourceCatalog`
            Catalog of points to beamform at.

        Returns
        -------
        """
        # Number of poinytings
        self.nsource = len(source_cat['position'])
        # Pointings RA and Dec
        self.sdec = source_cat['position']['dec']
        self.sra = source_cat['position']['ra']
        # Number of polarizations. This should be generalized to be a parameter

        # Container to hold the formed beams
        #formed_beam = np.zeros((nsource, npol, self.nfreq), dtype=float)
        #pol = np.array(['I', 'Q', 'U', 'V']), pol = np.array(['I'])
        formed_beam = containers.FormedBeam(freq=self.freq, object_id=source_cat.index_map['object_id'], pol=np.array(['X','Y']))

        # For each source
        for src in range(self.nsource):
            self.beamform_source(src, formed_beam)

        return formed_beam


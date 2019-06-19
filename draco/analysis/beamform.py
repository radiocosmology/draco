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

    Attributes
    ----------
    collapse_ha : bool
        Wether or not to sum over hour-angle/time to complete
        the beamforming. Default is True, which sums over.
    """

    collapse_ha = config.Property(proptype=bool, default=True)


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

    def _ha_side(self, data, timetrack=900.):
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
        if self.is_sstream:
            approx_time_perbin = 24. * 3600. / float(len(self.ra))  # seconds
        else:
            approx_time_perbin = data.time[1] - data.time[0]

        # Track for 15 min at each side of transit
        return int(timetrack / approx_time_perbin)

    def _ha_array(self, ra, source_ra_index, source_ra, 
                  ha_side, is_sstream=True):
        """

        Parameters
        ----------
        ra : array
            RA axis in the data
        source_ra_index : int
            Index in data.index_map['ra'] closest to source_ra
        source_ra : float
            RA of the quasar
        ha_side : int
            Number of RA/HA bins on each side of transit.
        is_sstream : bool
            True if data is sidereal stream. Flase if time stream


        Returns
        -------
        ha_array : np.ndarray
            Hour angle array in the range -180. to 180
        ra_index_range : np.ndarray of int
            Indices (in data.index_map['ra']) corresponding
            to ha_array.
        """
        # RA range to track this quasar through the beam.
        ra_index_range = np.arange(source_ra_index - ha_side,
                                   source_ra_index + ha_side + 1,
                                   dtype=np.int32)
        # Number of RA bins in data.
        nra = len(ra)
        # Number of HA bins to fringestop to.
        nha = 2 * ha_side  + 1

        if is_sstream:
            # Wrap RA indices around edges.
            ra_index_range[ra_index_range < 0] += nra
            ra_index_range[ra_index_range >= nra] -= nra
            # Hour angle array (convert to radians)
            ha_array = np.deg2rad(ra[ra_index_range] - source_ra)
            # For later convenience it is better if `ha_array` is
            # in the range -pi to pi instead of 0 to 2pi.
            ha_array = (ha_array + np.pi) % (2.*np.pi) - np.pi
            return ha_array, ra_index_range

        else:
            # Mask-out indices out of range
            ha_mask = (ra_index_range >= 0) & (ra_index_range < nra)
            # Return smaller HA range, and mask.
            ra_index_range = ra_index_range[ha_mask]
            # Hour angle array (convert to radians)
            ha_array = np.deg2rad(ra[ra_index_range] - source_ra)
            # For later convenience it is better if `ha_array` is
            # in the range -pi to pi instead of 0 to 2pi.
            ha_array = (ha_array + np.pi) % (2.*np.pi) - np.pi
            return ha_array, ra_index_range, ha_mask

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
        if self.is_sstream:
            sra_index = np.searchsorted(self.ra, self.sra[src])
            # Compute hour angle array
            # TODO: I could have a calculation here to have the array go
            # until the beam drop to some fraction of the maximum value.
            ha_array, ra_index_range = self._ha_array(
                self.ra, sra_index, self.sra[src],
                self.ha_side, self.is_sstream)
        else:
            # Cannot use searchsorted, because RA might not be
            # monotonically increasing. Slower.
            # TODO: It would be nice to check that the choice of 
            # sra_index is the one closest to the centre of the 
            # data (in case there are two transits).
            transit_diff = abs(self.ra - self.sra[src])
            sra_index = np.argmin(transit_diff)
            # For now, skip sources that do not transit in the data
            ra_cadence = self.ra[1]-self.ra[0]
            if transit_diff[sra_index] > 1.5 * ra_cadence:
                return 0
            # Compute hour angle array
            ha_array, ra_index_range, ha_mask = self._ha_array(
                self.ra, sra_index, self.sra[src],
                self.ha_side, self.is_sstream)

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

            if self.collapse_ha:
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
                formed_beam.fbeam[src, pol, :] = np.sum(
                        formed_beam_full.reshape(
                            mpiutil.size, self.nfreq), axis=0)

            else:
                # The length of the HA axis might not be equal to 
                # self.nha for all sources in case self.is_sstream 
                # is False. It could be smaller if some HA are
                # outside the range of the data. I need this number
                # explicitly to gather ranks.
                nha = len(ha_array)
                # Gather all ranks
                formed_beam_full = np.zeros(
                        (mpiutil.size*self.nfreq, nha), dtype=float)
                mpiutil.world.Allgather(this_formed_beam,
                                        formed_beam_full)
                # Sum across ranks to complete the FT
                if self.is_sstream:
                    formed_beam.fbeam[src, pol, :] = np.sum(
                            formed_beam_full.reshape(
                                mpiutil.size, self.nfreq, nha), axis=0)
                    formed_beam.ha[src, :] = ha_array

                else:
                    # Populate only where ha_mask is true.
                    # The arrays have been initialized to zeros.
                    print 'Haa', mpiutil.size, nha, formed_beam_full.shape
                    print 'Huu', formed_beam.fbeam[src, pol].shape, formed_beam.fbeam[src, pol, :, ha_mask].shape
                    formed_beam.fbeam[src, pol][:, ha_mask] = np.sum(
                            formed_beam_full.reshape(
                                mpiutil.size, self.nfreq, nha), axis=0)
                    formed_beam.ha[src, ha_mask] = ha_array


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

    def process(self, data):
        """

        Parameters
        ----------
        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to beamform on.

        Returns
        -------
        """
        # Extract data info
        if 'ra' in data.index_map.keys():
            self.is_sstream = True
            self.ra = data.index_map['ra']
        else:
            self.is_sstream = False
            # Convert data timestamps into LSAs (degrees)
            self.ra = self.telescope.unix_to_lsa(data.time)

        self.freq = data.index_map['freq']['centre']
        self.nfreq = len(self.freq)

        # Ensure data is distributed in something other than
        # frequency (0) to create bad frequency mask.
        data.redistribute(1)
        self.bad_freq_mask = np.invert(np.all(
                        data.vis[:] == 0., axis=(1, 2)))

        # Number of RA bins to track each source at each side of transit
        # TODO: Should make this declination dependent to ensure going 
        # well beyond the primary beam.
        self.ha_side = self._ha_side(data)
        self.nha = 2 * self.ha_side + 1

        # Everything under this line needs to be generalysed for any
        # Particular polarization XX, YY, XY, YX.
        # Compute baselines and map visibilities polarization
        self.bvec_m, pol_array = self._bvec_m(data)

        # Ensure data is distributed in freq axis, to extract
        # copol producs.
        data.redistribute(0)
        # Reduced visibility data. Co-polarization only:
        # This increases the amount kept in memory by less than
        # a factor of 2 since it only copies co-pol visibilities.
        self.copol_vis = [data.vis[:, pol_array == 1, :],
                          data.vis[:, pol_array == 2, :]]  # [x-pol, y-pol]
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

        if self.collapse_ha:
            # Container to hold the formed beams
            #pol = np.array(['I', 'Q', 'U', 'V']), pol = np.array(['I'])
            formed_beam = containers.FormedBeam(
                    freq=self.freq,
                    object_id=self.source_cat.index_map['object_id'],
                    pol=np.array(['X','Y']))
        else:
            # Container to hold the formed beams
            formed_beam = containers.FormedBeamHA(
                    freq=self.freq, ha=np.arange(self.nha, dtype=int),
                    object_id=self.source_cat.index_map['object_id'],
                    pol=np.array(['X','Y']))
            # Initialize container to zeros. 
            formed_beam.fbeam[:] = 0.
            formed_beam.ha[:] = 0.  # TODO: Should this be np.nan instead?

        # For each source, beamform and populate container.
        for src in range(self.nsource):
            self.beamform_source(src, formed_beam)

        return formed_beam


class BeamFormCat(BeamFormBase):
    """ Version of BeamForm aimed at beamforming at a list of different source
        catalogs in a single dataset.

    """

    def setup(self, manager, data):
        """

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.
        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to beamform on.

        """
        # Extract data info
        if 'ra' in data.index_map.keys():
            self.is_sstream = True
            self.ra = data.index_map['ra']
        else:
            self.is_sstream = False
            # Convert data timestamps into LSAs (degrees)
            self.ra = self.telescope.unix_to_lsa(data.time)
        self.freq = data.index_map['freq']['centre']
        self.nfreq = len(self.freq)
        self.npol = 2
        self.latitude = np.deg2rad(ephemeris.CHIMELATITUDE)
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)

        # Ensure data is distributed in something other than
        # frequency (0) to create bad frequency mask.
        data.redistribute(1)
        self.bad_freq_mask = np.invert(np.all(
                        data.vis[:] == 0., axis=(1, 2)))

        # Number of RA bins to track each source at each side of transit
        # TODO: Should make this declination dependent to ensure going 
        # well beyond the primary beam.
        self.ha_side = self._ha_side(data)
        self.nha = 2 * self.ha_side  + 1

        # Everything under this line needs to be generalysed for any
        # Particular polarization XX, YY, XY, YX.
        # Compute baselines and map visibilities polarization
        self.bvec_m, pol_array = self._bvec_m(data)

        # Ensure data is distributed in freq axis, to extract
        # copol producs.
        data.redistribute(0)
        # Reduced visibility data. Co-polarization only:
        # This increases the amount kept in memory by less than
        # a factor of 2 since it only copies co-pol visibilities.
        self.copol_vis = [data.vis[:, pol_array == 1, :],
                          data.vis[:, pol_array == 2, :]]  # [x-pol, y-pol]
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

        if self.collapse_ha:
            # Container to hold the formed beams
            #pol = np.array(['I', 'Q', 'U', 'V']), pol = np.array(['I'])
            formed_beam = containers.FormedBeam(
                    freq=self.freq,
                    object_id=self.source_cat.index_map['object_id'],
                    pol=np.array(['X','Y']))
        else:
            # Container to hold the formed beams
            formed_beam = containers.FormedBeamHA(
                    freq=self.freq, ha=np.arange(self.nha, dtype=int),
                    object_id=self.source_cat.index_map['object_id'],
                    pol=np.array(['X','Y']))
            # Initialize container to zeros. 
            formed_beam.fbeam[:] = 0.
            formed_beam.ha[:] = 0.  # TODO: Should this be np.nan instead?

        # For each source
        for src in range(self.nsource):
            self.beamform_source(src, formed_beam)

        return formed_beam


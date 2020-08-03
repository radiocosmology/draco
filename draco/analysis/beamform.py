# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
from mpi4py import MPI

from caput import config, mpiutil, mpiarray
from cora.util import units

from ..core import task, containers, io
from ..util._fast_tools import beamform
from ..util.tools import baseline_vector, polarization_map, invert_no_zero
from ..util.tools import calculate_redundancy

# Constants
NU21 = units.nu21
C = units.c


class BeamFormBase(task.SingleTask):
    """ Base class for beam forming tasks.

    Defines a few useful methods. Not to be used directly
    but as parent class for BeamForm and BeamFormCat.

    Attributes
    ----------
    collapse_ha : bool
        Wether or not to sum over hour-angle/time to complete
        the beamforming. Default is True, which sums over.
    polarization : string
        One of:
        'I' : Stokes I only.
        'full' : 'XX', 'XY', 'YX' and 'YY' in this order.
        'copol' : 'XX' and 'YY' only.
        'stokes' : 'I', 'Q', 'U' and 'V' in this order. Not implemented.
    weight : string ('natural', 'uniform', or 'inverse_variance')
        How to weight the redundant baselines when adding:
            'natural' - each baseline weighted by its redundancy (default)
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute
    timetrack : float
        How long (in seconds) to track sources at each side of transit.
        Total transit time will be ~ 2 * timetrack.
    freqside : int
        Number of frequencies to process at each side of the source.
        Default (None) processes all frequencies.
    """

    collapse_ha = config.Property(proptype=bool, default=True)
    polarization = config.enum(["I", "full", "copol", "stokes"], default="full")
    weight = config.enum(["natural", "uniform", "inverse_variance"], default="natural")
    timetrack = config.Property(proptype=float, default=900.0)
    freqside = config.Property(proptype=int, default=None)

    def setup(self, manager):
        """ Generic setup method.

        To be complemented by specific
        setup methods in daughter tasks.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.
        """
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)
        # Polarizations.
        if self.polarization == "I":
            self.process_pol = ["XX", "YY"]
            self.return_pol = ["I"]
        elif self.polarization == "full":
            self.process_pol = ["XX", "XY", "YX", "YY"]
            self.return_pol = self.process_pol
        elif self.polarization == "copol":
            self.process_pol = ["XX", "YY"]
            self.return_pol = self.process_pol
        elif self.polarization == "stokes":
            self.process_pol = ["XX", "XY", "YX", "YY"]
            self.return_pol = ["I", "Q", "U", "V"]
            msg = "Stokes parameters are not implemented"
            raise RuntimeError(msg)
        else:
            # This should never happen. config.enum should bark first.
            msg = "Invalid polarization parameter: {0}"
            msg = msg.format(self.polarization)
            raise ValueError(msg)
        # Number of polarizations to process
        self.npol = len(self.process_pol)
        self.latitude = np.deg2rad(self.telescope.latitude)

    def process(self):
        """ Generic process method.

        Performs all the beamforming,
        but not the data parsing. To be complemented by specific
        process methods in daughter tasks.

        Returns
        -------
        formed_beam : `containers.FormedBeam` or `containers.FormedBeamHA`
            Formed beams at each source. Shape depends on parameter
            `collapse_ha`.
        """
        # Contruct containers for formed beams
        if self.collapse_ha:
            # Container to hold the formed beams
            formed_beam = containers.FormedBeam(
                freq=self.freq,
                object_id=self.source_cat.index_map["object_id"],
                pol=np.array(self.return_pol),
                distributed=True,
            )
        else:
            # Container to hold the formed beams
            formed_beam = containers.FormedBeamHA(
                freq=self.freq,
                ha=np.arange(self.nha, dtype=np.int),
                object_id=self.source_cat.index_map["object_id"],
                pol=np.array(self.return_pol),
                distributed=True,
            )
            # Initialize container to zeros.
            formed_beam.ha[:] = 0.0

        # Initialize container to zeros.
        formed_beam.beam[:] = 0.0
        formed_beam.weight[:] = 0.0
        # Copy catalog information
        formed_beam["position"][:] = self.source_cat["position"][:]
        if "redshift" in self.source_cat:
            formed_beam["redshift"][:] = self.source_cat["redshift"][:]
        else:
            # TODO: If there is not redshift information,
            # should I have a different formed_beam container?
            formed_beam["redshift"]["z"][:] = 0.0
            formed_beam["redshift"]["z_error"][:] = 0.0
        # Ensure container is distributed in frequency
        formed_beam.redistribute("freq")

        if self.freqside is None:
            # Indices of local frequency axis. Full axis if freqside is None.
            f_local_indices = np.arange(self.ls, dtype=np.int32)
            f_mask = np.zeros(self.ls, dtype=bool)

        # For each source, beamform and populate container.
        for src in range(self.nsource):

            # Declination of this source
            dec = self.sdec[src]

            if self.freqside is not None:
                # Get the frequency bin this source is closest to.
                freq_diff = abs(self.freq["centre"] - self.sfreq[src])
                sfreq_index = np.argmin(freq_diff)
                # Start and stop indices to process in global frequency axis
                freq_idx0 = np.amax([0, sfreq_index - self.freqside])
                freq_idx1 = np.amin([self.nfreq, sfreq_index + self.freqside + 1])
                # Mask in full frequency axis
                f_mask = np.ones(self.nfreq, dtype=bool)
                f_mask[freq_idx0:freq_idx1] = False
                # Restrict frequency mask to local range
                f_mask = f_mask[self.lo : self.lo + self.ls]

                # TODO: In principle I should be able to skip
                # sources that have no indices to be processed
                # in this rank. I am getting a NaN error, however.
                # I may need an mpiutil.barrier() call before the
                # return statement.
                # if f_mask.all():
                #    # If there are no indices to be processed in
                #    #the local frequency range, skip source.
                #    continue

                # Frequency indices to process in local range
                f_local_indices = np.arange(self.ls, dtype=np.int32)[np.invert(f_mask)]

            if self.is_sstream:
                # Get RA bin this source is closest to.
                # Phasing will actually be done at src position.
                sra_index = np.searchsorted(self.ra, self.sra[src])
            else:
                # Cannot use searchsorted, because RA might not be
                # monotonically increasing. Slower.
                # Notice: in case there is more than one transit,
                # this will pick a single transit quasy-randomly!
                transit_diff = abs(self.ra - self.sra[src])
                sra_index = np.argmin(transit_diff)
                # For now, skip sources that do not transit in the data
                ra_cadence = self.ra[1] - self.ra[0]
                if transit_diff[sra_index] > 1.5 * ra_cadence:
                    continue

            # Compute hour angle array
            ha_array, ra_index_range, ha_mask = self._ha_array(
                self.ra, sra_index, self.sra[src], self.ha_side, self.is_sstream
            )

            # Arrays to store beams and weights for this source
            # for all polarizations prior to combining polarizations
            if self.collapse_ha:
                formed_beam_full = np.zeros((self.npol, self.ls), dtype=np.float64)
                weight_full = np.zeros((self.npol, self.ls), dtype=np.float64)
            else:
                formed_beam_full = np.zeros(
                    (self.npol, self.ls, self.nha), dtype=np.float64
                )
                weight_full = np.zeros((self.npol, self.ls, self.nha), dtype=np.float64)
            # For each polarization
            for pol in range(self.npol):

                # Compute primary beams to be used in the weighting
                primary_beam = self._beamfunc(
                    ha_array[np.newaxis, :],
                    self.process_pol[pol],
                    self.freq_local[:, np.newaxis],
                    dec,
                )

                # Fringestop and sum over products
                # 'beamform' does not normalize sum.
                this_formed_beam = beamform(
                    self.vis[pol],
                    self.sumweight[pol],
                    dec,
                    self.latitude,
                    np.cos(ha_array),
                    np.sin(ha_array),
                    self.bvec[pol][0],
                    self.bvec[pol][1],
                    f_local_indices,
                    ra_index_range,
                )

                sumweight_inrange = self.sumweight[pol][:, ra_index_range, :]
                visweight_inrange = self.visweight[pol][:, ra_index_range, :]

                if self.collapse_ha:
                    # Sum over RA. Does not multiply by weights because
                    # this_formed_beam was never normalized (this avoids
                    # re-work and makes code more efficient).
                    this_sumweight = np.sum(
                        np.sum(sumweight_inrange, axis=-1) * primary_beam, axis=1
                    )

                    formed_beam_full[pol] = np.sum(
                        this_formed_beam * primary_beam, axis=1
                    ) * invert_no_zero(this_sumweight)

                    if self.weight != "inverse_variance":
                        this_weight2 = np.sum(
                            np.sum(
                                sumweight_inrange ** 2
                                * invert_no_zero(visweight_inrange),
                                axis=-1,
                            )
                            * primary_beam ** 2,
                            axis=1,
                        )
                    else:
                        this_weight2 = np.sum(
                            np.sum(sumweight_inrange, axis=-1) * primary_beam ** 2,
                            axis=1,
                        )

                    weight_full[pol] = this_sumweight ** 2 * invert_no_zero(
                        this_weight2
                    )
                else:
                    # Need to divide by weight here for proper
                    # normalization because it is not done in
                    # beamform()
                    this_sumweight = np.sum(sumweight_inrange, axis=-1)
                    # Populate only where ha_mask is true. Zero otherwise.
                    formed_beam_full[pol][
                        :, ha_mask
                    ] = this_formed_beam * invert_no_zero(this_sumweight)
                    if self.weight != "inverse_variance":
                        this_weight2 = np.sum(
                            sumweight_inrange ** 2 * invert_no_zero(visweight_inrange),
                            axis=-1,
                        )
                        # Populate only where ha_mask is true. Zero otherwise.
                        weight_full[pol][
                            :, ha_mask
                        ] = this_sumweight ** 2 * invert_no_zero(this_weight2)
                    else:
                        weight_full[pol][:, ha_mask] = this_sumweight

                # Ensure weights are zero for non-processed frequencies
                weight_full[pol][f_mask] = 0.0

            # Combine polarizations if needed.
            # TODO: For now I am ignoring differences in the X and
            # Y beams and just adding them as is.
            if self.polarization == "I":
                formed_beam_full = np.sum(
                    formed_beam_full * weight_full, axis=0
                ) * invert_no_zero(np.sum(weight_full, axis=0))
                weight_full = np.sum(weight_full, axis=0)
                # Add an axis for the polarization
                if self.collapse_ha:
                    formed_beam_full = np.reshape(formed_beam_full, (1, self.ls))
                    weight_full = np.reshape(weight_full, (1, self.ls))
                else:
                    formed_beam_full = np.reshape(
                        formed_beam_full, (1, self.ls, self.nha)
                    )
                    weight_full = np.reshape(weight_full, (1, self.ls, self.nha))
            elif self.polarization == "stokes":
                # TODO: Not implemented
                pass

            # Populate container.
            formed_beam.beam[src] = formed_beam_full
            formed_beam.weight[src] = weight_full
            if not self.collapse_ha:
                if self.is_sstream:
                    formed_beam.ha[src, :] = ha_array
                else:
                    # Populate only where ha_mask is true.
                    formed_beam.ha[src, ha_mask] = ha_array

        return formed_beam

    def _ha_side(self, data, timetrack=900.0):
        """ Number of RA/time bins to track the source at each side of transit.

        Parameters
        ----------
        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to read time from.
        timetrack : float
            Time in seconds to track at each side of transit.
            Default is 15 minutes.

        Returns
        -------
        ha_side : int
            Number of RA bins to track the source at each side of transit.
        """
        # TODO: Instead of a fixed time for transit, I could have a minimum
        # drop in the beam at a conventional distance from the NCP.
        if "ra" in data.index_map:
            # In seconds
            approx_time_perbin = 24.0 * 3600.0 / float(len(data.index_map["ra"]))
        else:
            approx_time_perbin = data.time[1] - data.time[0]

        # Track for `timetrack` seconds at each side of transit
        return int(timetrack / approx_time_perbin)

    def _ha_array(self, ra, source_ra_index, source_ra, ha_side, is_sstream=True):
        """ Hour angle for each RA/time bin to be processed.

        Also return the indices of these bins in the full RA/time axis.

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
        ra_index_range = np.arange(
            source_ra_index - ha_side, source_ra_index + ha_side + 1, dtype=np.int32
        )
        # Number of RA bins in data.
        nra = len(ra)

        if is_sstream:
            # Wrap RA indices around edges.
            ra_index_range[ra_index_range < 0] += nra
            ra_index_range[ra_index_range >= nra] -= nra
            # Hour angle array (convert to radians)
            ha_array = np.deg2rad(ra[ra_index_range] - source_ra)
            # For later convenience it is better if `ha_array` is
            # in the range -pi to pi instead of 0 to 2pi.
            ha_array = (ha_array + np.pi) % (2.0 * np.pi) - np.pi
            # In this case the ha_mask is trivial
            ha_mask = np.ones(len(ra_index_range), dtype=bool)
        else:
            # Mask-out indices out of range
            ha_mask = (ra_index_range >= 0) & (ra_index_range < nra)
            # Return smaller HA range, and mask.
            ra_index_range = ra_index_range[ha_mask]
            # Hour angle array (convert to radians)
            ha_array = np.deg2rad(ra[ra_index_range] - source_ra)
            # For later convenience it is better if `ha_array` is
            # in the range -pi to pi instead of 0 to 2pi.
            ha_array = (ha_array + np.pi) % (2.0 * np.pi) - np.pi

        return ha_array, ra_index_range, ha_mask

    # TODO: This is very CHIME specific. Should probably be moved somewhere else.
    def _beamfunc(self, ha, pol, freq, dec, zenith=0.70999994):
        """ Simple and fast beam model to be used as beamforming weights.

        Parameters
        ----------
        ha : array or float
            Hour angle (in radians) to compute beam at.
        freq : array or float
            Frequency in MHz
        dec : array or float
            Declination in radians
        pol : int or string
            Polarization index. 0: X, 1: Y, >=2: XY
            or one of 'XX', 'XY', 'YX', 'YY'
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

        pollist = ["XX", "YY", "XY", "YX"]
        if pol in pollist:
            pol = pollist.index(pol)

        def _sig(pp, freq, dec):
            """
            """
            sig_amps = [14.87857614, 9.95746878]
            return sig_amps[pp] / freq / np.cos(dec)

        def _amp(pp, dec, zenith):
            """
            """

            def _flat_top_gauss6(x, A, sig, x0):
                """Flat-top gaussian. Power of 6."""
                return A * np.exp(-abs((x - x0) / sig) ** 6)

            def _flat_top_gauss3(x, A, sig, x0):
                """Flat-top gaussian. Power of 3."""
                return A * np.exp(-abs((x - x0) / sig) ** 3)

            prm_ns_x = np.array([9.97981768e-01, 1.29544939e00, 0.0])
            prm_ns_y = np.array([9.86421047e-01, 8.10213326e-01, 0.0])

            if pp == 0:
                return _flat_top_gauss6(dec - (0.5 * np.pi - zenith), *prm_ns_x)
            else:
                return _flat_top_gauss3(dec - (0.5 * np.pi - zenith), *prm_ns_y)

        ha0 = 0.0
        if pol < 2:
            # XX or YY
            return _amp(pol, dec, zenith) * np.exp(
                -(((ha - ha0) / _sig(pol, freq, dec)) ** 2)
            )
        else:
            # XY or YX
            return (
                _amp(0, dec, zenith)
                * np.exp(-(((ha - ha0) / _sig(0, freq, dec)) ** 2))
                * _amp(1, dec, zenith)
                * np.exp(-(((ha - ha0) / _sig(1, freq, dec)) ** 2))
            ) ** 0.5

    def _process_data(self, data):
        """ Store code for parsing and formating data prior to beamforming.
        """
        # Easy access to communicator
        self.comm_ = data.comm

        # Extract data info
        if "ra" in data.index_map:
            self.is_sstream = True
            self.ra = data.index_map["ra"]
        else:
            self.is_sstream = False
            # Convert data timestamps into LSAs (degrees)
            self.ra = self.telescope.unix_to_lsa(data.time)

        self.freq = data.index_map["freq"]
        self.nfreq = len(self.freq)
        # Ensure data is distributed in freq axis
        data.redistribute(0)

        # Number of RA bins to track each source at each side of transit
        self.ha_side = self._ha_side(data, self.timetrack)
        self.nha = 2 * self.ha_side + 1

        # polmap: indices of each vis product in
        # polarization list: ['XX', 'XY', 'YX', 'YY']
        polmap = polarization_map(data.index_map, self.telescope)
        # Baseline vectors in meters
        bvec_m = baseline_vector(data.index_map, self.telescope)

        # MPI distribution values
        self.lo = data.vis.local_offset[0]
        self.ls = data.vis.local_shape[0]
        self.freq_local = self.freq["centre"][self.lo : self.lo + self.ls]
        # These are to be used when gathering results in the end.
        # Tuple (not list!) of number of frequencies in each rank
        self.fsize = tuple(mpiutil.world.allgather(self.ls))
        # Tuple (not list!) of displacements of each rank array in full array
        self.foffset = tuple(mpiutil.world.allgather(self.lo))

        fullpol = ["XX", "XY", "YX", "YY"]
        # Save subsets of the data for each polarization, changing
        # the ordering to 'C' (needed for the cython part).
        # This doubles the memory usage.
        self.vis, self.visweight, self.bvec, self.sumweight = [], [], [], []
        for pol in self.process_pol:
            pol = fullpol.index(pol)
            polmask = polmap == pol
            # Swap order of product(1) and RA(2) axes, to reduce striding
            # through memory later on.
            self.vis.append(
                np.copy(np.moveaxis(data.vis[:, polmask, :], 1, 2), order="C")
            )
            # Restrict visweight to the local frequencies
            self.visweight.append(
                np.copy(
                    np.moveaxis(
                        data.weight[self.lo : self.lo + self.ls][:, polmask, :], 1, 2
                    ).astype(np.float64),
                    order="C",
                )
            )
            # Multiply bvec_m by frequencies to get vector in wavelengths.
            # Shape: (2, nfreq_local, nvis), for each pol.
            self.bvec.append(
                np.copy(
                    bvec_m[:, np.newaxis, polmask]
                    * self.freq_local[np.newaxis, :, np.newaxis]
                    * 1e6
                    / C,
                    order="C",
                )
            )
            if self.weight == "inverse_variance":
                # Weights for sum are just the visibility weights
                self.sumweight.append(self.visweight[-1])
            else:
                # Ensure zero visweights result in zero sumweights
                this_sumweight = (self.visweight[-1] > 0.0).astype(np.float64)
                ssi = data.input_flags[:]
                ssp = data.index_map["prod"][:]
                sss = data.reverse_map["stack"]["stack"][:]
                nstack = data.vis.shape[1]
                # this redundancy takes into account input flags.
                # It has shape (nstack, ntime)
                redundancy = np.moveaxis(
                    calculate_redundancy(ssi, ssp, sss, nstack)[polmask].astype(
                        np.float64
                    ),
                    0,
                    1,
                )[np.newaxis, :, :]
                # redundancy = (self.telescope.redundancy[polmask].
                #        astype(np.float64)[np.newaxis, np.newaxis, :])
                this_sumweight *= redundancy
                if self.weight == "uniform":
                    this_sumweight = (this_sumweight > 0.0).astype(np.float64)
                self.sumweight.append(np.copy(this_sumweight, order="C"))


class BeamForm(BeamFormBase):
    """ BeamForm for a single source catalog and multiple visibility datasets.

    """

    def setup(self, manager, source_cat):
        """ Parse the source catalog and performs the generic setup.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.

        source_cat : :class:`containers.SourceCatalog`
            Catalog of points to beamform at.

        """
        super(BeamForm, self).setup(manager)

        # Extract source catalog information
        self.source_cat = source_cat
        # Number of pointings
        self.nsource = len(self.source_cat["position"])
        # Pointings RA and Dec
        self.sdec = np.deg2rad(self.source_cat["position"]["dec"][:])
        self.sra = self.source_cat["position"]["ra"]
        if self.freqside is not None:
            # Frequency of each source.
            self.sfreq = NU21 / (self.source_cat["redshift"]["z"][:] + 1.0)  # MHz

    def process(self, data):
        """ Parse the visibility data and beamforms all sources.

        Parameters
        ----------
        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to beamform on.

        Returns
        -------
        formed_beam : `containers.FormedBeam` or `containers.FormedBeamHA`
            Formed beams at each source.
        """
        # Process and make available various data
        self._process_data(data)

        # Call generic process method.
        return super(BeamForm, self).process()


class BeamFormCat(BeamFormBase):
    """ BeamForm for multiple source catalogs and a single visibility dataset.

    """

    def setup(self, manager, data):
        """ Parse the visibility data and performs the generic setup.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.

        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to beamform on.

        """
        super(BeamFormCat, self).setup(manager)

        # Process and make available various data
        self._process_data(data)

    def process(self, source_cat):
        """ Parse the source catalog and beamforms all sources.

        Parameters
        ----------
        source_cat : :class:`containers.SourceCatalog`
            Catalog of points to beamform at.

        Returns
        -------
        formed_beam : `containers.FormedBeam` or `containers.FormedBeamHA`
            Formed beams at each source.
        """
        # Source catalog to beamform at
        self.source_cat = source_cat
        # Number of pointings
        self.nsource = len(self.source_cat["position"])
        # Pointings RA and Dec
        self.sdec = np.deg2rad(self.source_cat["position"]["dec"][:])
        self.sra = self.source_cat["position"]["ra"]
        if self.freqside is not None:
            # Frequency of each source.
            self.sfreq = NU21 / (self.source_cat["redshift"]["z"][:] + 1.0)  # MHz

        # Call generic process method.
        return super(BeamFormCat, self).process()

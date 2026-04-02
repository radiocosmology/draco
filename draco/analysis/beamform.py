"""Beamform visibilities to the location of known sources."""

import healpy
import numpy as np
import scipy.interpolate
from caput import config
from caput.astro import constants
from caput.astro import skyfield as skf
from caput.astro import time as ctime
from caput.pipeline import tasklib
from skyfield.api import Angle, Star

from draco.analysis.sidereal import _search_nearest

from ..core import containers, io
from ..util import interferometry
from ..util._fast_tools import beamform
from ..util.tools import (
    baseline_vector,
    calculate_redundancy,
    correct_phase_wrap,
    find_contiguous_slices,
    invert_no_zero,
    polarization_map,
)

# Constants
NU21 = constants.nu21
C = constants.c


class BeamFormBase(tasklib.base.ContainerTask):
    """Base class for beam forming tasks.

    Defines a few useful methods. Not to be used directly
    but as parent class for BeamForm and BeamFormCat.

    Attributes
    ----------
    collapse_ha : bool
        Sum over hour-angle/time to complete the beamforming. Default is True.
    polarization : string
        Determines the polarizations that will be output:
            - 'I' : Stokes I only.
            - 'full' : 'XX', 'XY', 'YX' and 'YY' in this order. (default)
            - 'copol' : 'XX' and 'YY' only.
            - 'stokes' : 'I', 'Q', 'U' and 'V' in this order. Not implemented.
    weight : string
        How to weight the redundant baselines when adding:
            - 'natural' : each baseline weighted by its redundancy (default)
            - 'uniform' : each baseline given equal weight
            - 'inverse_variance' : each baseline weighted by the weight attribute
    no_beam_model : string
        Do not include a primary beam factor in the beamforming
        weights, i.e., use uniform weighting as a function of hour angle
        and declination.
    timetrack : float
        How long (in seconds) to track sources at each side of transit.
        Default is 900 seconds.  Total transit time will be 2 * timetrack.
    variable_timetrack : bool
        Scale the total time to track each source by the secant of the
        source declination, so that all sources are tracked through
        the same angle on the sky.  Default is False.
    freqside : int
        Number of frequencies to process at each side of the source.
        Default (None) processes all frequencies.
    """

    collapse_ha = config.Property(proptype=bool, default=True)
    polarization = config.enum(["I", "full", "copol", "stokes"], default="full")
    weight = config.enum(["natural", "uniform", "inverse_variance"], default="natural")
    no_beam_model = config.Property(proptype=bool, default=False)
    timetrack = config.Property(proptype=float, default=900.0)
    variable_timetrack = config.Property(proptype=bool, default=False)
    freqside = config.Property(proptype=int, default=None)
    data_available = True

    def setup(self, manager):
        """Generic setup method.

        To be complemented by specific setup methods in daughter tasks.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.
        """
        # Get the TransitTelescope object
        self.telescope = io.get_telescope(manager)
        self.latitude = np.deg2rad(self.telescope.latitude)

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

        self.npol = len(self.process_pol)

        self.map_pol_feed = {
            pstr: list(self.telescope.polarisation).index(pstr) for pstr in ["X", "Y"]
        }

        # Ensure that if we are using variable time tracking,
        # then we are also collapsing over hour angle.
        if self.variable_timetrack:
            if self.collapse_ha:
                self.log.info(
                    "Tracking source for declination dependent amount of time "
                    f"[{self.timetrack:0.0f} seconds at equator]"
                )
            else:
                raise NotImplementedError(
                    "Must collapse over hour angle if tracking "
                    "sources for declination dependent "
                    "amount of time."
                )

        else:
            self.log.info(
                f"Tracking source for fixed amount of time [{self.timetrack:0.0f} seconds]"
            )

    def process(self):
        """Generic process method.

        Performs all the beamforming, but not the data parsing.
        To be complemented by specific process methods in daughter tasks.

        Returns
        -------
        formed_beam : `containers.FormedBeam` or `containers.FormedBeamHA`
            Formed beams at each source. Shape depends on parameter
            `collapse_ha`.
        """
        # Perform data dependent beam initialization
        self._initialize_beam_with_data()

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
                ha=np.arange(self.nha, dtype=np.int64),
                object_id=self.source_cat.index_map["object_id"],
                pol=np.array(self.return_pol),
                distributed=True,
            )
            # Initialize container to zeros.
            formed_beam.ha[:] = 0.0

        formed_beam.attrs["tag"] = "_".join(
            [tag for tag in [self.tag_data, self.tag_catalog] if tag is not None]
        )

        # Initialize container to zeros.
        formed_beam.beam[:] = 0.0
        formed_beam.weight[:] = 0.0

        # Copy catalog information
        formed_beam["position"][:] = self.source_cat["position"][:]
        if "redshift" in self.source_cat:
            formed_beam.add_dataset("redshift")
            formed_beam["redshift"][:] = self.source_cat["redshift"][:]

        # Ensure container is distributed in frequency
        formed_beam.redistribute("freq")

        if self.freqside is None:
            # Indices of local frequency axis. Full axis if freqside is None.
            f_local_indices = np.arange(self.ls, dtype=np.int32)
            f_mask = np.zeros(self.ls, dtype=bool)

        fbb = formed_beam.beam[:]
        fbw = formed_beam.weight[:]

        # For each source, beamform and populate container.
        for src in range(self.nsource):
            if src % 1000 == 0:
                self.log.info(f"Source {src}/{self.nsource}")

            # Declination of this source
            dec = np.radians(self.sdec[src])

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
                f_mask = f_mask[self.lo : (self.lo + self.ls)]

                # TODO: In principle I should be able to skip
                # sources that have no indices to be processed
                # in this rank. I am getting a NaN error, however.
                # I may need an mpiutil.barrier() call before the
                # return statement.
                if f_mask.all():
                    # If there are no indices to be processed in
                    # the local frequency range, skip source.
                    continue

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
                # this will pick a single transit quasi-randomly!
                transit_diff = abs(self.ra - self.sra[src])
                sra_index = np.argmin(transit_diff)
                # For now, skip sources that do not transit in the data
                ra_cadence = self.ra[1] - self.ra[0]
                if transit_diff[sra_index] > 1.5 * ra_cadence:
                    continue

            if self.variable_timetrack:
                ha_side = int(self.ha_side / np.cos(dec))
            else:
                ha_side = int(self.ha_side)

            # Compute hour angle array
            ha_array, ra_index_range, ha_mask = self._ha_array(
                self.ra, sra_index, self.sra[src], ha_side, self.is_sstream
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

            # Loop over polarisations
            for pol, pol_str in enumerate(self.process_pol):
                primary_beam = self._beamfunc(pol_str, dec, ha_array)

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
                        np.sum(sumweight_inrange, axis=-1) * primary_beam**2, axis=1
                    )

                    formed_beam_full[pol] = np.sum(
                        this_formed_beam * primary_beam, axis=1
                    ) * invert_no_zero(this_sumweight)

                    if self.weight != "inverse_variance":
                        this_weight2 = np.sum(
                            np.sum(
                                sumweight_inrange**2
                                * invert_no_zero(visweight_inrange),
                                axis=-1,
                            )
                            * primary_beam**2,
                            axis=1,
                        )

                        weight_full[pol] = this_sumweight**2 * invert_no_zero(
                            this_weight2
                        )

                    else:
                        weight_full[pol] = this_sumweight

                else:
                    # Need to divide by weight here for proper
                    # normalization because it is not done in
                    # beamform()
                    this_sumweight = np.sum(sumweight_inrange, axis=-1)
                    # Populate only where ha_mask is true. Zero otherwise.
                    formed_beam_full[pol][:, ha_mask] = (
                        this_formed_beam * invert_no_zero(this_sumweight)
                    )
                    if self.weight != "inverse_variance":
                        this_weight2 = np.sum(
                            sumweight_inrange**2 * invert_no_zero(visweight_inrange),
                            axis=-1,
                        )
                        # Populate only where ha_mask is true. Zero otherwise.
                        weight_full[pol][:, ha_mask] = (
                            this_sumweight**2 * invert_no_zero(this_weight2)
                        )
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
            fbb[src] = formed_beam_full

            # Scale the weights by a factor of 2 to account for the fact that we
            # have taken the real-component of the fringestopped visibility, which
            # has a variance that is 1/2 the variance of the complex visibility
            # that was encoded in our original weight dataset.
            fbw[src] = 2.0 * weight_full

            if not self.collapse_ha:
                if self.is_sstream:
                    formed_beam.ha[src, :] = ha_array
                else:
                    # Populate only where ha_mask is true.
                    formed_beam.ha[src, ha_mask] = ha_array

        return formed_beam

    def process_finish(self):
        """Clear lists holding copies of data.

        These lists will persist beyond this task being done, so
        the data stored there will continue to use memory.
        """
        for attr in ["vis", "visweight", "bvec", "sumweight"]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def _ha_array(self, ra, source_ra_index, source_ra, ha_side, is_sstream=True):
        """Hour angle for each RA/time bin to be processed.

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

    def _initialize_beam_with_data(self):
        """Beam initialization that requires data.

        This is called at the start of the process method
        and can be overridden to perform any beam initialization
        that requires the data and catalog to be parsed first.
        """
        # Find the index of the local frequencies in
        # the frequency axis of the telescope instance
        if not self.no_beam_model:
            self.freq_local_telescope_index = np.array(
                [
                    np.argmin(np.abs(nu - self.telescope.frequencies))
                    for nu in self.freq_local
                ]
            )

    def _beamfunc(self, pol, dec, ha):
        """Calculate the primary beam at the location of a source as it transits.

        Uses the frequencies in the freq_local_telescope_index attribute.

        Parameters
        ----------
        pol : str
            String specifying the polarisation,
            either 'XX', 'XY', 'YX', or 'YY'.
        dec : float
            The declination of the source in radians.
        ha : np.ndarray[nha,]
            The hour angle of the source in radians.

        Returns
        -------
        primary_beam : np.ndarray[nfreq, nha]
            The primary beam as a function of frequency and hour angle
            at the sources declination for the requested polarisation.
        """
        nfreq = self.freq_local.size

        if self.no_beam_model:
            return np.ones((nfreq, ha.size), dtype=np.float64)

        angpos = np.array([(0.5 * np.pi - dec) * np.ones_like(ha), ha]).T

        primary_beam = np.zeros((nfreq, ha.size), dtype=np.float64)

        for ff, freq in enumerate(self.freq_local_telescope_index):
            bii = self.telescope.beam(self.map_pol_feed[pol[0]], freq, angpos)

            if pol[0] != pol[1]:
                bjj = self.telescope.beam(self.map_pol_feed[pol[1]], freq, angpos)
            else:
                bjj = bii

            primary_beam[ff] = np.sum(bii * bjj.conjugate(), axis=1)

        return primary_beam

    def _process_data(self, data):
        """Store code for parsing and formating data prior to beamforming."""
        # Easy access to communicator
        self.comm_ = data.comm

        self.tag_data = data.attrs["tag"] if "tag" in data.attrs else None

        # Extract data info
        if "ra" in data.index_map:
            self.is_sstream = True
            self.ra = data.index_map["ra"]

            # Calculate the epoch for the data so we can calculate the correct
            # CIRS coordinates
            if "lsd" not in data.attrs:
                raise ValueError(
                    "SiderealStream must have an LSD attribute to calculate the epoch."
                )

            lsd = np.mean(data.attrs["lsd"])
            self.epoch = self.telescope.lsd_to_unix(lsd)

            dt = 240.0 * constants.SIDEREAL_S * np.median(np.abs(np.diff(self.ra)))

        else:
            self.is_sstream = False
            # Convert data timestamps into LSAs (degrees)
            self.ra = self.telescope.unix_to_lsa(data.time)
            self.epoch = data.time.mean()

            dt = np.median(np.abs(np.diff(data.time)))

        self.freq = data.index_map["freq"]
        self.nfreq = len(self.freq)
        # Ensure data is distributed in freq axis
        data.redistribute(0)

        # Number of RA bins to track each source at each side of transit
        self.ha_side = self.timetrack / dt
        self.nha = 2 * int(self.ha_side) + 1

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
        self.fsize = tuple(data.comm.allgather(self.ls))
        # Tuple (not list!) of displacements of each rank array in full array
        self.foffset = tuple(data.comm.allgather(self.lo))

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

    def _process_catalog(self, catalog):
        """Process the catalog to get CIRS coordinates at the correct epoch.

        Note that `self._process_data` must have been called before this.
        """
        if "position" not in catalog:
            raise ValueError("Input is missing a position table.")

        if not hasattr(self, "epoch"):
            self.log.warning("Epoch not set. Was the requested data not available?")
            self.data_available = False
            return

        coord = catalog.attrs.get("coordinates", None)
        if coord == "CIRS":
            self.log.info("Catalog already in CIRS coordinates.")
            self.sra = catalog["position"]["ra"]
            self.sdec = catalog["position"]["dec"]

        else:
            self.log.info("Converting catalog from ICRS to CIRS coordinates.")
            self.sra, self.sdec = icrs_to_cirs(
                catalog["position"]["ra"], catalog["position"]["dec"], self.epoch
            )

        if self.freqside is not None:
            if "redshift" not in catalog:
                raise ValueError("Input is missing a required redshift table.")
            self.sfreq = NU21 / (catalog["redshift"]["z"][:] + 1.0)  # MHz

        self.source_cat = catalog
        self.nsource = len(self.sra)

        self.tag_catalog = catalog.attrs["tag"] if "tag" in catalog.attrs else None


class BeamForm(BeamFormBase):
    """BeamForm for a single source catalog and multiple visibility datasets."""

    def setup(self, manager, source_cat):
        """Parse the source catalog and performs the generic setup.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.

        source_cat : :class:`containers.SourceCatalog`
            Catalog of points to beamform at.

        """
        super().setup(manager)
        self.catalog = source_cat

    def process(self, data):
        """Parse the visibility data and beamforms all sources.

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
        self._process_catalog(self.catalog)

        if not self.data_available:
            return None

        # Call generic process method.
        return super().process()


class BeamFormCat(BeamFormBase):
    """BeamForm for multiple source catalogs and a single visibility dataset."""

    def setup(self, manager, data):
        """Parse the visibility data and performs the generic setup.

        Parameters
        ----------
        manager : either `ProductManager`, `BeamTransfer` or `TransitTelescope`
            Contains a TransitTelescope object describing the telescope.

        data : `containers.SiderealStream` or `containers.TimeStream`
            Data to beamform on.

        """
        super().setup(manager)

        # Process and make available various data
        self._process_data(data)

    def process(self, source_cat):
        """Parse the source catalog and beamforms all sources.

        Parameters
        ----------
        source_cat : :class:`containers.SourceCatalog`
            Catalog of points to beamform at.

        Returns
        -------
        formed_beam : `containers.FormedBeam` or `containers.FormedBeamHA`
            Formed beams at each source.
        """
        self._process_catalog(source_cat)

        if not self.data_available:
            return None

        # Call generic process method.
        return super().process()


class BeamFormExternalMixin:
    """Base class for tasks that beamform using an external model of the primary beam.

    The primary beam is provided to the task during setup.  Do not use this class
    directly, instead use BeamFormExternal and BeamFormExternalCat.
    """

    def setup(self, beam, *args):
        """Initialize the beam.

        Parameters
        ----------
        beam : GridBeam
            Model for the primary beam.
        args : optional
            Additional argument to pass to the super class
        """
        super().setup(*args)
        self._initialize_beam(beam)

    def _initialize_beam(self, beam):
        """Initialize based on the beam container type.

        Parameters
        ----------
        beam : GridBeam
            Container holding the model for the primary beam.
            Currently only accepts GridBeam type containers.
        """
        if isinstance(beam, containers.GridBeam):
            self._initialize_grid_beam(beam)
            self._beamfunc = self._grid_beam

        else:
            raise ValueError(f"Do not recognize beam container: {beam.__class__}")

    def _initialize_beam_with_data(self):
        """Ensure that the beam and visibilities have the same frequency axis."""
        if not np.array_equal(self.freq_local, self._beam_freq):
            raise RuntimeError("Beam and visibility frequency axes do not match.")

    def _initialize_grid_beam(self, gbeam):
        """Create an interpolator for a GridBeam.

        Parameters
        ----------
        gbeam : GridBeam
            Model for the primary beam on a celestial grid where
            (theta, phi) = (declination, hour angle) in degrees.  The beam
            must be in power units and must have a length 1 input axis that
            contains the "baseline averaged" beam, which will be applied to
            all baselines of a given polarisation.
        """
        # Make sure the beam is in celestial coordinates
        if gbeam.coords != "celestial":
            raise RuntimeError(
                "GridBeam must be converted to celestial coordinates for beamforming."
            )

        # Make sure there is a single beam to use for all inputs
        if gbeam.input.size > 1:
            raise NotImplementedError(
                "Do not support input-dependent beams at the moment."
            )

        # Distribute over frequencies, extract local frequencies
        gbeam.redistribute("freq")

        lo = gbeam.beam.local_offset[0]
        nfreq = gbeam.beam.local_shape[0]
        self._beam_freq = gbeam.freq[lo : lo + nfreq]

        # Find the relevant indices into the polarisation axis
        process_pol = getattr(self, "process_pol", list(gbeam.pol))
        ipol = np.array([list(gbeam.pol).index(pstr) for pstr in process_pol])
        npol = ipol.size
        self._beam_pol = [gbeam.pol[ip] for ip in ipol]

        # Extract beam
        flag = gbeam.weight[:].local_array[:, :, 0][:, ipol] > 0.0
        beam = np.where(flag, gbeam.beam[:].local_array[:, :, 0][:, ipol].real, 0.0)

        # Convert the declination and hour angle axis to radians, make sure they are sorted
        ha = (gbeam.phi + 180.0) % 360.0 - 180.0
        isort = np.argsort(ha)
        ha = np.radians(ha[isort])

        dec = np.radians(gbeam.theta)

        # Create a 2D interpolator for the beam at each frequency and polarisation
        self._beam = [
            [
                scipy.interpolate.RectBivariateSpline(dec, ha, beam[ff, pp][:, isort])
                for pp in range(npol)
            ]
            for ff in range(nfreq)
        ]

        # Create a similair interpolator for the flag array
        self._beam_flag = [
            [
                scipy.interpolate.RectBivariateSpline(
                    dec, ha, flag[ff, pp][:, isort].astype(np.float32)
                )
                for pp in range(npol)
            ]
            for ff in range(nfreq)
        ]

        self.log.info("Grid beam initialized.")

    def _grid_beam(self, pol, dec, ha):
        """Interpolate a GridBeam to the requested declination and hour angles.

        Parameters
        ----------
        pol : str
            String specifying the polarisation,
            either 'XX', 'XY', 'YX', or 'YY'.
        dec : float
            The declination of the source in radians.
        ha : np.ndarray[nha,]
            The hour angle of the source in radians.

        Returns
        -------
        primay_beam : np.ndarray[nfreq, nha]
            The primary beam as a function of frequency and hour angle
            at the sources declination for the requested polarisation.
        """
        pp = self._beam_pol.index(pol)

        primay_beam = np.array(
            [self._beam[ff][pp](dec, ha)[0] for ff in range(self._beam_freq.size)]
        )

        # If the interpolated flags deviate from 1.0, then we mask
        # the interpolated beam, since some fraction the underlying
        # data used to construct the interpolator was masked.
        flag = np.array(
            [
                np.abs(self._beam_flag[ff][pp](dec, ha)[0] - 1.0) < 0.01
                for ff in range(self._beam_freq.size)
            ]
        )

        return np.where(flag, primay_beam, 0.0)


class BeamFormExternal(BeamFormExternalMixin, BeamForm):
    """Beamform a single catalog and multiple datasets using an external beam model.

    The setup method requires [beam, manager, source_cat] as arguments.
    """


class BeamFormExternalCat(BeamFormExternalMixin, BeamFormCat):
    """Beamform multiple catalogs and a single dataset using an external beam model.

    The setup method requires [beam, manager, data] as arguments.
    """


class RingMapBeamForm(tasklib.base.ContainerTask):
    """Beamform by extracting the pixel containing each source form a RingMap.

    This is significantly faster than `Beamform` or `BeamformCat` with the caveat
    that they can beamform exactly on a source whereas this task is at the mercy of
    what was done to produce the `RingMap` (use `DeconvolveHybridM` for best
    results).

    Unless it has an explicit `lsd` attribute, the ring map is assumed to be in the
    same coordinate epoch as the catalog. If it does, the input catalog is assumed to be
    in ICRS and then is precessed to the CIRS coordinates in the epoch of the map.
    """

    def setup(self, telescope: io.TelescopeConvertible, ringmap: containers.RingMap):
        """Set the telescope object.

        Parameters
        ----------
        telescope
            The telescope object to use.
        ringmap
            The ringmap to extract the sources from. See the class documentation for how
            the epoch is determined.
        """
        self.telescope = io.get_telescope(telescope)
        self.ringmap = ringmap

    def process(self, catalog: containers.SourceCatalog) -> containers.FormedBeam:
        """Extract sources from a ringmap.

        Parameters
        ----------
        catalog
            The catalog to extract sources from.

        Returns
        -------
        sources
            The source spectra.
        """
        ringmap = self.ringmap

        src_ra, src_dec = self._process_catalog(catalog)

        # Get the pixel indices
        ra_ind, el_ind, src_ind = self._source_ind(src_ra, src_dec)

        # Container to hold the formed beams
        formed_beam = containers.FormedBeam(
            object_id=catalog.index_map["object_id"][src_ind],
            axes_from=ringmap,
            attrs_from=catalog,
            distributed=True,
        )

        # Initialize container to zeros.
        formed_beam.beam[:] = 0.0
        formed_beam.weight[:] = 0.0

        # Copy catalog information
        formed_beam["position"][:] = catalog["position"][src_ind]
        if "redshift" in catalog:
            formed_beam.add_dataset("redshift")
            formed_beam["redshift"][:] = catalog["redshift"][src_ind]

        # Ensure containers are distributed in frequency
        formed_beam.redistribute("freq")
        ringmap.redistribute("freq")

        has_weight = "weight" in ringmap.datasets

        # Dereference the datasets
        fbb = formed_beam.beam[:]
        fbw = formed_beam.weight[:]
        rmm = ringmap.map[:]
        rmw = ringmap.weight[:] if has_weight else invert_no_zero(ringmap.rms[:]) ** 2

        # Loop over sources and extract the polarised pencil beams containing them from
        # the ringmaps
        for si, (ri, ei) in enumerate(zip(ra_ind, el_ind)):
            fbb[si] = rmm[0, :, :, ri, ei]
            fbw[si] = rmw[:, :, ri, ei] if has_weight else rmw[:, :, ri]

        return formed_beam

    def _process_catalog(
        self, catalog: containers.SourceCatalog
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the current epoch coordinates of the catalog."""
        if "position" not in catalog:
            raise ValueError("Input is missing a position table.")

        # Calculate the epoch for the data so we can calculate the correct
        # CIRS coordinates
        if "lsd" not in self.ringmap.attrs:
            self.log.info(
                "Input map has no epoch set, assuming that it matches the catalog."
            )
            src_ra, src_dec = catalog["position"]["ra"], catalog["position"]["dec"]

        else:
            lsd = (
                self.ringmap.attrs["lsd"][0]
                if isinstance(self.ringmap.attrs["lsd"], np.ndarray)
                else self.ringmap.attrs["lsd"]
            )
            epoch = self.telescope.lsd_to_unix(lsd)

            # Get the source positions at the current epoch
            src_ra, src_dec = icrs_to_cirs(
                catalog["position"]["ra"], catalog["position"]["dec"], epoch
            )

        return src_ra, src_dec

    def _source_ind(
        self, src_ra: np.ndarray, src_dec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the indices of the pixels nearest to an array of source coordinates.

        Given arrays of source right ascensions and declinations, this method finds
        the indices into the RA and elevation axis of a ringmap corresponding to the
        nearest pixels.  It excludes any sources that fall outside the ringmap.

        Parameters
        ----------
        src_ra : np.ndarray[nsource,]
            Source right ascensions in degrees.
        src_dec : np.ndarray[nsource,]
            Source declinations in degrees.

        Returns
        -------
        ra_ind : np.ndarray[nvalid,]
            Indices into the RA axis corresponding to the nearest pixel
            for valid sources.
        el_ind : np.ndarray[nvalid,]
            Indices into the elevation axis corresponding to the nearest pixel
            for valid sources.
        src_ind : np.ndarray[nvalid,]
            Indices of the sources that fall within the map's RA and elevation range.
        """
        src_el = np.sin(np.radians(src_dec - self.telescope.latitude))

        # Get the map grid in RA and sin(ZA)
        ra = self.ringmap.index_map["ra"]
        el = self.ringmap.index_map["el"]

        delta_ra = np.median(np.abs(np.diff(ra)))
        delta_el = np.median(np.abs(np.diff(el)))

        # Get the source indices in RA
        # NOTE: that we need to take into account that sources might be less than 360
        # deg, but still closer to ind=0
        ra_ind = _search_nearest(np.append(ra, 360.0 + ra[0]), src_ra) % ra.size

        ra_sep = src_ra - ra[ra_ind]
        ra_sep = (ra_sep + 180.0) % 360.0 - 180.0

        # Get the indices for the ZA direction
        el_ind = _search_nearest(el, src_el)

        el_sep = src_el - el[el_ind]

        # Make sure no source are outside of the range covered by the map
        src_flag = (np.abs(ra_sep) > (0.5 * delta_ra)) | (
            np.abs(el_sep) > (0.5 * delta_el)
        )

        if np.any(src_flag):
            self.log.info(
                f"There are {np.sum(src_flag)} (of {src_flag.size}) "
                "sources outside of the RA and Declination range covered by the map."
            )

        src_ind = np.flatnonzero(~src_flag)
        ra_ind = ra_ind[src_ind]
        el_ind = el_ind[src_ind]

        return ra_ind, el_ind, src_ind


class RingMapStack2D(RingMapBeamForm):
    """Stack RingMap's on sources directly.

    Parameters
    ----------
    num_ra, num_dec : int
        The number of RA and DEC pixels to stack either side of the source.
    num_freq : int
        Number of final frequency channels either side of the source redshift to
        stack.
    freq_width : float
        Length of frequency interval either side of source to use in MHz.
        If set to zero, then will default to use the ringmaps native resolution.
    weight : {"patch", "dec", "enum"}
        How to weight the data. If `"input"` the data is weighted on a pixel by pixel
        basis according to the input data. If `"patch"` then the inverse of the
        variance of the extracted patch is used. If `"dec"` then the inverse variance
        of each declination strip is used.
    """

    num_ra = config.Property(proptype=int, default=10)
    num_dec = config.Property(proptype=int, default=10)
    num_freq = config.Property(proptype=int, default=256)
    freq_width = config.Property(proptype=float, default=0.0)
    weight = config.enum(["patch", "dec", "input"], default="input")

    def process(self, catalog: containers.SourceCatalog) -> containers.FormedBeam:
        """Extract sources from a ringmap.

        Parameters
        ----------
        catalog
            The catalog to extract sources from.

        Returns
        -------
        sources
            The source spectra.
        """
        from mpi4py import MPI

        ringmap = self.ringmap

        # Get the current epoch catalog position
        src_ra, src_dec = self._process_catalog(catalog)
        src_z = catalog["redshift"]["z"]

        # Get the pixel indices
        ra_ind, el_ind, src_ind = self._source_ind(src_ra, src_dec)

        # Exclude sources outside of the map
        src_z = src_z[src_ind]

        # Ensure containers are distributed in frequency
        ringmap.redistribute("freq")

        # Get the frequencies on this rank
        fs = ringmap.map.local_offset[2]
        fe = fs + ringmap.map.local_shape[2]
        local_freq = ringmap.freq[fs:fe]

        # RA and EL grid info
        ra = ringmap.index_map["ra"]
        el = ringmap.index_map["el"]

        dra = np.median(np.abs(np.diff(ra)))
        dell = np.median(np.abs(np.diff(el)))

        nra = ra.size
        nel = el.size

        # Determine if the RA axis wraps around from 360 to 0 deg
        tol = dra / 100.0
        ra_wraps = np.isclose(ra[-1] + dra, 360.0, atol=tol) and np.isclose(
            ra[0], 0.0, atol=tol
        )

        # Calculate the frequencies bins to use
        nbins = 2 * self.num_freq + 1
        if self.freq_width > 0:
            bin_edges = np.linspace(
                -self.freq_width, self.freq_width, nbins + 1, endpoint=True
            )
        else:
            df = np.median(np.abs(np.diff(ringmap.freq)))
            bin_edges = (np.arange(-self.num_freq, self.num_freq + 2) - 0.5) * df

        # Calculate the edges of the frequency distribution, sources outside this range
        # will be dropped
        global_fmin = ringmap.freq.min()
        global_fmax = ringmap.freq.max()

        # Dereference the datasets
        rmm = ringmap.map[:].local_array
        if "weight" in ringmap.datasets:
            rmw = ringmap.weight[:].local_array
        else:
            rmw = invert_no_zero(ringmap.rms[:].local_array) ** 2
            rmw = rmw[..., np.newaxis] * np.ones(el.size)

        # Create temporary array to accumulate into
        wstack = np.zeros(
            (nbins + 2, len(ringmap.pol), 2 * self.num_ra + 1, 2 * self.num_dec + 1)
        )
        weight = np.zeros(
            (nbins + 2, len(ringmap.pol), 2 * self.num_ra + 1, 2 * self.num_dec + 1)
        )

        rmvar = rmm[0].var(axis=2)
        w_global = invert_no_zero(np.where(rmvar < 3e-7, 0.0, rmvar))

        # Loop over sources and extract the polarised pencil beams containing them from
        # the ringmaps
        for si, (ri, ei, z) in enumerate(zip(ra_ind, el_ind, src_z)):
            source_freq = NU21 / (1 + z)

            if source_freq > global_fmax or source_freq < global_fmin:
                continue

            # Get bin indices
            bin_ind = np.digitize(local_freq - source_freq, bin_edges)

            # Get the el slice to extract the enclosing angular region
            estart, estop = ei - self.num_dec, ei + self.num_dec + 1
            ei_start, ei_stop = max(estart, 0), min(estop, nel)

            ei_slice = slice(ei_start, ei_stop)
            ei_out = slice(max(0, -estart), self.num_dec * 2 + 1 - max(0, estop - nel))

            # Get the ra slice to extract the enclosing angular region
            rstart, rstop = ri - self.num_ra, ri + self.num_ra + 1

            if ra_wraps and ((rstart < 0) or (rstop > nra)):
                # The source is near one of the ends of the RA axis,
                # so we will need to wrap around.  Do this by creating
                # two slices, one at each end, and concatenating them.
                ri_slice = [slice((nra + rstart) % nra, nra), slice(0, rstop % nra)]
                ri_out = slice(None)

                b = np.concatenate(
                    tuple(rmm[0, :, :, slc, ei_slice] for slc in ri_slice), axis=2
                )
                w = np.concatenate(
                    tuple(rmw[:, :, slc, ei_slice] for slc in ri_slice), axis=2
                )

            else:
                ri_start, ri_stop = max(rstart, 0), min(rstop, nra)
                ri_slice = slice(ri_start, ri_stop)

                ri_out = slice(
                    max(0, -rstart), self.num_ra * 2 + 1 - max(0, rstop - nra)
                )

                b = rmm[0, :, :, ri_slice, ei_slice]
                w = rmw[:, :, ri_slice, ei_slice]

            # Determine the weight that is given to this source
            if self.weight == "patch":
                # Replace the weights with the variance of the patch
                w = (w != 0) * invert_no_zero(b.var(axis=(2, 3)))[
                    :, :, np.newaxis, np.newaxis
                ]
            elif self.weight == "dec":
                # w = (w != 0) * invert_no_zero(b.var(axis=2))[:, :, np.newaxis, :]
                w = (w != 0) * w_global[:, :, np.newaxis, ei_slice]

            bw = b * w

            # TODO: this is probably slow so should be moved into Cython
            for lfi, bi in enumerate(bin_ind):
                wstack[bi, :, ri_out, ei_out] += bw[:, lfi]
                weight[bi, :, ri_out, ei_out] += w[:, lfi]

        # Arrays to reduce the data into
        wstack_all = np.zeros_like(wstack)
        weight_all = np.zeros_like(weight)

        self.comm.Allreduce(wstack, wstack_all, op=MPI.SUM)
        self.comm.Allreduce(weight, weight_all, op=MPI.SUM)

        stack_all = wstack_all * invert_no_zero(weight_all)

        # Create the axes of the 3D grid
        delta_f = np.zeros(nbins, dtype=[("centre", float), ("width", float)])
        delta_f["centre"] = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        delta_f["width"] = bin_edges[1:] - bin_edges[:-1]

        delta_ra = np.arange(-self.num_ra, self.num_ra + 1) * dra
        delta_dec = np.degrees(
            np.arcsin(np.arange(-self.num_dec, self.num_dec + 1) * dell)
        )

        # Create the container to store the data in
        stack = containers.Stack3D(
            freq=delta_f,
            delta_ra=delta_ra,
            delta_dec=delta_dec,
            axes_from=ringmap,
            attrs_from=ringmap,
        )
        stack.attrs["tag"] = catalog.attrs["tag"]
        stack.stack[:] = stack_all[1:-1].transpose((1, 2, 3, 0))
        stack.weight[:] = weight_all[1:-1].transpose((1, 2, 3, 0))

        return stack


class HybridVisBeamForm(tasklib.base.ContainerTask):
    """Beamform on a catalog of sources using the HybridVisStream data product.

    Attributes
    ----------
    window : float
        Window size in degrees.  For each source, right ascensions corresponding to
        abs(ra - source_ra) <= window are extracted from the hybrid beamformed
        visibility at the declination closest to the sources location.  Default is
        5 degrees.
    ignore_rot : bool
        Ignore the telescope rotation_angle when calculating the baseline distances
        used to beamform in the east-west direction.  Defaults to False.
    """

    window = config.Property(proptype=float, default=5.0)
    ignore_rot = config.Property(proptype=bool, default=False)

    def setup(self, manager, catalog):
        """Define the observer and the catalog of sources.

        Parameters
        ----------
        manager : draco.core.io.TelescopeConvertible
            Observer object holding the geographic location of the telescope.
            Note that if ignore_rot is False and this object has a non-zero
            rotation_angle, then the beamforming will account for the phase
            due to the north-south component of the rotation.

        catalog : draco.core.containers.SourceCatalog
            Beamform on sources in this catalog.
        """
        self.telescope = io.get_telescope(manager)
        self.latitude = np.radians(self.telescope.latitude)
        if not self.ignore_rot and hasattr(self.telescope, "rotation_angle"):
            self.log.info(
                "Correcting for phase due to north-south component of a "
                f"{self.telescope.rotation_angle:0.2f} degree rotation."
            )
            self.rot = np.radians(self.telescope.rotation_angle)
        else:
            self.rot = 0.0

        self.catalog = catalog

    def process(self, hvis):
        """Finish beamforming in the east-west direction.

        Parameters
        ----------
        hvis : draco.core.containers.HybridVisStream
            Visibilities beamformed in the north-south direction to
            a grid of declinations along the meridian.

        Returns
        -------
        out : draco.core.containers.HybridFormedBeamHA
            Visibilities beamformed to the location of sources
            in a catalog.
        """
        hvis.redistribute("freq")

        fringestopped = hvis.attrs.get("fringestopped", False)

        # Get source positions
        lsd = hvis.attrs.get("lsd", hvis.attrs.get("csd"))

        src_ra, src_dec = (
            self.catalog["position"]["ra"][:],
            self.catalog["position"]["dec"][:],
        )
        if lsd is not None:
            epoch = np.atleast_1d(self.telescope.lsd_to_unix(lsd))
            coords = [icrs_to_cirs(src_ra, src_dec, ep) for ep in epoch]
            src_ra = np.mean([coord[0] for coord in coords], axis=0)
            src_dec = np.mean([coord[1] for coord in coords], axis=0)

        # Find nearest declination
        dec = np.degrees(np.arcsin(hvis.index_map["el"]) + self.latitude)
        nearest_dec = _search_nearest(dec, src_dec)

        delta_dec = np.max(np.abs(np.diff(dec)))
        valid_src = np.abs(src_dec - dec[nearest_dec]) < delta_dec

        self.log.info(
            f"There are {np.sum(valid_src)} catalog sources in this declination range."
        )

        # Find hour angle window
        ra = hvis.ra
        ha_arr = correct_phase_wrap(ra[np.newaxis, :] - src_ra[:, np.newaxis], deg=True)
        valid = np.abs(ha_arr) <= self.window

        nha = np.sum(valid, axis=-1)

        ra_rad = np.radians(ra)

        # Calculate baseline distances
        freq = hvis.freq[hvis.vis[:].local_bounds]
        lmbda = C * 1e-6 / freq

        ew = hvis.index_map["ew"]
        u = ew[np.newaxis, :, np.newaxis] / lmbda[:, np.newaxis, np.newaxis]
        v = np.sin(self.rot) * u

        # Dereference input datasets
        vis = hvis.vis[:].local_array  # pol, freq, ew, el, ra
        weight = hvis.weight[:].local_array  # pol, freq, ew, ra

        # Create the output container
        out = containers.FormedBeamHAEW(
            object_id=self.catalog.index_map["object_id"],
            ha=np.arange(np.max(nha), dtype=int),
            axes_from=hvis,
            attrs_from=hvis,
            distributed=hvis.distributed,
            comm=hvis.comm,
        )

        if "redshift" in hvis:
            out.add_dataset("redshift")
            out.redshift[:] = hvis.redshift[:]

        out.position["ra"][:] = src_ra
        out.position["dec"][:] = src_dec

        out.redistribute("freq")

        # Dereference output datasets
        ofb = out.beam[:].local_array
        ofb[:] = 0.0

        owe = out.weight[:].local_array
        owe[:] = 0.0

        oha = out.ha[:]
        oha[:] = 0.0

        # Loop over sources and fringestop
        for ss, (idec, sdec) in enumerate(zip(nearest_dec, np.radians(src_dec))):
            in_range = np.flatnonzero(valid[ss])
            if (in_range.size == 0) or not valid_src[ss]:
                continue

            cos_dec = np.cos(np.radians(dec[idec]))

            isort = np.argsort(ha_arr[ss, in_range])
            in_range = in_range[isort]

            islcs = find_contiguous_slices(in_range)
            count = 0
            for islc in islcs:
                svis = vis[..., idec, islc]  # pol, freq, ew, ha
                sweight = weight[..., islc]

                nsample = svis.shape[-1]
                oslc = slice(count, count + nsample)
                count = count + nsample

                oha[ss, oslc] = ha_arr[ss, islc]
                ha = np.radians(ha_arr[ss, islc])

                # Loop over local frequencies and fringestop
                for ff in range(svis.shape[1]):
                    owe[ss, :, ff, :, oslc] = sweight[:, ff]

                    # Calculate the phase
                    phi = interferometry.fringestop_phase(
                        ha, self.latitude, sdec, u[ff], v[ff]
                    )

                    # If the container has already been fringestopped,
                    # then we need to remove that contribution from the phase
                    if fringestopped:
                        omega = 2.0 * np.pi * ew * cos_dec / lmbda[ff]
                        phi *= np.exp(
                            -1.0j * omega[:, np.newaxis] * ra_rad[np.newaxis, islc]
                        )

                    ofb[ss, :, ff, :, oslc] = svis[:, ff] * phi

        return out


class FitBeamFormed(BeamFormExternalMixin, tasklib.base.ContainerTask):
    """Fit beamformed visibilities as a function of hour angle to a primary beam model.

    Must provide a GridBeam object in celestial coordinates as argument to setup.

    Attributes
    ----------
    weight : "uniform" or "inverse_variance"
        How to weight different hour angles during the fit.
    max_ha : float
        Only consider hour angles less than this value in degrees.
        If not provided, then will include all hour angles in the
        input container.
    epsilon : float
        Regularisation used during the fit.
    """

    weight = config.enum(["uniform", "inverse_variance"], default="uniform")
    max_ha = config.Property(proptype=float, default=None)
    min_num_background = config.Property(proptype=int, default=5)
    min_frac_beam = config.Property(proptype=float, default=0.50)
    epsilon = config.Property(proptype=float, default=1.0e-10)

    def process(self, data):
        """Fit a model to the beamformed visibilites along the hour angle axis.

        Parameters
        ----------
        data : FormedBeamHA or FormedBeamHAEW
            Visibilities beamformed to the location of sources in a catalog.

        Returns
        -------
        out : FitFormedBeam or FitFormedBeamEW
            Best-fit parameters describing the hour-angle dependence.
        """
        container_lookup = {
            containers.FormedBeamHA: containers.FitFormedBeam,
            containers.FormedBeamHAEW: containers.FitFormedBeamEW,
        }

        # Distrbitue over frequency
        data.redistribute("freq")

        # Identify local frequencies
        self.freq_local = data.freq[data.beam[:].local_bounds]
        self._initialize_beam_with_data()

        # Create output container
        OutputContainer = container_lookup[data.__class__]

        out = OutputContainer(
            axes_from=data,
            attrs_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )

        out.redistribute("freq")

        # Initialize everything to zero
        for dset in out.datasets.values():
            dset[:] = 0.0

        # Copy over source coordinates
        out.position[:] = data.position[:]
        if "redshift" in data:
            out.add_dataset("redshift")
            out.redshift[:] = data.redshift[:]

        # Dereference datasets
        beam = data.beam[:].local_array
        weight = data.weight[:].local_array

        obeam = out.beam[:].local_array
        oweight = out.weight[:].local_array

        obkg = out.background[:].local_array
        oweightbkg = out.weight_background[:].local_array
        ocorr = out.corr_background_beam[:].local_array

        # Get source coordinates
        src_dec = np.radians(data.position["dec"][:])

        src_ha = data.ha[:]
        max_nha = src_ha.shape[1]

        # Loop over sources
        for ss, sdec in enumerate(src_dec):
            # Ignore missing sources
            if not np.any(weight[ss] > 0.0):
                continue

            # Extract hour angle axis
            nha = max_nha - np.min(np.flatnonzero(src_ha[ss, ::-1] != 0.0))
            slc = slice(0, nha)
            sha = np.radians(src_ha[ss, slc])

            # Loop over polarisation
            for pp, pol in enumerate(data.pol):
                b = beam[ss, pp, ..., slc]
                w = weight[ss, pp, ..., slc]

                if self.weight == "uniform":
                    sigma = np.sqrt(invert_no_zero(w))
                    w = w > 0.0

                if self.max_ha is not None:
                    flag_ha = np.abs(sha) <= np.radians(self.max_ha)
                    w = w * flag_ha
                else:
                    flag_ha = np.ones(nha, dtype=bool)

                # Get the template as a function of hour angle
                X = self.get_template(pol, sdec, sha)
                if "ew" in out.index_map:
                    X = X[:, np.newaxis, :, :]

                # Identify frequencies/baselines that are missing a significant fraction of samples
                f = w > 0
                offsrc = X[..., 1] < 0.05
                flag_background = np.sum(f * offsrc, axis=-1) > self.min_num_background
                flag_beam = (
                    np.sum(f * X[..., 1], axis=-1)
                    * invert_no_zero(np.sum(flag_ha * X[..., 1], axis=-1))
                ) > self.min_frac_beam

                flag = flag_background & flag_beam

                if not np.any(flag):
                    continue

                # Construct the inverse parameter covariance matrix
                XT = np.swapaxes(X, -2, -1)
                A = np.matmul(XT, w[..., np.newaxis] * X) + np.eye(2) * self.epsilon

                # Solve for the parameters and their covariance
                proj_wb = np.sum(
                    XT * (w * b)[..., np.newaxis, :], axis=-1, keepdims=True
                )

                coeff = np.linalg.solve(A, proj_wb)[..., 0]
                cov = np.linalg.solve(A, np.eye(2))

                # Save to output container
                obeam[ss, pp] = coeff[..., 1]
                obkg[ss, pp] = coeff[..., 0]

                if self.weight == "uniform":
                    B = np.matmul(cov, XT * (w * sigma)[..., np.newaxis, :])
                    cov = np.matmul(B, np.swapaxes(B, -2, -1))

                oweight[ss, pp] = flag * invert_no_zero(cov[..., 1, 1])
                oweightbkg[ss, pp] = flag * invert_no_zero(cov[..., 0, 0])

                ocorr[ss, pp] = cov[..., 0, 1] * np.sqrt(
                    oweight[ss, pp] * oweightbkg[ss, pp]
                )

        return out

    def get_template(self, pol, dec, ha):
        """Get the template as a function of hour angle to fit to transit.

        Parameters
        ----------
        pol : str
            String specifying the polarisation,
            either 'XX', 'XY', 'YX', or 'YY'.
        dec : float
            The declination of the source in radians.
        ha : np.ndarray[nha,]
            The hour angle of the source in radians.

        Returns
        -------
        t : np.ndarray[nha, 2]
            Template for the source transit as a function of hour angle.
            First column is entirely 1 and corresponds to an overall additive
            offset. Second column is the primary beam model versus hour angle.
        """
        t = np.ones((self.freq_local.size, ha.size, 2), dtype=float)
        t[..., 1] = self._beamfunc(pol, dec, ha)

        return t


class HealpixBeamForm(tasklib.base.ContainerTask):
    """Beamform by extracting the pixel containing each source form a Healpix map.

    Unless it has an explicit `epoch` attribute, the Healpix map is assumed to be in the
    same coordinate epoch as the catalog. If it does, the input catalog is assumed to be
    in ICRS and then is precessed to the CIRS coordinates in the epoch of the map.

    Attributes
    ----------
    fwhm : float
        Smooth the map with a Gaussian with the specified FWHM in degrees. If `None`
        (default), leave at native map resolution. This will modify the input map in
        place.
    """

    fwhm = config.Property(proptype=float, default=None)

    def setup(self, hpmap: containers.Map):
        """Set the map to extract beams from at each catalog location.

        Parameters
        ----------
        hpmap
            The Healpix map to extract the sources from.
        """
        self.map = hpmap
        mv = self.map.map[:]
        self.map.redistribute("freq")

        self.log.info("Smoothing input Healpix map.")
        for lfi, _ in mv.enumerate(axis=0):
            for pi in range(mv.shape[1]):
                mv[lfi, pi] = healpy.smoothing(
                    mv[lfi, pi], fwhm=np.radians(self.fwhm), verbose=False
                )

    def process(self, catalog: containers.SourceCatalog) -> containers.FormedBeam:
        """Extract sources from a ringmap.

        Parameters
        ----------
        catalog
            The catalog to extract sources from.

        Returns
        -------
        formed_beam
            The source spectra.
        """
        if "position" not in catalog:
            raise ValueError("Input is missing a position table.")

        # Container to hold the formed beams
        formed_beam = containers.FormedBeam(
            object_id=catalog.index_map["object_id"],
            axes_from=self.map,
            distributed=True,
        )

        # Initialize container to zeros.
        formed_beam.beam[:] = 0.0
        formed_beam.weight[:] = 0.0

        # Copy catalog information
        formed_beam["position"][:] = catalog["position"][:]
        if "redshift" in catalog:
            formed_beam.add_dataset("redshift")
            formed_beam["redshift"][:] = catalog["redshift"][:]

        # Get the source positions at the epoch of the input map
        epoch = self.map.attrs.get("epoch", None)
        epoch = ctime.ensure_unix(epoch) if epoch is not None else None
        if epoch:
            src_ra, src_dec = icrs_to_cirs(
                catalog["position"]["ra"], catalog["position"]["dec"], epoch
            )
        else:
            self.log.info(
                "Input map has no epoch set, assuming that it matches the catalog."
            )
            src_ra = catalog["position"]["ra"]
            src_dec = catalog["position"]["dec"]

        # Use Healpix to get the pixels containing the sources
        pix_ind = healpy.ang2pix(self.map.nside, src_ra, src_dec, lonlat=True)

        # Ensure containers are distributed in frequency
        formed_beam.redistribute("freq")
        self.map.redistribute("freq")

        formed_beam.beam[:] = self.map.map[:, :, pix_ind].transpose(2, 1, 0)
        # Set to some non-zero value as the Map container doesn't have a weight
        formed_beam.weight[:] = 1.0

        return formed_beam


def icrs_to_cirs(ra, dec, epoch, apparent=True):
    """Convert a set of positions from ICRS to CIRS at a given data.

    Parameters
    ----------
    ra, dec : float or np.ndarray
        Positions of source in ICRS coordinates including an optional
        redshift position.
    epoch : time_like
        Time to convert the positions to. Can be any type convertible to a
        time using `caput.astro.time.ensure_unix`.
    apparent : bool
        Calculate the apparent position (includes abberation and deflection).

    Returns
    -------
    ra_cirs, dec_cirs : float or np.ndarray
        Arrays of the positions in *CIRS* coordiantes.
    """
    positions = Star(ra=Angle(degrees=ra), dec=Angle(degrees=dec))

    epoch = skf.unix_to_skyfield_time(ctime.ensure_unix(epoch))

    earth = skf.skyfield_wrapper.ephemeris["earth"]
    positions = earth.at(epoch).observe(positions)

    if apparent:
        positions = positions.apparent()

    ra_cirs, dec_cirs, _ = positions.cirs_radec(epoch)

    return ra_cirs._degrees, dec_cirs._degrees

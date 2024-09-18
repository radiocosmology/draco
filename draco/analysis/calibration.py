import numpy as np
import scipy.constants
from mpi4py import MPI

from caput import interferometry, mpiutil, config

from draco.core import task, io
from draco.util import tools


class EigenCalibration(task.SingleTask):
    """Deteremine response of each feed to a point source.

    Extract the feed response from an eigendecomposition of the
    N2 visibility matrix.  Flag frequencies that have low dynamic
    range, orthogonalize the polarizations, fringestop, and reference
    the phases appropriately.

    Attributes
    ----------
    source : str
        Name of the source (same format as `ephemeris.source_dictionary`).
    eigen_ref : int
        Index of the feed that is current phase reference of the eigenvectors.
    phase_ref : list
        Two element list that indicates the chan_id of the feeds to use
        as phase reference for the [Y, X] polarisation.
    med_phase_ref : bool
        Overides `phase_ref`, instead referencing the phase with respect
        to the median value over feeds of a given polarisation.
    neigen : int
        Number of eigenvalues to include in the orthogonalization.
    max_hour_angle : float
        The maximum hour angle in degrees to consider in the analysis.
        Hour angles between [window * max_hour_angle, max_hour_angle] will
        be used for the determination of the off source eigenvalue.
    window : float
        Fraction of the maximum hour angle considered still on source.
    dyn_rng_threshold : float
        Ratio of the second largest eigenvalue on source to the largest eigenvalue
        off source below which frequencies and times will be considered contaminated
        and discarded from further analysis.
    """

    source = config.Property(default=None)
    eigen_ref = config.Property(proptype=int, default=0)
    phase_ref = config.Property(proptype=list, default=[1152, 1408])
    med_phase_ref = config.Property(proptype=bool, default=False)
    neigen = config.Property(proptype=int, default=2)
    max_hour_angle = config.Property(proptype=float, default=10.0)
    window = config.Property(proptype=float, default=0.75)
    dyn_rng_threshold = config.Property(proptype=float, default=3.0)
        
    def setup(self, manager: io.TelescopeConvertible):
        """Set the telescope instance.

        Parameters
        ----------
        manager : manager.ProductManager, optional
            The telescope/manager used to determine
            feed polarisation/position and perform
            ephemeris calculations.
        """
        self.telescope = io.get_telescope(manager)

    def process(self, data: containers.TODContainer) -> containers.SiderealContainer:
        """Determine feed response from eigendecomposition.

        Parameters
        ----------
        data : containers.TODContainer
            Must contain vis, weight, erms, evec, and eval datasets.

        Returns
        -------
        response : containers.SiderealStream
            Response of each feed to the point source.
        """
        # Ensure that we are distributed over frequency
        data.redistribute("freq")

        # Determine local dimensions
        nfreq, neigen, ninput, ntime = data.datasets["evec"].local_shape

        # Find the local frequencies
        freq = data.freq[data.vis[:].local_bounds]

        # Determine source name.  If not provided as config property, then check data attributes.
        source_name = self.source or data.attrs.get("source_name", None)
        if source_name is None:
            raise ValueError(
                "The source name must be specified as a configuration property "
                "or added to input container attributes by an earlier task."
            )

        # Compute flux of source
        source_obj = fluxcat.FluxCatalog[source_name]
        inv_rt_flux_density = tools.invert_no_zero(
            np.sqrt(source_obj.predict_flux(freq))
        )

        # Determine source coordinates
        ttrans = self.telescope.transit_times(source_obj.skyfield, data.time[0])[0]
        lsd = int(np.floor(self.telescope.unix_to_lsd(ttrans)))

        src_ra, src_dec = self.telescope.object_coords(
            source_obj.skyfield, date=ttrans, deg=True
        )

        ra = self.telescope.unix_to_lsa(data.time)

        ha = ra - src_ra
        ha = ((ha + 180.0) % 360.0) - 180.0
        ha = np.radians(ha)

        max_ha_off_source = np.minimum(
            np.max(np.abs(ha)), np.radians(self.max_hour_angle)
        )
        min_ha_off_source = self.window * max_ha_off_source
        off_source = (np.abs(ha) >= min_ha_off_source) & (
            np.abs(ha) <= max_ha_off_source
        )

        itrans = np.argmin(np.abs(ha))

        src_dec = np.radians(src_dec)
        lat = np.radians(self.telescope.latitude)

        # Dereference datasets
        evec = data.datasets["evec"][:].local_array
        evalue = data.datasets["eval"][:].local_array
        erms = data.datasets["erms"][:].local_array
        vis = data.datasets["vis"][:].local_array
        weight = data.flags["vis_weight"][:].local_array

        # Check for negative autocorrelations (bug observed in older data)
        negative_auto = vis.real < 0.0
        if np.any(negative_auto):
            vis[negative_auto] = 0.0 + 0.0j
            weight[negative_auto] = 0.0

        # Find inputs that were not included in the eigenvalue decomposition
        eps = 10.0 * np.finfo(evec.dtype).eps
        evec_all_zero = np.all(np.abs(evec[:, 0]) < eps, axis=(0, 2))

        input_flags = np.zeros(ninput, dtype=bool)
        for ii in range(ninput):
            input_flags[ii] = np.logical_not(
                mpiutil.allreduce(evec_all_zero[ii], op=MPI.LAND, comm=data.comm)
            )

        self.log.info(
            "%d inputs missing from eigenvalue decomposition." % np.sum(~input_flags)
        )

        # Check that we have data for the phase reference
        for ref in self.phase_ref:
            if not input_flags[ref]:
                ValueError(
                    "Requested phase reference (%d) "
                    "was not included in decomposition." % ref
                )

        # Determine index of the x and y pol feeds        
        xfeeds = np.flatnonzero((self.telescope.polarisation == "X") & input_flags)
        xfeeds = np.flatnonzero((self.telescope.polarisation == "Y") & input_flags)

        nfeed = xfeeds.size + yfeeds.size

        pol = [yfeeds, xfeeds]
        polstr = ["Y", "X"]
        npol = len(pol)

        # Determine the phase reference for each polarisation
        phase_ref_by_pol = [
            pol[pp].tolist().index(self.phase_ref[pp]) for pp in range(npol)
        ]

        # Create new product map for the output container that has `input_b` set to
        # the phase reference feed.  Necessary to apply the timing correction later.
        prod = data.prod.copy()
        for pp, feeds in enumerate(pol):
            prod["input_b"][feeds] = self.phase_ref[pp]

        # Compute distances
        dist = self.telescope.feedpositions.copy()
        for pp, feeds in enumerate(pol):
            dist_ref = dist[self.phase_ref[pp], :]
            dist[feeds, :] = dist[feeds, :] - dist_ref

        # Check for feeds that do not have a valid distance (feedpos are set to nan)
        no_distance = np.flatnonzero(np.any(np.isnan(dist), axis=1))
        if (no_distance.size > 0) and np.any(input_flags[no_distance]):
            raise RuntimeError(
                "Do not have positions for feeds: %s"
                % str(no_distance[input_flags[no_distance]])
            )

        # Determine the number of eigenvalues to include in the orthogonalization
        neigen = min(max(npol, self.neigen), neigen)

        # Calculate dynamic range
        eval0_off_source = np.median(evalue[:, 0, off_source], axis=-1)

        dyn = evalue[:, 1, :] * tools.invert_no_zero(eval0_off_source[:, np.newaxis])

        # Determine frequencies and times to mask
        not_rfi = np.any(weight > 0, axis=(1, 2))[:, np.newaxis]

        self.log.info(
            "Using a dynamic range threshold of %0.2f." % self.dyn_rng_threshold
        )
        dyn_flag = dyn > self.dyn_rng_threshold

        converged = erms > 0.0

        flag = converged & dyn_flag & not_rfi

        # Calculate base error
        base_err = erms[:, np.newaxis, :]

        # Check for sign flips
        ref_resp = evec[:, 0:neigen, self.eigen_ref, :]

        sign0 = 1.0 - 2.0 * (ref_resp.real < 0.0)

        # Check that we have the correct reference feed
        if np.any(np.abs(ref_resp.imag) > eps):
            ValueError("Reference feed %d is incorrect." % self.eigen_ref)

        # Create output container
        response = containers.SiderealStream(
            ra=ra,
            prod=prod,
            stack=None,
            attrs_from=data,
            axes_from=data,
            distributed=data.distributed,
            comm=data.comm,
        )

        response.input_flags[:] = input_flags[:, np.newaxis]

        # Create attributes identifying the transit
        response.attrs["source_name"] = source_name
        response.attrs["transit_time"] = ttrans
        response.attrs["lsd"] = lsd
        response.attrs["tag"] = "%s_lsd_%d" % (source_name.lower(), lsd)

        # Add an attribute that indicates if the transit occured during the daytime
        is_daytime = 0
        sun = self.telescope.skyfield.ephemeris["sun"]
        solar_rise = self.telescope.rise_times(sun, ttrans - 86400.0, diameter=0.6)
        for sr in solar_rise:
            ss = self.telescope.set_times(sun, sr, diameter=0.6)[0]
            if (ttrans >= sr) and (ttrans <= ss):
                is_daytime = 1
                break
        response.attrs["daytime_transit"] = is_daytime

        # Dereference the output datasets
        out_vis = response.vis[:]
        out_weight = response.weight[:]

        # Loop over polarizations
        for pp, feeds in enumerate(pol):
            # Create the polarization masking vector
            P = np.zeros((1, ninput, 1), dtype=np.float64)
            P[:, feeds, :] = 1.0

            # Loop over frequencies
            for ff in range(nfreq):
                ww = weight[ff, feeds, :]

                # Normalize by eigenvalue and correct for pi phase flips in process.
                resp = (
                    sign0[ff, :, np.newaxis, :]
                    * evec[ff, 0:neigen, :, :]
                    * np.sqrt(evalue[ff, 0:neigen, np.newaxis, :])
                )

                # Rotate to single-pol response
                # Move time to first axis for the matrix multiplication
                invL = tools.invert_no_zero(
                    np.rollaxis(evalue[ff, 0:neigen, np.newaxis, :], -1, 0)
                )
                UT = np.rollaxis(resp, -1, 0)
                U = np.swapaxes(UT, -1, -2)

                mu, vp = np.linalg.eigh(np.matmul(UT.conj(), P * U))

                rsign0 = 1.0 - 2.0 * (vp[:, 0, np.newaxis, :].real < 0.0)

                resp = mu[:, np.newaxis, :] * np.matmul(U, rsign0 * vp * invL)

                # Extract feeds of this pol
                # Transpose so that time is back to last axis
                resp = resp[:, feeds, -1].T

                # Compute error on response
                dataflg = (
                    flag[ff, np.newaxis, :]
                    & (np.abs(resp) > 0.0)
                    & (ww > 0.0)
                    & np.isfinite(ww)
                ).astype(np.float32)

                resp_err = (
                    dataflg
                    * base_err[ff, :, :]
                    * np.sqrt(vis[ff, feeds, :].real)
                    * tools.invert_no_zero(np.sqrt(mu[np.newaxis, :, -1]))
                )

                # Reference to specific input
                resp *= np.exp(
                    -1.0j * np.angle(resp[phase_ref_by_pol[pp], np.newaxis, :])
                )

                # Fringestop
                lmbda = scipy.constants.c * 1e-6 / freq[ff]

                resp *= interferometry.fringestop_phase(
                    ha[np.newaxis, :],
                    lat,
                    src_dec,
                    dist[feeds, 0, np.newaxis] / lmbda,
                    dist[feeds, 1, np.newaxis] / lmbda,
                )

                # Normalize by source flux
                resp *= inv_rt_flux_density[ff]
                resp_err *= inv_rt_flux_density[ff]

                # If requested, reference phase to the median value
                if self.med_phase_ref:
                    phi0 = np.angle(resp[:, itrans, np.newaxis])
                    resp *= np.exp(-1.0j * phi0)
                    resp *= np.exp(
                        -1.0j * np.median(np.angle(resp), axis=0, keepdims=True)
                    )
                    resp *= np.exp(1.0j * phi0)

                out_vis[ff, feeds, :] = resp
                out_weight[ff, feeds, :] = tools.invert_no_zero(resp_err**2)

        return response
    

class DetermineSourceTransit(task.SingleTask):
    """Determine the sources that are transiting within time range covered by container.

    Attributes
    ----------
    source_list : list of str
        List of source names to consider.  If not specified, all sources
        contained in `ch_ephem.sources.source_dictionary` will be considered.
    freq : float
        Frequency in MHz.  Sort the sources by the flux at this frequency.
    require_transit: bool
        If this is True and a source transit is not found in the container,
        then the task will return None.
    """

    source_list = config.Property(proptype=list, default=[])
    freq = config.Property(proptype=float, default=None) # No assumption about which telescope
    require_transit = config.Property(proptype=bool, default=True)

    def setup(self, fluxcatalog):
        """Set list of sources, sorted by flux in descending order.
        
        Parameters
        ----------
        fluxcatalog : FluxCatalog object 
            TODO: move this object out of ch_util.fluxcat
        """
        self.source_list = sorted(
            self.source_list, # This must be set
            key=lambda src: fluxcatalog[src].predict_flux(self.freq),
            reverse=True,
        )

    def process(self, sstream, observer, source_dictionary):
        """Add attributes to container describing source transit contained within.

        Parameters
        ----------
        sstream : containers.SiderealStream, containers.TimeStream, or equivalent
            Container covering the source transit.

        observer : caput.time.Observer object representing a local observer in
            terms of coordinates, etc.

        source_dictionary : A dictionary whose keys are source names and
            values are `skyfield.starlib.Star` objects.

        Returns
        -------
        sstream : containers.SiderealStream, containers.TimeStream, or equivalent
            Container covering the source transit, now with `source_name` and
            `transit_time` attributes.
        """
        # Determine the time covered by input container
        if "time" in sstream.index_map:
            timestamp = sstream.time
        else:
            lsd = sstream.attrs.get("lsd", sstream.attrs.get("csd"))
            timestamp = observer.lsd_to_unix(lsd + sstream.ra / 360.0)

        # Loop over sources and check if there is a transit within time range
        # covered by container.  If so, then add attributes describing that source
        # and break from the loop.
        contains_transit = False
        for src in self.source_list:
            transit_time = observer.transit_times(
                source_dictionary[src], timestamp[0], timestamp[-1]
            )
            if transit_time.size > 0:
                self.log.info(
                    "Data stream contains %s transit on LSD %d."
                    % (src, observer.unix_to_lsd(transit_time[0]))
                )
                sstream.attrs["source_name"] = src
                sstream.attrs["transit_time"] = transit_time[0]
                contains_transit = True
                break

        if contains_transit or not self.require_transit:
            return sstream

        return None
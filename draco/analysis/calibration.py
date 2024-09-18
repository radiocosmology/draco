import numpy as np
import scipy.constants
from mpi4py import MPI

from caput import interferometry, mpiutil, config, mpiarray

from draco.core import task, io, containers
from draco.util import tools

import json


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
    """
    TODO: move generalized FluxCatalog object out of ch_util.fluxcat and import
    
    Determine the sources that are transiting within time range covered by container.

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
        """Set list of sources, sorted by flux in descending order."""
        self.source_list = sorted(
            self.source_list, # This must be set
            key=lambda src: fluxcat.FluxCatalog[src].predict_flux(self.freq),
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
    

class TransitFit(task.SingleTask):
    """
    TODO: Check defaults
    
    Fit model to the transit of a point source.

    Multiple model choices are available and can be specified through the `model`
    config property.  Default is `gauss_amp_poly_phase`, a nonlinear fit
    of a gaussian in amplitude and a polynomial in phase to the complex data.
    There is also `poly_log_amp_poly_phase`, an iterative weighted least squares
    fit of a polynomial to log amplitude and phase.  The type of polynomial can be
    chosen through the `poly_type`, `poly_deg_amp`, and `poly_deg_phi` config properties.

    Attributes
    ----------
    model : str
        Name of the model to fit.  One of 'gauss_amp_poly_phase' or
        'poly_log_amp_poly_phase'.
    nsigma : float
        Number of standard deviations away from transit to fit.
    absolute_sigma : bool
        Set to True if the errors provided are absolute.  Set to False if
        the errors provided are relative, in which case the parameter covariance
        will be scaled by the chi-squared per degree-of-freedom.
    poly_type : str
        Type of polynomial.  Either 'standard', 'hermite', or 'chebychev'.
        Relevant if `poly = True`.
    poly_deg_amp : int
        Degree of the polynomial to fit to amplitude.
        Relevant if `poly = True`.
    poly_deg_phi : int
        Degree of the polynomial to fit to phase.
        Relevant if `poly = True`.
    niter : int
        Number of times to update the errors using model amplitude.
        Relevant if `poly = True`.
    moving_window : int
        Number of standard deviations away from peak to fit.
        The peak location is updated with each iteration.
        Must be less than `nsigma`.  Relevant if `poly = True`.
    """

    model = config.enum(
        ["gauss_amp_poly_phase", "poly_log_amp_poly_phase"],
        default="gauss_amp_poly_phase",
    )
    nsigma = config.Property(
        proptype=(lambda x: x if x is None else float(x)), default=0.60
    )
    absolute_sigma = config.Property(proptype=bool, default=False)
    poly_type = config.Property(proptype=str, default="standard")
    poly_deg_amp = config.Property(proptype=int, default=5)
    poly_deg_phi = config.Property(proptype=int, default=5)
    niter = config.Property(proptype=int, default=5)
    moving_window = config.Property(
        proptype=(lambda x: x if x is None else float(x)), default=0.30
    )

    def setup(self):
        """Define model to fit to transit."""
        self.fit_kwargs = {"absolute_sigma": self.absolute_sigma}

        if self.model == "gauss_amp_poly_phase":
            self.ModelClass = cal_utils.FitGaussAmpPolyPhase
            self.model_kwargs = {
                "poly_type": self.poly_type,
                "poly_deg_phi": self.poly_deg_phi,
            }

        elif self.model == "poly_log_amp_poly_phase":
            self.ModelClass = cal_utils.FitPolyLogAmpPolyPhase
            self.model_kwargs = {
                "poly_type": self.poly_type,
                "poly_deg_amp": self.poly_deg_amp,
                "poly_deg_phi": self.poly_deg_phi,
            }
            self.fit_kwargs.update(
                {"niter": self.niter, "moving_window": self.moving_window}
            )

        else:
            raise ValueError(
                f"Do not recognize model {self.model}.  Options are "
                "`gauss_amp_poly_phase` and `poly_log_amp_poly_phase`."
            )

    def process(self, response, inputmap):
        """
        TODO: Move generalized ccontainers.TransitFitParams out of 
        ch_pipeline.core.containers and import it


        Fit model to the point source response for each feed and frequency.

        Parameters
        ----------
        response : containers.SiderealStream
            SiderealStream covering the source transit.  Must contain
            `source_name` and `transit_time` attributes.
        inputmap : list of CorrInput's
            List describing the inputs as ordered in response.

        Returns
        -------
        fit : ccontainers.TransitFitParams
            Parameters of the model fit and their covariance.
        """
        # Ensure that we are distributed over frequency
        response.redistribute("freq")

        # Determine local dimensions
        nfreq, ninput, nra = response.vis.local_shape

        # Find the local frequencies
        freq = response.freq[response.vis[:].local_bounds]

        # Calculate the hour angle using the source and transit time saved to attributes
        source_obj = sources.source_dictionary[response.attrs["source_name"]]
        ttrans = response.attrs["transit_time"]

        src_ra, src_dec = coord.object_coords(source_obj, date=ttrans, deg=True)

        ha = response.ra[:] - src_ra
        ha = ((ha + 180.0) % 360.0) - 180.0

        # Determine the fit window
        input_flags = np.any(response.input_flags[:], axis=-1)

        xfeeds = np.array(
            [
                idf
                for idf, inp in enumerate(inputmap)
                if input_flags[idf] and tools.is_array_x(inp)
            ]
        )
        yfeeds = np.array(
            [
                idf
                for idf, inp in enumerate(inputmap)
                if input_flags[idf] and tools.is_array_y(inp)
            ]
        )

        pol = {"X": xfeeds, "Y": yfeeds}

        sigma = np.zeros((nfreq, ninput), dtype=np.float32)
        for pstr, feed in pol.items():
            sigma[:, feed] = cal_utils.guess_fwhm(
                freq, pol=pstr, dec=np.radians(src_dec), sigma=True, voltage=True
            )[:, np.newaxis]

        # Dereference datasets
        vis = response.vis[:].local_array
        weight = response.weight[:].local_array
        err = np.sqrt(tools.invert_no_zero(weight))

        # Flag data that is outside the fit window set by nsigma config parameter
        if self.nsigma is not None:
            err *= (
                np.abs(ha[np.newaxis, np.newaxis, :])
                <= (self.nsigma * sigma[:, :, np.newaxis])
            ).astype(err.dtype)

        # Instantiate the model fitter
        model = self.ModelClass(**self.model_kwargs)

        # Fit the model
        model.fit(ha, vis, err, width=sigma, **self.fit_kwargs)

        # Create an output container
        fit = ccontainers.TransitFitParams(
            param=model.parameter_names,
            component=model.component,
            axes_from=response,
            attrs_from=response,
            distributed=response.distributed,
            comm=response.comm,
        )

        fit.add_dataset("chisq")
        fit.add_dataset("ndof")

        # Transfer fit information to container attributes
        fit.attrs["model_kwargs"] = json.dumps(model.model_kwargs)
        fit.attrs["model_class"] = ".".join(
            [getattr(self.ModelClass, key) for key in ["__module__", "__name__"]]
        )

        # Save datasets
        fit.parameter[:] = model.param[:]
        fit.parameter_cov[:] = model.param_cov[:]
        fit.chisq[:] = model.chisq[:]
        fit.ndof[:] = model.ndof[:]

        return fit
    

class GainFromTransitFit(task.SingleTask):
    """Determine gain by evaluating the best-fit model for the point source transit.

    Attributes
    ----------
    evaluate : str
        Evaluate the model at this location, either 'transit' or 'peak'.
    chisq_per_dof_threshold : float
        Set gain and weight to zero if the chisq per degree of freedom
        of the fit is less than this threshold.
    alpha : float
        Use confidence level 1 - alpha for the uncertainty on the gain.
    """

    evaluate = config.enum(["transit", "peak"], default="transit")
    chisq_per_dof_threshold = config.Property(proptype=float, default=20.0)
    alpha = config.Property(proptype=float, default=0.32)

    def process(self, fit):
        """
        TODO: Modify docstring when we know where TransitFitParams will live
        
        Determine gain from best-fit model.

        Parameters
        ----------
        fit : ccontainers.TransitFitParams
            Parameters of the model fit and their covariance.
            Must also contain 'model_class' and 'model_kwargs'
            attributes that can be used to evaluate the model.

        Returns
        -------
        gain : containers.StaticGainData
            Gain and uncertainty on the gain.
        """
        from pydoc import locate

        # Distribute over frequency
        fit.redistribute("freq")

        nfreq, ninput, _ = fit.parameter.local_shape

        # Import the function for evaluating the model and keyword arguments
        ModelClass = locate(fit.attrs["model_class"])
        model_kwargs = json.loads(fit.attrs["model_kwargs"])

        # Create output container
        out = containers.StaticGainData(
            axes_from=fit, attrs_from=fit, distributed=fit.distributed, comm=fit.comm
        )
        out.add_dataset("weight")

        # Dereference datasets
        param = fit.parameter[:].local_array
        param_cov = fit.parameter_cov[:].local_array
        chisq = fit.chisq[:].local_array
        ndof = fit.ndof[:].local_array

        chisq_per_dof = chisq * tools.invert_no_zero(ndof.astype(np.float32))

        gain = out.gain[:]
        weight = out.weight[:]

        # Instantiate the model object
        model = ModelClass(
            param=param, param_cov=param_cov, chisq=chisq, ndof=ndof, **model_kwargs
        )

        # Suppress numpy floating errors
        with np.errstate(all="ignore"):
            # Determine hour angle of evaluation
            if self.evaluate == "peak":
                ha = model.peak()
                elementwise = True
            else:
                ha = 0.0
                elementwise = False

            # Predict model and uncertainty at desired hour angle
            g = model.predict(ha, elementwise=elementwise)

            gerr = model.uncertainty(ha, alpha=self.alpha, elementwise=elementwise)

            # Use convention that you multiply by gain to calibrate
            gain[:] = tools.invert_no_zero(g)
            weight[:] = tools.invert_no_zero(np.abs(gerr) ** 2) * np.abs(g) ** 4

            # Can occassionally get Infs when evaluating fits to anomalous data.
            # Replace with zeros. Also zero data where the chi-squared per
            # degree of freedom is greater than threshold.
            not_valid = ~(
                np.isfinite(gain)
                & np.isfinite(weight)
                & np.all(chisq_per_dof <= self.chisq_per_dof_threshold, axis=-1)
            )

            if np.any(not_valid):
                gain[not_valid] = 0.0 + 0.0j
                weight[not_valid] = 0.0

        return out
    

class FlagAmplitude(task.SingleTask):
    """Flag feeds and frequencies with outlier gain amplitude.

    Attributes
    ----------
    min_amp_scale_factor : float
        Flag feeds and frequencies where the amplitude of the gain
        is less than `min_amp_scale_factor` times the median amplitude
        over all feeds and frequencies.
    max_amp_scale_factor : float
        Flag feeds and frequencies where the amplitude of the gain
        is greater than `max_amp_scale_factor` times the median amplitude
        over all feeds and frequencies.
    nsigma_outlier : float
        Flag a feed at a particular frequency if the gain amplitude
        is greater than `nsigma_outlier` from the median value over
        all feeds of the same polarisation at that frequency.
    nsigma_med_outlier : float
        Flag a frequency if the median gain amplitude over all feeds of a
        given polarisation is `nsigma_med_outlier` away from the local median.
    window_med_outlier : int
        Number of frequency bins to use to determine the local median for
        the test outlined in the description of `nsigma_med_outlier`.
    threshold_good_freq: float
        If a frequency has less than this fraction of good inputs, then
        it is considered bad and the data for all inputs is flagged.
    threshold_good_input : float
        If an input has less than this fraction of good frequencies, then
        it is considered bad and the data for all frequencies is flagged.
        Note that the fraction is relative to the number of frequencies
        that pass the test described in `threshold_good_freq`.
    valid_gains_frac_good_freq : float
        If the fraction of frequencies that remain after flagging is less than
        this value, then the task will return None and the processing of the
        sidereal day will not proceed further.
    """

    min_amp_scale_factor = config.Property(proptype=float, default=0.05)
    max_amp_scale_factor = config.Property(proptype=float, default=20.0)
    nsigma_outlier = config.Property(proptype=float, default=10.0)
    nsigma_med_outlier = config.Property(proptype=float, default=10.0)
    window_med_outlier = config.Property(proptype=int, default=24)
    threshold_good_freq = config.Property(proptype=float, default=0.70)
    threshold_good_input = config.Property(proptype=float, default=0.80)
    valid_gains_frac_good_freq = config.Property(proptype=float, default=0.0)

    def process(self, gain, inputmap):
        """
        TODO: Move cal_utils.estimate_directional_scale and cal_utils.flag_outliers
        to somewhere in draco and import
        
        Set weight to zero for feeds and frequencies with outlier gain amplitude.

        Parameters
        ----------
        gain : containers.StaticGainData
            Gain derived from point source transit.
        inputmap : list of CorrInput's
            List describing the inputs as ordered in gain.

        Returns
        -------
        gain : containers.StaticGainData
            The input gain container with modified weights.
        """
        # Distribute over frequency
        gain.redistribute("freq")

        nfreq, ninput = gain.gain.local_shape

        sfreq = gain.gain.local_offset[0]
        efreq = sfreq + nfreq

        # Dereference datasets
        flag = gain.weight[:].local_array > 0.0
        amp = np.abs(gain.gain[:].local_array)

        # Determine x and y pol index
        xfeeds = np.array(
            [idf for idf, inp in enumerate(inputmap) if tools.is_array_x(inp)]
        )
        yfeeds = np.array(
            [idf for idf, inp in enumerate(inputmap) if tools.is_array_y(inp)]
        )
        pol = [yfeeds, xfeeds]
        polstr = ["Y", "X"]

        # Hard cutoffs on the amplitude
        med_amp = np.median(amp[flag])
        min_amp = med_amp * self.min_amp_scale_factor
        max_amp = med_amp * self.max_amp_scale_factor

        flag &= (amp >= min_amp) & (amp <= max_amp)

        # Flag outliers in amplitude for each frequency
        for pp, feeds in enumerate(pol):
            med_amp_by_pol = np.zeros(nfreq, dtype=np.float32)
            sig_amp_by_pol = np.zeros(nfreq, dtype=np.float32)

            for ff in range(nfreq):
                this_flag = flag[ff, feeds]

                if np.any(this_flag):
                    med, slow, shigh = cal_utils.estimate_directional_scale(
                        amp[ff, feeds[this_flag]]
                    )
                    lower = med - self.nsigma_outlier * slow
                    upper = med + self.nsigma_outlier * shigh

                    flag[ff, feeds] &= (amp[ff, feeds] >= lower) & (
                        amp[ff, feeds] <= upper
                    )

                    med_amp_by_pol[ff] = med
                    sig_amp_by_pol[ff] = (
                        0.5
                        * (shigh - slow)
                        / np.sqrt(np.sum(this_flag, dtype=np.float32))
                    )

            # Flag frequencies that are outliers with respect to local median
            if self.nsigma_med_outlier:
                # Collect med_amp_by_pol for all frequencies on rank 0
                if gain.comm.rank == 0:
                    full_med_amp_by_pol = np.zeros(gain.freq.size, dtype=np.float32)
                else:
                    full_med_amp_by_pol = None

                mpiutil.gather_local(
                    full_med_amp_by_pol,
                    med_amp_by_pol,
                    (sfreq,),
                    root=0,
                    comm=gain.comm,
                )

                # Flag outlier frequencies on rank 0
                not_outlier = None
                if gain.comm.rank == 0:
                    med_flag = full_med_amp_by_pol > 0.0

                    not_outlier = cal_utils.flag_outliers(
                        full_med_amp_by_pol,
                        med_flag,
                        window=self.window_med_outlier,
                        nsigma=self.nsigma_med_outlier,
                    )

                    self.log.info(
                        "Pol %s:  %d frequencies are outliers."
                        % (polstr[pp], np.sum(~not_outlier & med_flag, dtype=np.int64))
                    )

                # Broadcast outlier frequencies to other ranks
                not_outlier = gain.comm.bcast(not_outlier, root=0)
                gain.comm.Barrier()

                flag[:, feeds] &= not_outlier[sfreq:efreq, np.newaxis]

        # Determine bad frequencies
        flag_freq = (
            np.sum(flag, axis=1, dtype=np.float32) / float(ninput)
        ) > self.threshold_good_freq

        good_freq = list(sfreq + np.flatnonzero(flag_freq))
        good_freq = np.array(mpiutil.allreduce(good_freq, op=MPI.SUM, comm=gain.comm))

        flag &= flag_freq[:, np.newaxis]

        self.log.info("%d good frequencies after flagging amplitude." % good_freq.size)

        # If fraction of good frequencies is less than threshold, stop and return None
        frac_good_freq = good_freq.size / float(gain.freq.size)
        if frac_good_freq < self.valid_gains_frac_good_freq:
            self.log.info(
                f"Only {100.0 * frac_good_freq:0.1f}% of frequencies remain after flagging amplitude.  Will "
                "not process this sidereal day further."
            )
            return None

        # Determine bad inputs
        flag = mpiarray.MPIArray.wrap(flag, axis=0, comm=gain.comm)
        flag = flag.redistribute(1)

        fraction_good = np.sum(
            flag[good_freq, :], axis=0, dtype=np.float32
        ) * tools.invert_no_zero(float(good_freq.size))
        flag_input = fraction_good > self.threshold_good_input

        good_input = list(flag.local_offset[1] + np.flatnonzero(flag_input))
        good_input = np.array(mpiutil.allreduce(good_input, op=MPI.SUM, comm=gain.comm))

        flag[:] &= flag_input[np.newaxis, :]

        self.log.info("%d good inputs after flagging amplitude." % good_input.size)

        # Redistribute flags back over frequencies and update container
        flag = flag.redistribute(0)

        gain.weight[:] *= flag.astype(gain.weight.dtype)

        return gain
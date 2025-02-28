"""Some utility functions for calibration"""
from abc import ABCMeta, abstractmethod
import logging

import numpy as np
from scipy.optimize import curve_fit
import scipy.stats

from . import tools

# Set up logging TODO: Do we want to do this every time on import in draco?
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def estimate_directional_scale(z, c=2.1):
    """Calculate robust, direction dependent estimate of scale.

    Parameters
    ----------
    z: np.ndarray
        1D array containing the data.
    c: float
        Cutoff in number of MAD.  Data points whose absolute value is
        larger than c * MAD from the median are saturated at the
        maximum value in the estimator.

    Returns
    -------
    zmed : float
        The median value of z.
    sa : float
        Estimate of scale for z <= zmed.
    sb : float
        Estimate of scale for z > zmed.
    """
    zmed = np.median(z)

    x = z - zmed

    xa = x[x <= 0.0]
    xb = x[x >= 0.0]

    def huber_rho(dx, c=2.1):
        num = float(dx.size)

        s0 = 1.4826 * np.median(np.abs(dx))

        dx_sig0 = dx * tools.invert_no_zero(s0)

        rho = (dx_sig0 / c) ** 2
        rho[rho > 1.0] = 1.0

        return 1.54 * s0 * np.sqrt(2.0 * np.sum(rho) / num)

    sa = huber_rho(xa, c=c)
    sb = huber_rho(xb, c=c)

    return zmed, sa, sb

def _sliding_window(arr, window):
    # Advanced numpy tricks
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def flag_outliers(raw, flag, window=25, nsigma=5.0):
    """Flag outliers with respect to rolling median.

    Parameters
    ----------
    raw : np.ndarray[nsample,]
        Raw data sampled at fixed rate.  Use the `flag` parameter to indicate missing
        or invalid data.
    flag : np.ndarray[nsample,]
        Boolean array where True indicates valid data and False indicates invalid data.
    window : int
        Window size (in number of samples) used to determine local median.
    nsigma : float
        Data is considered an outlier if it is greater than this number of
        median absolute deviations away from the local median.
    Returns
    -------
    not_outlier : np.ndarray[nsample,]
        Boolean array where True indicates valid data and False indicates data that is
        either an outlier or had flag = True.
    """
    # Make sure we have an even window size
    if window % 2:
        window += 1

    hwidth = window // 2 - 1

    nraw = raw.size
    dtype = raw.dtype

    # Replace flagged samples with nan
    good = np.flatnonzero(flag)

    data = np.full((nraw,), np.nan, dtype=dtype)
    data[good] = raw[good]

    # Expand the edges
    expanded_data = np.concatenate(
        (
            np.full((hwidth,), np.nan, dtype=dtype),
            data,
            np.full((hwidth + 1,), np.nan, dtype=dtype),
        )
    )

    # Apply median filter
    smooth = np.nanmedian(_sliding_window(expanded_data, window), axis=-1)

    # Calculate RMS of residual
    resid = np.abs(data - smooth)

    rwidth = 9 * window
    hrwidth = rwidth // 2 - 1

    expanded_resid = np.concatenate(
        (
            np.full((hrwidth,), np.nan, dtype=dtype),
            resid,
            np.full((hrwidth + 1,), np.nan, dtype=dtype),
        )
    )

    sig = 1.4826 * np.nanmedian(_sliding_window(expanded_resid, rwidth), axis=-1)

    return resid < (nsigma * sig)

def _propagate_uncertainty(jac, cov, tval):
    """Propagate uncertainty on parameters to uncertainty on model prediction.

    Parameters
    ----------
    jac : np.ndarray[..., nparam] (elementwise) or np.ndarray[..., nparam, nha]
        The jacobian defined as
        jac[..., i, j] = d(model(ha)) / d(param[i]) evaluated at ha[j]
    cov : [..., nparam, nparam]
        Covariance of model parameters.
    tval : np.ndarray[...]
        Quantile of a standardized Student's t random variable.
        The 1-sigma uncertainties will be scaled by this value.

    Returns
    -------
    err : np.ndarray[...] (elementwise) or np.ndarray[..., nha]
        Uncertainty on the model.
    """
    if jac.ndim == cov.ndim:
        # Corresponds to non-elementwise analysis
        df2 = np.sum(jac * np.matmul(cov, jac), axis=-2)
    else:
        # Corresponds to elementwise analysis
        df2 = np.sum(jac * np.sum(cov * jac[..., np.newaxis], axis=-1), axis=-1)

    # Expand the tval array so that it can be broadcast against
    # the sum squared error df2
    add_dim = df2.ndim - tval.ndim
    if add_dim > 0:
        tval = tval[(np.s_[...],) + (None,) * add_dim]

    return tval * np.sqrt(df2)

def _correct_phase_wrap(ha):
    """Ensure hour angle is between -180 and 180 degrees.

    Parameters
    ----------
    ha : np.ndarray or float
        Hour angle in degrees.

    Returns
    -------
    out : same as ha
        Hour angle between -180 and 180 degrees.
    """
    return ((ha + 180.0) % 360.0) - 180.0


class FitTransit(metaclass=ABCMeta):
    """Base class for fitting models to point source transits.

    The `fit` method should be used to populate the `param`, `param_cov`, `chisq`,
    and `ndof` attributes.  The `predict` and `uncertainty` methods can then be used
    to obtain the model prediction for the response and uncertainty on this quantity
    at a given hour angle.

    Attributes
    ----------
    param : np.ndarray[..., nparam]
        Best-fit parameters.
    param_cov : np.ndarray[..., nparam, nparam]
        Covariance of the fit parameters.
    chisq : np.ndarray[...]
        Chi-squared of the fit.
    ndof : np.ndarray[...]
        Number of degrees of freedom.

    Abstract Methods
    ----------------
    Any subclass of FitTransit must define these methods:
        peak
        _fit
        _model
        _jacobian
    """

    _tval = {}
    component = np.array(["complex"], dtype=np.bytes_)

    def __init__(self, *args, **kwargs):
        """Instantiates a FitTransit object.

        Parameters
        ----------
        param : np.ndarray[..., nparam]
            Best-fit parameters.
        param_cov : np.ndarray[..., nparam, nparam]
            Covariance of the fit parameters.
        chisq : np.ndarray[..., ncomponent]
            Chi-squared.
        ndof : np.ndarray[..., ncomponent]
            Number of degrees of freedom.
        """
        # Save keyword arguments as attributes
        self.param = kwargs.pop("param", None)
        self.param_cov = kwargs.pop("param_cov", None)
        self.chisq = kwargs.pop("chisq", None)
        self.ndof = kwargs.pop("ndof", None)
        self.model_kwargs = kwargs

    def predict(self, ha, elementwise=False):
        """Predict the point source response.

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            The hour angle in degrees.
        elementwise : bool
            If False, then the model will be evaluated at the
            requested hour angles for every set of parameters.
            If True, then the model will be evaluated at a
            separate hour angle for each set of parameters
            (requires `ha.shape == self.N`).

        Returns
        -------
        model : np.ndarray[..., nha] or float
            Model for the point source response at the requested
            hour angles.  Complex valued.
        """
        with np.errstate(all="ignore"):
            mdl = self._model(ha, elementwise=elementwise)
        return np.where(np.isfinite(mdl), mdl, 0.0 + 0.0j)

    def uncertainty(self, ha, alpha=0.32, elementwise=False):
        """Predict the uncertainty on the point source response.

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            The hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.
        elementwise : bool
            If False, then the uncertainty will be evaluated at
            the requested hour angles for every set of parameters.
            If True, then the uncertainty will be evaluated at a
            separate hour angle for each set of parameters
            (requires `ha.shape == self.N`).

        Returns
        -------
        err : np.ndarray[..., nha]
            Uncertainty on the point source response at the
            requested hour angles.
        """
        x = np.atleast_1d(ha)
        with np.errstate(all="ignore"):
            err = _propagate_uncertainty(
                self._jacobian(x, elementwise=elementwise),
                self.param_cov,
                self.tval(alpha, self.ndof),
            )
        return np.squeeze(np.where(np.isfinite(err), err, 0.0))

    def fit(self, ha, resp, resp_err, width=5, absolute_sigma=False, **kwargs):
        """Apply subclass defined `_fit` method to multiple transits.

        This function can be used to fit the transit for multiple inputs
        and frequencies.  Populates the `param`, `param_cov`, `chisq`, and `ndof`
        attributes.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[..., nha]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[..., nha]
            Error on the measured response.
        width : np.ndarray[...]
            Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.
        """
        shp = resp.shape[:-1]
        dtype = ha.dtype

        if not np.isscalar(width) and (width.shape != shp):
            ValueError(f"Keyword with must be scalar or have shape {shp!s}.")

        self.param = np.full(shp + (self.nparam,), np.nan, dtype=dtype)
        self.param_cov = np.full(shp + (self.nparam, self.nparam), np.nan, dtype=dtype)
        self.chisq = np.full(shp + (self.ncomponent,), np.nan, dtype=dtype)
        self.ndof = np.full(shp + (self.ncomponent,), 0, dtype=int)

        with np.errstate(all="ignore"):
            for ind in np.ndindex(*shp):
                wi = width if np.isscalar(width) else width[ind[: width.ndim]]

                err = resp_err[ind]
                good = np.flatnonzero(err > 0.0)

                if (good.size // 2) <= self.nparam:
                    continue

                try:
                    param, param_cov, chisq, ndof = self._fit(
                        ha[good],
                        resp[ind][good],
                        err[good],
                        width=wi,
                        absolute_sigma=absolute_sigma,
                        **kwargs,
                    )
                except (ValueError, KeyError) as error:
                    logger.debug(f"Index {ind!s} failed with error: {error}")
                    continue

                self.param[ind] = param
                self.param_cov[ind] = param_cov
                self.chisq[ind] = chisq
                self.ndof[ind] = ndof

    @property
    def parameter_names(self):
        """
        Array of strings containing the name of the fit parameters.

        Returns
        -------
        parameter_names : np.ndarray[nparam,]
            Names of the parameters.
        """
        return np.array(["param%d" % p for p in range(self.nparam)], dtype=np.bytes_)

    @property
    def param_corr(self):
        """
        Parameter correlation matrix.

        Returns
        -------
        param_corr : np.ndarray[..., nparam, nparam]
            Correlation of the fit parameters.
        """
        idiag = tools.invert_no_zero(
            np.sqrt(np.diagonal(self.param_cov, axis1=-2, axis2=-1))
        )
        return self.param_cov * idiag[..., np.newaxis, :] * idiag[..., np.newaxis]

    @property
    def N(self):
        """
        Number of independent transit fits contained in this object.

        Returns
        -------
        N : tuple
            Numpy-style shape indicating the number of
            fits that the object contains.  Is None
            if the object contains a single fit.
        """
        if self.param is not None:
            return self.param.shape[:-1] or None
        return None

    @property
    def nparam(self):
        """
        Number of parameters.

        Returns
        -------
        nparam :  int
            Number of fit parameters.
        """
        return self.param.shape[-1]

    @property
    def ncomponent(self):
        """
        Number of components.

        Returns
        -------
        ncomponent : int
            Number of components (i.e, real and imag, amp and phase,
            complex) that have been fit.
        """
        return self.component.size

    def __getitem__(self, val):
        """Instantiates a new TransitFit object containing some subset of the fits."""

        if self.N is None:
            raise KeyError(
                "Attempting to slice TransitFit object containing single fit."
            )

        return self.__class__(
            param=self.param[val],
            param_cov=self.param_cov[val],
            ndof=self.ndof[val],
            chisq=self.chisq[val],
            **self.model_kwargs,
        )

    @abstractmethod
    def peak(self):
        """Calculate the peak of the transit.

        Any subclass of FitTransit must define this method.
        """
        return

    @abstractmethod
    def _fit(self, ha, resp, resp_err, width=None, absolute_sigma=False):
        """Fit data to the model.

        Any subclass of FitTransit must define this method.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[nha,]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[nha,]
            Error on the measured response.
        width : np.ndarray
            Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.

        Returns
        -------
        param : np.ndarray[nparam,]
            Best-fit model parameters.
        param_cov : np.ndarray[nparam, nparam]
            Covariance of the best-fit model parameters.
        chisq : float
            Chi-squared of the fit.
        ndof : int
            Number of degrees of freedom of the fit.
        """
        return

    @abstractmethod
    def _model(self, ha):
        """Calculate the model for the point source response.

        Any subclass of FitTransit must define this method.

        Parameters
        ----------
        ha : np.ndarray
            Hour angle in degrees.
        """
        return

    @abstractmethod
    def _jacobian(self, ha):
        """Calculate the jacobian of the model for the point source response.

        Any subclass of FitTransit must define this method.

        Parameters
        ----------
        ha : np.ndarray
            Hour angle in degrees.

        Returns
        -------
        jac : np.ndarray[..., nparam, nha]
            The jacobian defined as
            jac[..., i, j] = d(model(ha)) / d(param[i]) evaluated at ha[j]
        """
        return

    @classmethod
    def tval(cls, alpha, ndof):
        """Quantile of a standardized Student's t random variable.

        This quantity is slow to compute.  Past values will be cached
        in a dictionary shared by all instances of the class.

        Parameters
        ----------
        alpha : float
            Calculate the quantile corresponding to the lower tail probability
            1 - alpha / 2.
        ndof : np.ndarray or int
            Number of degrees of freedom of the Student's t variable.

        Returns
        -------
        tval : np.ndarray or float
            Quantile of a standardized Student's t random variable.
        """
        prob = 1.0 - 0.5 * alpha

        arr_ndof = np.atleast_1d(ndof)
        tval = np.zeros(arr_ndof.shape, dtype=np.float32)

        for ind, nd in np.ndenumerate(arr_ndof):
            key = (int(100.0 * prob), nd)
            if key not in cls._tval:
                cls._tval[key] = scipy.stats.t.ppf(prob, nd)
            tval[ind] = cls._tval[key]

        if np.isscalar(ndof):
            tval = np.squeeze(tval)

        return tval


class FitPoly(FitTransit):
    """Base class for fitting polynomials to point source transits.

    Maps methods of np.polynomial to methods of the class for the
    requested polynomial type.
    """

    def __init__(self, poly_type="standard", *args, **kwargs):
        """Instantiates a FitPoly object.

        Parameters
        ----------
        poly_type : str
            Type of polynomial.  Can be 'standard', 'hermite', or 'chebyshev'.
        """
        super().__init__(poly_type=poly_type, *args, **kwargs)

        self._set_polynomial_model(poly_type)

    def _set_polynomial_model(self, poly_type):
        """Map methods of np.polynomial to methods of the class."""
        if poly_type == "standard":
            self._vander = np.polynomial.polynomial.polyvander
            self._eval = np.polynomial.polynomial.polyval
            self._deriv = np.polynomial.polynomial.polyder
            self._root = np.polynomial.polynomial.polyroots
        elif poly_type == "hermite":
            self._vander = np.polynomial.hermite.hermvander
            self._eval = np.polynomial.hermite.hermval
            self._deriv = np.polynomial.hermite.hermder
            self._root = np.polynomial.hermite.hermroots
        elif poly_type == "chebyshev":
            self._vander = np.polynomial.chebyshev.chebvander
            self._eval = np.polynomial.chebyshev.chebval
            self._deriv = np.polynomial.chebyshev.chebder
            self._root = np.polynomial.chebyshev.chebroots
        else:
            raise ValueError(
                f"Do not recognize polynomial type {poly_type}."
                "Options are 'standard', 'hermite', or 'chebyshev'."
            )

        self.poly_type = poly_type

    def _fast_eval(self, ha, param=None, elementwise=False):
        """Evaluate the polynomial at the requested hour angle."""
        if param is None:
            param = self.param

        vander = self._vander(ha, param.shape[-1] - 1)

        if elementwise:
            out = np.sum(vander * param, axis=-1)
        elif param.ndim == 1:
            out = np.dot(vander, param)
        else:
            out = np.matmul(param, np.rollaxis(vander, -1))

        return np.squeeze(out, axis=-1) if np.isscalar(ha) else out
    

class FitAmpPhase(FitTransit):
    """Base class for fitting models to the amplitude and phase.

    Assumes an independent fit to amplitude and phase, and provides
    methods for predicting the uncertainty on each.
    """

    component = np.array(["amplitude", "phase"], dtype=np.bytes_)

    def uncertainty_amp(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on amplitude at given hour angle(s).

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the amplitude in fractional units.
        """
        x = np.atleast_1d(ha)
        err = _propagate_uncertainty(
            self._jacobian_amp(x, elementwise=elementwise),
            self.param_cov[..., : self.npara, : self.npara],
            self.tval(alpha, self.ndofa),
        )
        return np.squeeze(err, axis=-1) if np.isscalar(ha) else err

    def uncertainty_phi(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on phase at given hour angle(s).

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the phase in radians.
        """
        x = np.atleast_1d(ha)
        err = _propagate_uncertainty(
            self._jacobian_phi(x, elementwise=elementwise),
            self.param_cov[..., self.npara :, self.npara :],
            self.tval(alpha, self.ndofp),
        )
        return np.squeeze(err, axis=-1) if np.isscalar(ha) else err

    def uncertainty(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on the response at given hour angle(s).

        Returns the quadrature sum of the amplitude and phase uncertainty.

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the response.
        """
        with np.errstate(all="ignore"):
            return np.abs(self._model(ha, elementwise=elementwise)) * np.sqrt(
                self.uncertainty_amp(ha, alpha=alpha, elementwise=elementwise) ** 2
                + self.uncertainty_phi(ha, alpha=alpha, elementwise=elementwise) ** 2
            )

    def _jacobian(self, ha):
        raise NotImplementedError(
            "Fits to amplitude and phase are independent.  "
            "Use _jacobian_amp and _jacobian_phi instead."
        )

    @abstractmethod
    def _jacobian_amp(self, ha):
        """Calculate the jacobian of the model for the amplitude."""
        return

    @abstractmethod
    def _jacobian_phi(self, ha):
        """Calculate the jacobian of the model for the phase."""
        return

    @property
    def nparam(self):
        return self.npara + self.nparp


class FitGaussAmpPolyPhase(FitPoly, FitAmpPhase):
    """Class that enables fits of a gaussian to amplitude and a polynomial to phase."""

    component = np.array(["complex"], dtype=np.bytes_)
    npara = 3

    def __init__(self, poly_deg_phi=5, *args, **kwargs):
        """Instantiates a FitGaussAmpPolyPhase object.

        Parameters
        ----------
        poly_deg_phi : int
            Degree of the polynomial to fit to phase.
        """
        super().__init__(poly_deg_phi=poly_deg_phi, *args, **kwargs)

        self.poly_deg_phi = poly_deg_phi
        self.nparp = poly_deg_phi + 1

    def _fit(self, ha, resp, resp_err, width=5, absolute_sigma=False, param0=None):
        """Fit gaussian to amplitude and polynomial to phase.

        Uses non-linear least squares (`scipy.optimize.curve_fit`) to
        fit the model to the complex valued data.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[nha,]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[nha,]
            Error on the measured response.
        width : float
             Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.
        param0 : np.ndarray[nparam,]
            Initial guess at the parameters for the Levenberg-Marquardt algorithm.
            If these are not provided, then this function will make reasonable guesses.

        Returns
        -------
        param : np.ndarray[nparam,]
            Best-fit model parameters.
        param_cov : np.ndarray[nparam, nparam]
            Covariance of the best-fit model parameters.
        chisq : float
            Chi-squared of the fit.
        ndof : int
            Number of degrees of freedom of the fit.
        """
        if ha.size < (min(self.npara, self.nparp) + 1):
            raise RuntimeError("Number of data points less than number of parameters.")

        # We will fit the complex data.  Break n-element complex array y(x)
        # into 2n-element real array [Re{y(x)}, Im{y(x)}] for fit.
        x = np.tile(ha, 2)
        y = np.concatenate((resp.real, resp.imag))
        err = np.tile(resp_err, 2)

        # Initial estimate of parameter values:
        # [peak_amplitude, centroid, fwhm, phi_0, phi_1, phi_2, ...]
        if param0 is None:
            param0 = [np.max(np.nan_to_num(np.abs(resp))), 0.0, 2.355 * width]
            param0.append(np.median(np.nan_to_num(np.angle(resp, deg=True))))
            param0 += [0.0] * (self.nparp - 1)
            param0 = np.array(param0)

        # Perform the fit.
        param, param_cov = curve_fit(
            self._get_fit_func(),
            x,
            y,
            sigma=err,
            p0=param0,
            absolute_sigma=absolute_sigma,
            jac=self._get_fit_jac(),
        )

        chisq = np.sum(
            (
                np.abs(resp - self._model(ha, param=param))
                * tools.invert_no_zero(resp_err)
            )
            ** 2
        )
        ndof = y.size - self.nparam

        return param, param_cov, chisq, ndof

    def peak(self):
        """Return the peak of the transit.

        Returns
        -------
        peak : float
            Centroid of the gaussian fit to amplitude.
        """
        return self.param[..., 1]

    def _get_fit_func(self):
        """Generates a function that can be used by `curve_fit` to compute the model."""

        def fit_func(x, *param):
            """Function used by `curve_fit` to compute the model.

            Parameters
            ----------
            x : np.ndarray[2 * nha,]
                Hour angle in degrees replicated twice for the real
                and imaginary components, i.e., `x = np.concatenate((ha, ha))`.
            *param : floats
                Parameters of the model.

            Returns
            -------
            model : np.ndarray[2 * nha,]
                Model for the complex valued point source response,
                packaged as `np.concatenate((model.real, model.imag))`.
            """
            peak_amplitude, centroid, fwhm = param[:3]
            poly_coeff = param[3:]

            nreal = len(x) // 2
            xr = x[:nreal]

            dxr = _correct_phase_wrap(xr - centroid)

            model_amp = peak_amplitude * np.exp(-4.0 * np.log(2.0) * (dxr / fwhm) ** 2)
            model_phase = self._eval(xr, poly_coeff)

            return np.concatenate(
                (model_amp * np.cos(model_phase), model_amp * np.sin(model_phase))
            )

        return fit_func

    def _get_fit_jac(self):
        """Get `curve_fit` Jacobian

        Generates a function that can be used by `curve_fit` to compute
        Jacobian of the model."""

        def fit_jac(x, *param):
            """Function used by `curve_fit` to compute the jacobian.

            Parameters
            ----------
            x : np.ndarray[2 * nha,]
                Hour angle in degrees.  Replicated twice for the real
                and imaginary components, i.e., `x = np.concatenate((ha, ha))`.
            *param : float
                Parameters of the model.

            Returns
            -------
            jac : np.ndarray[2 * nha, nparam]
                The jacobian defined as
                jac[i, j] = d(model(ha)) / d(param[j]) evaluated at ha[i]
            """

            peak_amplitude, centroid, fwhm = param[:3]
            poly_coeff = param[3:]

            nparam = len(param)
            nx = len(x)
            nreal = nx // 2

            jac = np.empty((nx, nparam), dtype=x.dtype)

            dx = _correct_phase_wrap(x - centroid)

            dxr = dx[:nreal]
            xr = x[:nreal]

            model_amp = peak_amplitude * np.exp(-4.0 * np.log(2.0) * (dxr / fwhm) ** 2)
            model_phase = self._eval(xr, poly_coeff)
            model = np.concatenate(
                (model_amp * np.cos(model_phase), model_amp * np.sin(model_phase))
            )

            dmodel_dphase = np.concatenate((-model[nreal:], model[:nreal]))

            jac[:, 0] = tools.invert_no_zero(peak_amplitude) * model
            jac[:, 1] = 8.0 * np.log(2.0) * dx * tools.invert_no_zero(fwhm) ** 2 * model
            jac[:, 2] = (
                8.0 * np.log(2.0) * dx**2 * tools.invert_no_zero(fwhm) ** 3 * model
            )
            jac[:, 3:] = (
                self._vander(x, self.poly_deg_phi) * dmodel_dphase[:, np.newaxis]
            )

            return jac

        return fit_jac

    def _model(self, ha, param=None, elementwise=False):
        if param is None:
            param = self.param

        # Evaluate phase
        model_phase = self._fast_eval(
            ha, param[..., self.npara :], elementwise=elementwise
        )

        # Evaluate amplitude
        amp_param = param[..., : self.npara]
        ndim1 = amp_param.ndim
        if not elementwise and (ndim1 > 1) and not np.isscalar(ha):
            ndim2 = ha.ndim
            amp_param = amp_param[(slice(None),) * ndim1 + (None,) * ndim2]
            ha = ha[(None,) * (ndim1 - 1) + (slice(None),) * ndim2]

        slc = (slice(None),) * (ndim1 - 1)
        peak_amplitude = amp_param[slc + (0,)]
        centroid = amp_param[slc + (1,)]
        fwhm = amp_param[slc + (2,)]

        dha = _correct_phase_wrap(ha - centroid)

        model_amp = peak_amplitude * np.exp(-4.0 * np.log(2.0) * (dha / fwhm) ** 2)

        # Return complex valued quantity
        return model_amp * (np.cos(model_phase) + 1.0j * np.sin(model_phase))

    def _jacobian_amp(self, ha, elementwise=False):
        amp_param = self.param[..., : self.npara]

        shp = amp_param.shape
        ndim1 = amp_param.ndim

        if not elementwise:
            shp = shp + ha.shape

            if ndim1 > 1:
                ndim2 = ha.ndim
                amp_param = amp_param[(slice(None),) * ndim1 + (None,) * ndim2]
                ha = ha[(None,) * (ndim1 - 1) + (slice(None),) * ndim2]

        slc = (slice(None),) * (ndim1 - 1)
        peak_amplitude = amp_param[slc + (0,)]
        centroid = amp_param[slc + (1,)]
        fwhm = amp_param[slc + (2,)]

        dha = _correct_phase_wrap(ha - centroid)

        jac = np.zeros(shp, dtype=ha.dtype)
        jac[slc + (0,)] = tools.invert_no_zero(peak_amplitude)
        jac[slc + (1,)] = 8.0 * np.log(2.0) * dha * tools.invert_no_zero(fwhm) ** 2
        jac[slc + (2,)] = 8.0 * np.log(2.0) * dha**2 * tools.invert_no_zero(fwhm) ** 3

        return jac

    def _jacobian_phi(self, ha, elementwise=False):
        jac = self._vander(ha, self.poly_deg_phi)
        if not elementwise:
            jac = np.rollaxis(jac, -1)
            if self.N is not None:
                slc = (None,) * len(self.N)
                jac = jac[slc]

        return jac

    @property
    def parameter_names(self):
        """Array of strings containing the name of the fit parameters."""
        return np.array(
            ["peak_amplitude", "centroid", "fwhm"]
            + ["%s_poly_phi_coeff%d" % (self.poly_type, p) for p in range(self.nparp)],
            dtype=np.bytes_,
        )

    @property
    def ndofa(self):
        """
        Number of degrees of freedom for the amplitude fit.

        Returns
        -------
        ndofa : np.ndarray[...]
            Number of degrees of freedom of the amplitude fit.
        """
        return self.ndof[..., 0]

    @property
    def ndofp(self):
        """
        Number of degrees of freedom for the phase fit.

        Returns
        -------
        ndofp : np.ndarray[...]
            Number of degrees of freedom of the phase fit.
        """
        return self.ndof[..., 0]

class FitPolyLogAmpPolyPhase(FitPoly, FitAmpPhase):
    """Class that enables separate fits of a polynomial to log amplitude and phase."""

    def __init__(self, poly_deg_amp=5, poly_deg_phi=5, *args, **kwargs):
        """Instantiates a FitPolyLogAmpPolyPhase object.

        Parameters
        ----------
        poly_deg_amp : int
            Degree of the polynomial to fit to log amplitude.
        poly_deg_phi : int
            Degree of the polynomial to fit to phase.
        """
        super().__init__(
            poly_deg_amp=poly_deg_amp, poly_deg_phi=poly_deg_phi, *args, **kwargs
        )

        self.poly_deg_amp = poly_deg_amp
        self.poly_deg_phi = poly_deg_phi

        self.npara = poly_deg_amp + 1
        self.nparp = poly_deg_phi + 1

    def _fit(
        self,
        ha,
        resp,
        resp_err,
        width=None,
        absolute_sigma=False,
        moving_window=0.3,
        niter=5,
    ):
        """Fit polynomial to log amplitude and polynomial to phase.

        Use weighted least squares.  The initial errors on log amplitude
        are set to `resp_err / abs(resp)`.  If the niter parameter is greater than 1,
        then those errors will be updated with `resp_err / model_amp`, where `model_amp`
        is the best-fit model for the amplitude from the previous iteration.  The errors
        on the phase are set to `resp_err / model_amp` where `model_amp` is the best-fit
        model for the amplitude from the log amplitude fit.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[nha,]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[nha,]
            Error on the measured response.
        width : float
             Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.
        niter : int
            Number of iterations for the log amplitude fit.
        moving_window : float
            Only fit hour angles within +/- window * width from the peak.
            Note that the peak location is updated with each iteration.
            Set to None to fit all hour angles where resp_err > 0.0.

        Returns
        -------
        param : np.ndarray[nparam,]
            Best-fit model parameters.
        param_cov : np.ndarray[nparam, nparam]
            Covariance of the best-fit model parameters.
        chisq : np.ndarray[2,]
            Chi-squared of the fit to amplitude and phase.
        ndof : np.ndarray[2,]
            Number of degrees of freedom of the fit to amplitude and phase.
        """
        min_nfit = min(self.npara, self.nparp) + 1

        window = width * moving_window if (width and moving_window) else None

        # Prepare amplitude data
        model_amp = np.abs(resp)
        w0 = tools.invert_no_zero(resp_err) ** 2

        # Only perform fit if there is enough data.
        this_flag = (model_amp > 0.0) & (w0 > 0.0)
        ndata = int(np.sum(this_flag))
        if ndata < min_nfit:
            raise RuntimeError("Number of data points less than number of parameters.")

        # Prepare amplitude data
        ya = np.log(model_amp)

        # Prepare phase data.
        phi = np.angle(resp)
        phi0 = phi[np.argmin(np.abs(ha))]

        yp = phi - phi0
        yp += (yp < -np.pi) * 2 * np.pi - (yp > np.pi) * 2 * np.pi
        yp += phi0

        # Calculate vandermonde matrix
        A = self._vander(ha, self.poly_deg_amp)
        center = 0.0
        coeff = None

        # Iterate to obtain model estimate for amplitude
        for kk in range(niter):
            wk = w0 * model_amp**2

            if window is not None:
                if kk > 0:
                    center = self.peak(param=coeff)

                if np.isnan(center):
                    raise RuntimeError("No peak found.")

                wk *= (np.abs(ha - center) <= window).astype(np.float64)

                ndata = int(np.sum(wk > 0.0))
                if ndata < min_nfit:
                    raise RuntimeError(
                        "Number of data points less than number of parameters."
                    )

            C = np.dot(A.T, wk[:, np.newaxis] * A)
            coeff = lstsq(C, np.dot(A.T, wk * ya))[0]

            model_amp = np.exp(np.dot(A, coeff))

        # Compute final value for amplitude
        center = self.peak(param=coeff)

        if np.isnan(center):
            raise RuntimeError("No peak found.")

        wf = w0 * model_amp**2
        if window is not None:
            wf *= (np.abs(ha - center) <= window).astype(np.float64)

            ndata = int(np.sum(wf > 0.0))
            if ndata < min_nfit:
                raise RuntimeError(
                    "Number of data points less than number of parameters."
                )

        cova = inv(np.dot(A.T, wf[:, np.newaxis] * A))
        coeffa = np.dot(cova, np.dot(A.T, wf * ya))

        mamp = np.dot(A, coeffa)

        # Compute final value for phase
        A = self._vander(ha, self.poly_deg_phi)

        covp = inv(np.dot(A.T, wf[:, np.newaxis] * A))
        coeffp = np.dot(covp, np.dot(A.T, wf * yp))

        mphi = np.dot(A, coeffp)

        # Compute chisq per degree of freedom
        ndofa = ndata - self.npara
        ndofp = ndata - self.nparp

        ndof = np.array([ndofa, ndofp])
        chisq = np.array([np.sum(wf * (ya - mamp) ** 2), np.sum(wf * (yp - mphi) ** 2)])

        # Scale the parameter covariance by chisq per degree of freedom.
        # Equivalent to using RMS of the residuals to set the absolute error
        # on the measurements.
        if not absolute_sigma:
            scale_factor = chisq * tools.invert_no_zero(ndof.astype(np.float32))
            cova *= scale_factor[0]
            covp *= scale_factor[1]

        param = np.concatenate((coeffa, coeffp))

        param_cov = np.zeros((self.nparam, self.nparam), dtype=np.float32)
        param_cov[: self.npara, : self.npara] = cova
        param_cov[self.npara :, self.npara :] = covp

        return param, param_cov, chisq, ndof

    def peak(self, param=None):
        """Find the peak of the transit.

        Parameters
        ----------
        param : np.ndarray[..., nparam]
            Coefficients of the polynomial model for log amplitude.
            Defaults to `self.param`.

        Returns
        -------
        peak : np.ndarray[...]
            Location of the maximum amplitude in degrees hour angle.
            If the polynomial does not have a maximum, then NaN is returned.
        """
        if param is None:
            param = self.param

        der1 = self._deriv(param[..., : self.npara], m=1, axis=-1)
        der2 = self._deriv(param[..., : self.npara], m=2, axis=-1)

        shp = der1.shape[:-1]
        peak = np.full(shp, np.nan, dtype=der1.dtype)

        for ind in np.ndindex(*shp):
            ider1 = der1[ind]

            if np.any(~np.isfinite(ider1)):
                continue

            root = self._root(ider1)
            xmax = np.real(
                [
                    rr
                    for rr in root
                    if (rr.imag == 0) and (self._eval(rr, der2[ind]) < 0.0)
                ]
            )

            peak[ind] = xmax[np.argmin(np.abs(xmax))] if xmax.size > 0 else np.nan

        return peak

    def _model(self, ha, elementwise=False):
        amp = self._fast_eval(
            ha, self.param[..., : self.npara], elementwise=elementwise
        )
        phi = self._fast_eval(
            ha, self.param[..., self.npara :], elementwise=elementwise
        )

        return np.exp(amp) * (np.cos(phi) + 1.0j * np.sin(phi))

    def _jacobian_amp(self, ha, elementwise=False):
        jac = self._vander(ha, self.poly_deg_amp)
        if not elementwise:
            jac = np.rollaxis(jac, -1)
            if self.N is not None:
                slc = (None,) * len(self.N)
                jac = jac[slc]

        return jac

    def _jacobian_phi(self, ha, elementwise=False):
        jac = self._vander(ha, self.poly_deg_phi)
        if not elementwise:
            jac = np.rollaxis(jac, -1)
            if self.N is not None:
                slc = (None,) * len(self.N)
                jac = jac[slc]

        return jac

    @property
    def ndofa(self):
        """
        Number of degrees of freedom for the amplitude fit.

        Returns
        -------
        ndofa : np.ndarray[...]
            Number of degrees of freedom of the amplitude fit.
        """
        return self.ndof[..., 0]

    @property
    def ndofp(self):
        """
        Number of degrees of freedom for the phase fit.

        Returns
        -------
        ndofp : np.ndarray[...]
            Number of degrees of freedom of the phase fit.
        """
        return self.ndof[..., 1]

    @property
    def parameter_names(self):
        """Array of strings containing the name of the fit parameters."""
        return np.array(
            [f"{self.poly_type}_poly_amp_coeff{p}" for p in range(self.npara)]
            + [f"{self.poly_type}_poly_phi_coeff{p}" for p in range(self.nparp)],
            dtype=np.bytes_,
        )
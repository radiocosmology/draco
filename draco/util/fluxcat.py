"""
Catalog the measured flux densities of astronomical sources

This module contains tools for cataloging astronomical sources
and predicting their flux density at radio frequencies based on
previous measurements.
"""

from abc import ABCMeta, abstractmethod
import os
import fnmatch
import inspect
import warnings

from collections import OrderedDict
import json
import pickle

import numpy as np
import base64
import datetime
import time

from caput import misc
import caput.time as ctime

from ..ephem import catalogs

from .tools import ensure_list

# Define nominal frequency. Sources in catalog are ordered according to
# their predicted flux density at this frequency. Also acts as default
# pivot point in spectral fits.
FREQ_NOMINAL = 600.0

# Define the source collections that should be loaded when this module is imported.
# These catalogs are provided by ch_ephem.
DEFAULT_COLLECTIONS = [
    "primary_calibrators_perley2016",
    "specfind_v2_5Jy_vollmer2009",
]


# ==================================================================================
class FitSpectrum(metaclass=ABCMeta):
    """A base class for modeling and fitting spectra.  Any spectral model
    used by FluxCatalog should be derived from this class.

    The `fit` method should be used to populate the `param`, `param_cov`, and `stats`
    attributes.  The `predict` and `uncertainty` methods can then be used to obtain
    the flux density and uncertainty at arbitrary frequencies.

    Attributes
    ----------
    param : np.ndarray[nparam, ]
        Best-fit parameters.
    param_cov : np.ndarray[nparam, nparam]
        Covariance of the fit parameters.
    stats : dict
        Dictionary that contains statistics related to the fit.
        Must include 'chisq' and 'ndof'.

    Abstract Methods
    ----------------
    Any subclass of FitSpectrum must define these methods:
        fit
        _get_x
        _fit_func
        _deriv_fit_func
    """

    def __init__(self, param=None, param_cov=None, stats=None):
        """Instantiates a FitSpectrum object."""

        self.param = param
        self.param_cov = param_cov
        self.stats = stats

    def predict(self, freq):
        """Predicts the flux density at a particular frequency."""

        x = self._get_x(freq)

        return self._fit_func(x, *self.param)

    def uncertainty(self, freq, alpha=0.32):
        """Predicts the uncertainty on the flux density at a
        particular frequency.
        """

        from scipy.stats import t

        prob = 1.0 - alpha / 2.0
        tval = t.ppf(prob, self.stats["ndof"])
        nparam = len(self.param)

        x = self._get_x(freq)

        dfdp = self._deriv_fit_func(x, *self.param)

        if hasattr(x, "__iter__"):
            df2 = np.zeros(len(x), dtype=np.float64)
        else:
            df2 = 0.0

        for ii in range(nparam):
            for jj in range(nparam):
                df2 += dfdp[ii] * dfdp[jj] * self.param_cov[ii][jj]

        return tval * np.sqrt(df2)

    @abstractmethod
    def fit(self, freq, flux, eflux, *args):
        return

    @abstractmethod
    def _get_x(self, freq):
        return

    @staticmethod
    @abstractmethod
    def _fit_func(x, *param):
        return

    @staticmethod
    @abstractmethod
    def _deriv_fit_func(x, *param):
        return


class CurvedPowerLaw(FitSpectrum):
    """
    Class to fit a spectrum to a polynomial in log-log space, given by

    .. math::
        \\ln{S} = a_{0} + a_{1} \\ln{\\nu'} + a_{2} \\ln{\\nu'}^2 + a_{3} \\ln{\\nu'}^3 + \\dots

    where :math:`S` is the flux density, :math:`\\nu'` is the (normalized) frequency,
    and :math:`a_{i}` are the fit parameters.

    Parameters
    ----------
    nparam : int
        Number of parameters.  This sets the order of the polynomial.
        Default is 2 (powerlaw).
    freq_pivot : float
        The pivot frequency :math:`\\nu' = \\nu / freq_pivot`.
        Default is :py:const:`FREQ_NOMINAL`.
    """  # noqa: E501

    def __init__(self, freq_pivot=FREQ_NOMINAL, nparam=2, *args, **kwargs):
        """Instantiates a CurvedPowerLaw object"""

        super().__init__(*args, **kwargs)

        # Set the additional model kwargs
        self.freq_pivot = freq_pivot

        if self.param is not None:
            self.nparam = len(self.param)
        else:
            self.nparam = nparam

    def fit(self, freq, flux, eflux, flag=None):
        if flag is None:
            flag = np.ones(len(freq), dtype=bool)

        # Make sure we have enough measurements
        if np.sum(flag) >= self.nparam:
            # Apply flag
            fit_freq = freq[flag]
            fit_flux = flux[flag]
            fit_eflux = eflux[flag]

            # Convert to log space
            x = self._get_x(fit_freq)
            y = np.log(fit_flux)

            # Vandermonde matrix
            A = self._vandermonde(x, self.nparam)

            # Data covariance matrix
            C = np.diag((fit_eflux / fit_flux) ** 2.0)

            # Parameter estimate and covariance
            param_cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))

            param = np.dot(param_cov, np.dot(A.T, np.linalg.solve(C, y)))

            # Compute residuals
            resid = y - np.dot(A, param)

            # Change the overall normalization to linear units
            param[0] = np.exp(param[0])

            for ii in range(self.nparam):
                for jj in range(self.nparam):
                    param_cov[ii, jj] *= (param[0] if ii == 0 else 1.0) * (
                        param[0] if jj == 0 else 1.0
                    )

            # Save parameter estimate and covariance to instance
            self.param = param.tolist()
            self.param_cov = param_cov.tolist()

            # Calculate statistics
            if not isinstance(self.stats, dict):
                self.stats = {}
            self.stats["ndof"] = len(x) - self.nparam
            self.stats["chisq"] = np.sum(resid**2 / np.diag(C))

        # Return results
        return self.param, self.param_cov, self.stats

    def _get_x(self, freq):
        return np.log(freq / self.freq_pivot)

    @staticmethod
    def _vandermonde(x, nparam):
        return np.vstack(tuple([x**rank for rank in range(nparam)])).T

    @staticmethod
    def _fit_func(x, *param):
        return param[0] * np.exp(
            np.sum(
                [par * x ** (rank + 1) for rank, par in enumerate(param[1:])], axis=0
            )
        )

    @staticmethod
    def _deriv_fit_func(x, *param):
        z = param[0] * np.exp(
            np.sum(
                [par * x ** (rank + 1) for rank, par in enumerate(param[1:])], axis=0
            )
        )

        dfdp = np.array([z * x**rank for rank in range(len(param))])
        dfdp[0] /= param[0]

        return dfdp


class MetaFluxCatalog(type):
    """Metaclass for FluxCatalog.  Defines magic methods
    for the class that can act on and provide access to the
    catalog of all astronomical sources.
    """

    def __str__(self):
        return self.string()

    def __iter__(self):
        return self.iter()

    def __reversed__(self):
        return self.reversed()

    def __len__(self):
        return self.len()

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, item):
        try:
            obj = self.get(item)
        except KeyError:
            obj = None

        return obj is not None


class FluxCatalog(metaclass=MetaFluxCatalog):
    """
    Class for cataloging astronomical sources and predicting
    their flux density at radio frequencies based on spectral fits
    to previous measurements.

    Class methods act upon and provide access to the catalog of
    all sources.  Instance methods act upon and provide access
    to individual sources.  All instances are stored in an
    internal class dictionary.

    Attributes
    ----------
    fields : list
        List of attributes that are read-from and written-to the
        JSON catalog files.
    model_lookup : dict
        Dictionary that provides access to the various models that
        can be fit to the spectrum.  These models should be
        subclasses of FitSpectrum.
    """

    fields = [
        "ra",
        "dec",
        "alternate_names",
        "model",
        "model_kwargs",
        "stats",
        "param",
        "param_cov",
        "measurements",
    ]

    model_lookup = {"CurvedPowerLaw": CurvedPowerLaw}

    _entries = {}
    _collections = {}
    _alternate_name_lookup = {}

    def __init__(
        self,
        name,
        ra=None,
        dec=None,
        alternate_names=[],
        model="CurvedPowerLaw",
        model_kwargs=None,
        stats=None,
        param=None,
        param_cov=None,
        measurements=None,
        overwrite=0,
    ):
        """
        Instantiates a FluxCatalog object for an astronomical source.

        Parameters
        ----------
        name : string
            Name of the source.  The convention for the source name is to
            use the MAIN_ID in the SIMBAD database in all uppercase letters
            with spaces replaced by underscores.

        ra : float
            Right Ascension in degrees.

        dec : float
            Declination in degrees.

        alternate_names : list of strings
            Alternate names for the source.  Ideally should include all alternate names
            present in the SIMBAD database using the naming convention specified above.

        model : string
            Name of FitSpectrum subclass.

        model_kwargs : dict
            Dictionary containing keywords required by the model.

        stats : dict
            Dictionary containing statistics from model fit.

        param : list, length nparam
            Best-fit parameters.

        param_cov : 2D-list, size nparam x nparam
            Estimate of covariance of fit parameters.

        measurements : 2D-list, size nmeas x 7
            List of measurements of the form:
            [freq, flux, eflux, flag, catalog, epoch, citation].
            Should use the add_measurement method to populate this list.

        overwrite : int between 0 and 2
            Action to take in the event that this source is already in the catalog:
            - 0 - Return the existing entry.
            - 1 - Add the measurements to the existing entry.
            - 2 - Overwrite the existing entry.
            Default is 0.
        """

        # The name argument is a unique identifier into the catalog.
        # Check if there is already a source in the catalog with the
        # input name.  If there is, then the behavior is set by the
        # overwrite argument.
        if (overwrite < 2) and (name in FluxCatalog):
            # Return existing entry
            print(f"{name} already has an entry in catalog.", end=" ")
            if overwrite == 0:
                print("Returning existing entry.")
                self = FluxCatalog[name]

            # Add any measurements to existing entry
            elif overwrite == 1:
                print("Adding measurements to existing entry.")
                self = FluxCatalog[name]
                if measurements is not None:
                    self.add_measurement(*measurements)
                    self.fit_model()

        else:
            # Create new instance for this source.
            self.name = format_source_name(name)

            # Initialize object attributes
            # Basic info:
            self.ra = ra
            self.dec = dec

            self.alternate_names = [
                format_source_name(aname) for aname in ensure_list(alternate_names)
            ]

            # Measurements:
            self.measurements = measurements

            # Best-fit model:
            self.model = model
            self.param = param
            self.param_cov = param_cov
            self.stats = stats
            self.model_kwargs = model_kwargs
            if self.model_kwargs is None:
                self.model_kwargs = {}

            # Create model object
            self._model = self.model_lookup[self.model](
                param=self.param,
                param_cov=self.param_cov,
                stats=self.stats,
                **self.model_kwargs,
            )

            # Populate the kwargs that were used
            arg_list = misc.getfullargspec(self.model_lookup[self.model].__init__)
            if len(arg_list.args) > 1:
                keys = arg_list.args[1:]
                for key in keys:
                    if hasattr(self._model, key):
                        self.model_kwargs[key] = getattr(self._model, key)

            if not self.model_kwargs:
                self.model_kwargs = None

            # Save to class dictionary
            self._entries[self.name] = self

            # Add alternate names to class dictionary so they can be searched quickly
            for alt_name in self.alternate_names:
                if alt_name in self._alternate_name_lookup:
                    alt_source = self._alternate_name_lookup[alt_name]
                    warnings.warn(
                        f"The alternate name {alt_name} for {self.name} is already "
                        f"held by the source {alt_source}."
                    )
                else:
                    self._alternate_name_lookup[alt_name] = self.name

    def add_measurement(
        self, freq, flux, eflux, flag=True, catalog=None, epoch=None, citation=None
    ):
        """Add entries to the list of measurements.  Each argument/keyword
        can be a list of items with length equal to 'len(flux)', or
        alternatively a single item in which case the same value is used
        for all measurements.

        Parameters
        ----------
        freq : float, list of floats
            Frequency in MHz.

        flux : float, list of floats
            Flux density in Jansky.

        eflux : float, list of floats
            Uncertainty on flux density in Jansky.

        flag : bool, list of bool
            If True, use this measurement in model fit.
            Default is True.

        catalog : string or None, list of strings or Nones
            Name of the catalog from which this measurement originates.
            Default is None.

        epoch : float or None, list of floats or Nones
            Year when this measurement was taken.
            Default is None.

        citation : string or None, list of strings or Nones
            Citation where this measurement can be found
            (e.g., 'Baars et al. (1977)').
            Default is None.

        """

        # Ensure that all of the inputs are lists
        # of the same length as flux
        flux = ensure_list(flux)
        nmeas = len(flux)

        freq = ensure_list(freq, nmeas)
        eflux = ensure_list(eflux, nmeas)
        flag = ensure_list(flag, nmeas)
        catalog = ensure_list(catalog, nmeas)
        epoch = ensure_list(epoch, nmeas)
        citation = ensure_list(citation, nmeas)

        # Store as list
        meas = [
            [
                freq[mm],
                flux[mm],
                eflux[mm],
                flag[mm],
                catalog[mm],
                epoch[mm],
                citation[mm],
            ]
            for mm in range(nmeas)
        ]

        # Add measurements to internal list
        if self.measurements is None:
            self.measurements = meas
        else:
            self.measurements += meas

        # Sort internal list by frequency
        isort = np.argsort(self.freq)
        self.measurements = [self.measurements[mm] for mm in isort]

    def fit_model(self):
        """Fit the measurements stored in the 'measurements' attribute with the
        spectral model specified in the 'model' attribute. This populates the
        'param', 'param_cov', and 'stats' attributes.
        """

        arg_list = misc.getfullargspec(self._model.fit).args[1:]

        args = [self.freq[self.flag], self.flux[self.flag], self.eflux[self.flag]]

        if (self.epoch is not None) and ("epoch" in arg_list):
            args.append(self.epoch[self.flag])

        self.param, self.param_cov, self.stats = self._model.fit(*args)

    def plot(self, legend=True, catalog=True, residuals=False):
        """Plot the measurements, best-fit model, and confidence interval.

        Parameters
        ----------
        legend : bool
            Show legend.  Default is True.

        catalog : bool
            If True, then label and color code the measurements according to
            their catalog.  If False, then label and color code the measurements
            according to their citation.  Default is True.

        residuals : bool
            Plot the residuals instead of the measurements and best-fit model.
            Default is False.
        """

        import matplotlib.pyplot as plt

        # Define plot parameters
        colors = ["blue", "darkorchid", "m", "plum", "mediumvioletred", "palevioletred"]
        markers = ["o", "*", "s", "p", "^"]
        sizes = [10, 12, 12, 12, 12]

        font = {"family": "sans-serif", "weight": "normal", "size": 16}

        plt.rc("font", **font)

        nplot = 500

        # Plot the model fit and uncertainty
        xrng = [np.floor(np.log10(self.freq.min())), np.ceil(np.log10(self.freq.max()))]
        xrng = [min(xrng[0], 2.0), max(xrng[1], 3.0)]

        fplot = np.logspace(*xrng, num=nplot)

        xrng = [10.0**xx for xx in xrng]

        if residuals:
            flux_hat = self.predict_flux(self.freq)
            flux = (self.flux - flux_hat) / flux_hat
            eflux = self.eflux / flux_hat
            model = np.zeros_like(fplot)
            delta = self.predict_uncertainty(fplot) / self.predict_flux(fplot)
            ylbl = "Residuals: " + r"$(S - \hat{S}) / \hat{S}$"
            yrng = [-0.50, 0.50]
        else:
            flux = self.flux
            eflux = self.eflux
            model = self.predict_flux(fplot)
            delta = self.predict_uncertainty(fplot)
            ylbl = "Flux Density [Jy]"
            yrng = [model.min(), model.max()]

        plt.fill_between(
            fplot, model - delta, model + delta, facecolor="darkgray", alpha=0.3
        )
        plt.plot(
            fplot,
            model - delta,
            fplot,
            model + delta,
            color="black",
            linestyle="-",
            linewidth=0.5,
        )

        plt.plot(
            fplot, model, color="black", linestyle="-", linewidth=1.0, label=self.model
        )

        # Plot the measurements
        if catalog:
            cat_uniq = list(set(self.catalog))
        else:
            cat_uniq = list(set(self.citation))

        # Loop over catalogs/citations
        for ii, cat in enumerate(cat_uniq):
            if catalog:
                pind = np.array([cc == cat for cc in self.catalog])
            else:
                pind = np.array([cc == cat for cc in self.citation])

            if cat is None:
                pcol = "black"
                pmrk = "o"
                psz = 10
                lbl = "Meas."
            else:
                pcol = colors[ii % len(colors)]
                pmrk = markers[ii // len(colors)]
                psz = sizes[ii // len(colors)]
                lbl = cat

            plt.errorbar(
                self.freq[pind],
                flux[pind],
                yerr=eflux[pind],
                color=pcol,
                marker=pmrk,
                markersize=psz,
                linestyle="None",
                label=lbl,
            )

        # Set log axis
        ax = plt.gca()
        ax.set_xscale("log")
        if not residuals:
            ax.set_yscale("log")
        plt.xlim(xrng)
        plt.ylim(yrng)

        plt.grid(b=True, which="both")

        # Plot lines denoting CHIME band
        plt.axvspan(400.0, 800.0, color="green", alpha=0.1)

        # Create a legend
        if legend:
            plt.legend(loc="lower left", numpoints=1, prop=font)

        # Set labels
        plt.xlabel("Frequency [MHz]")
        plt.ylabel(ylbl)

        # Create block with statistics
        if not residuals:
            txt = r"$\chi^2 = %0.2f$ $(%d)$" % (self.stats["chisq"], self.stats["ndof"])

            plt.text(
                0.95,
                0.95,
                txt,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )

        # Create title
        ttl = self.name.replace("_", " ")
        plt.title(ttl)

    def predict_flux(self, freq, epoch=None):
        """Predict the flux density of the source at a particular
        frequency and epoch.

        Parameters
        ----------
        freq : float, np.array of floats
            Frequency in MHz.

        epoch : float, np.array of floats
            Year.  Defaults to current year.

        Returns
        -------
        flux : float, np.array of floats
            Flux density in Jansky.

        """

        arg_list = misc.getfullargspec(self._model.predict).args[1:]

        if (epoch is not None) and ("epoch" in arg_list):
            args = [freq, epoch]
        else:
            args = [freq]

        return self._model.predict(*args)

    def predict_uncertainty(self, freq, epoch=None):
        """Calculate the uncertainty in the estimate of the flux density
        of the source at a particular frequency and epoch.

        Parameters
        ----------
        freq : float, np.array of floats
            Frequency in MHz.

        epoch : float, np.array of floats
            Year.  Defaults to current year.

        Returns
        -------
        flux_uncertainty : float, np.array of floats
            Uncertainty on the flux density in Jansky.

        """

        arg_list = misc.getfullargspec(self._model.uncertainty).args[1:]

        if (epoch is not None) and ("epoch" in arg_list):
            args = [freq, epoch]
        else:
            args = [freq]

        return self._model.uncertainty(*args)

    def to_dict(self):
        """Returns an ordered dictionary containing attributes
        for this instance object.  Used to dump the information
        stored in the instance object to a file.

        Returns
        -------
        flux_body_dict : dict
            Dictionary containing all attributes listed in
            the 'fields' class attribute.
        """

        flux_body_dict = OrderedDict()

        for key in self.fields:
            if hasattr(self, key) and (getattr(self, key) is not None):
                flux_body_dict[key] = getattr(self, key)

        return flux_body_dict

    def __str__(self):
        """Returns a string containing basic information about the source.
        Called by the print statement.
        """
        flux = self.predict_flux(FREQ_NOMINAL)
        percent = 100.0 * self.predict_uncertainty(FREQ_NOMINAL) / flux
        return (
            f"{self.name:<25.25s} {self.ra:>6.2f} {self.dec:>6.2f} "
            f"{len(self):>6d} {flux:^15.1f} {percent:^15.1f}"
        )

    def __len__(self):
        """Returns the number of measurements of the source."""
        return len(self.measurements) if self.measurements is not None else 0

    def print_measurements(self):
        """Print all measurements."""

        out = []

        # Define header
        hdr = (
            f"{'Frequency':<10s} {'Flux':>8s} {'Error':>8s} {'Flag':>6s} "
            f"{'Catalog':>8s} {'Epoch':>8s}   {'Citation':<60s}"
        )

        units = (
            f"{'[MHz]':<10s} {'[Jy]':>8s} {'[%]':>8s} "
            f"{'':>6s} {'':>8s} {'':>8s}   {'':<60s}"
        )

        # Setup the title
        out.append("".join(["="] * max(len(hdr), len(units))))
        out.append("NAME: " + self.name.replace("_", " "))
        out.append(f"RA:   {self.ra:>6.2f} deg")
        out.append(f"DEC:  {self.dec:>6.2f} deg")
        out.append(f"{len(self.measurements):d} Measurements")

        out.append("".join(["-"] * max(len(hdr), len(units))))
        out.append(hdr)
        out.append(units)
        out.append("".join(["-"] * max(len(hdr), len(units))))

        # Add the measurements
        for meas in self.measurements:
            if meas[5] is None:
                epoch_fmt = "{5:>8s}"
            else:
                epoch_fmt = "{5:>8.1f}"

            fmt_string = (
                "{0:<10.1f} {1:>8.1f} {2:>8.1f} {3:>6s} {4:>8s} "
                + epoch_fmt
                + "   {6:<.60s}"
            )

            entry = fmt_string.format(
                meas[0],
                meas[1],
                100.0 * meas[2] / meas[1],
                "Good" if meas[3] else "Bad",
                meas[4] if meas[4] is not None else "--",
                meas[5] if meas[5] is not None else "--",
                meas[6] if meas[6] is not None else "--",
            )

            out.append(entry)

        # Print
        print("\n".join(out))

    @property
    def skyfield(self):
        """Skyfield star representation :class:`skyfield.starlib.Star`
        for the source.
        """
        return ctime.skyfield_star_from_ra_dec(self.ra, self.dec, self.name)

    @property
    def freq(self):
        """Frequency of measurements in MHz."""
        return np.array([meas[0] for meas in self.measurements])

    @property
    def flux(self):
        """Flux measurements in Jansky."""
        return np.array([meas[1] for meas in self.measurements])

    @property
    def eflux(self):
        """Error on the flux measurements in Jansky."""
        return np.array([meas[2] for meas in self.measurements])

    @property
    def flag(self):
        """Boolean flag indicating what measurements are used
        in the spectral fit.
        """
        return np.array([meas[3] for meas in self.measurements])

    @property
    def catalog(self):
        """Catalog from which each measurement originates."""
        return np.array([meas[4] for meas in self.measurements])

    @property
    def epoch(self):
        """Year that each measurement occured."""
        return np.array([meas[5] for meas in self.measurements])

    @property
    def citation(self):
        """Citation where more information on each measurement
        can be found.
        """
        return np.array([meas[6] for meas in self.measurements])

    @property
    def _sort_id(self):
        """Sources in the catalog are ordered according to this
        property.  Currently use the predicted flux at FREQ_NOMINAL
        in descending order.
        """
        # Multipy by -1 so that we will
        # sort from higher to lower flux
        return -self.predict_flux(
            FREQ_NOMINAL, epoch=get_epoch(datetime.datetime.now())
        )

    # =============================================================
    # Class methods that act on the entire catalog
    # =============================================================

    @classmethod
    def string(cls):
        """Print basic information about the sources in the catalog."""

        catalog_string = []

        # Print the header
        hdr = (
            f"{'Name':<25s} {'RA':^6s} {'Dec':^6s} "
            f"{'Nmeas':>6s} {'Flux':^15s} {'Error':^15s}"
        )

        units = (
            f"{'':<25s} {'[deg]':^6s} {'[deg]':^6s} {'':>6s} "
            f"{f'@{FREQ_NOMINAL} MHz [Jy]':^15s} {f'@{FREQ_NOMINAL} MHz [%]':^15s}"
        )

        catalog_string.append("".join(["-"] * max(len(hdr), len(units))))
        catalog_string.append(hdr)
        catalog_string.append(units)
        catalog_string.append("".join(["-"] * max(len(hdr), len(units))))

        # Loop over sorted entries and print
        for key in cls.sort():
            catalog_string.append(cls[key].__str__())

        return "\n".join(catalog_string)

    @classmethod
    def from_dict(cls, name, flux_body_dict):
        """Instantiates a FluxCatalog object for an astronomical source
        from a dictionary of kwargs.  Used when loading sources from a
        JSON catalog file.

        Parameters
        ----------
        name : str
            Name of the astronomical source.

        flux_body_dict : dict
            Dictionary containing some or all of the keyword arguments
            listed in the __init__ function for this class.

        Returns
        -------
        obj : FluxCatalog instance
            Object that can be used to predict the flux of this source,
            plot flux measurements, etc.

        """

        arg_list = misc.getfullargspec(cls.__init__).args[2:]

        kwargs = {
            field: flux_body_dict[field]
            for field in arg_list
            if field in flux_body_dict
        }

        return cls(name, **kwargs)

    @classmethod
    def get(cls, key):
        """Searches the catalog for a source.  First checks against the
        'name' of each entry, then checks against the 'alternate_names'
        of each entry.

        Parameters
        ----------
        key : str
            Name of the astronomical source.

        Returns
        -------
        obj : FluxCatalog instance
            Object that can be used to predict the flux of this source,
            plot flux measurements, etc.

        """

        # Check that key is a string
        if not isinstance(key, str):
            raise TypeError("Provide source name as string.")

        fkey = format_source_name(key)

        # First check names
        obj = cls._entries.get(fkey, None)

        # Next check alternate names
        if obj is None:
            afkey = cls._alternate_name_lookup.get(fkey, None)
            if afkey is not None:
                obj = cls._entries.get(afkey)

        # Check if the object was found
        if obj is None:
            raise KeyError(f"{fkey} was not found.")

        # Return the body corresponding to this source
        return obj

    @classmethod
    def delete(cls, source_name):
        """Deletes a source from the catalog.

        Parameters
        ----------
        source_name : str
            Name of the astronomical source.

        """

        try:
            obj = cls.get(source_name)
        except KeyError:
            key = None
        else:
            key = obj.name

        if key is not None:
            obj = cls._entries.pop(key)

            for akey in obj.alternate_names:
                cls._alternate_name_lookup.pop(akey, None)

            del obj

    @classmethod
    def sort(cls):
        """Sorts the entries in the catalog by their flux density
        at FREQ_NOMINAL in descending order.

        Returns
        -------
        names : list of str
            List of source names in correct order.

        """

        keys = []
        for name, body in cls._entries.items():
            keys.append((body._sort_id, name))

        keys.sort()

        return [key[1] for key in keys]

    @classmethod
    def keys(cls):
        """Alias for sort.

        Returns
        -------
        names : list of str
            List of source names in correct order.

        """
        return cls.sort()

    @classmethod
    def iter(cls):
        """Iterates through the sources in the catalog.

        Returns
        -------
        it : iterator
            Provides the name of each source in the catalog
            in the order specified by the 'sort' class method.

        """
        return iter(cls.sort())

    @classmethod
    def reversed(cls):
        """Iterates through the sources in the catalog
        in reverse order.

        Returns
        -------
        it : iterator
            Provides the name of each source in the catalog
            in the reverse order as that specified by the
            'sort' class method.

        """
        return reversed(cls.sort())

    @classmethod
    def iteritems(cls):
        """Iterates through the sources in the catalog.

        Returns
        -------
        it : iterator
            Provides (name, object) for each source in the catalog
            in the order specified by the 'sort' class method.

        """
        return iter([(key, cls._entries[key]) for key in cls.sort()])

    @classmethod
    def len(cls):
        """Number of sources in the catalog.

        Returns
        -------
        N : int

        """
        return len(cls._entries)

    @classmethod
    def available_collections(cls):
        """Search the local directory for potential collections that
        can be loaded.

        Returns
        -------
        collections : list of (str, [str, ...])
            List containing a tuple for each collection.  The tuple contains
            the filename of the collection (str) and the sources it contains
            (list of str).

        """

        # Determine the directory where this class is located
        current_file = inspect.getfile(cls.__class__)
        current_dir = os.path.abspath(os.path.dirname(os.path.dirname(current_file)))

        # Search this directory recursively for JSON files.
        # Load each one that is found into a dictionary and
        # return the number of sources and source names.
        matches = []
        for root, dirnames, filenames in os.walk(current_dir):
            for filename in fnmatch.filter(filenames, "*.json") + fnmatch.filter(
                filenames, "*.pickle"
            ):
                full_path = os.path.join(root, filename)

                # Read into dictionary
                with open(full_path) as fp:
                    collection_dict = json.load(fp, object_hook=json_numpy_obj_hook)

                # Append (path, number of sources, source names) to list
                matches.append((full_path, list(collection_dict.keys())))

        # Return matches
        return matches

    @classmethod
    def print_available_collections(cls, verbose=False):
        """Print information about the available collections.

        Parameters
        ----------
        verbose : bool
            If True, then print all source names in addition to the names
            of the files and number of sources.  Default is False.
        """
        for cc in cls.available_collections():
            _print_collection_summary(*cc, verbose=verbose)

    @classmethod
    def loaded_collections(cls):
        """Return the collections that have been loaded.

        Returns
        -------
        collections : list of (str, [str, ...])
            List containing a tuple for each collection.  The tuple contains
            the filename of the collection (str) and the sources it contains
            (list of str).
        """
        return list(cls._collections.items())

    @classmethod
    def print_loaded_collections(cls, verbose=False):
        """Print information about the collection that have been loaded.

        Parameters
        ----------
        verbose : bool
            If True, then print all source names in addition to the names
            of the files and number of sources.  Default is False.
        """
        for cat, sources in cls._collections.items():
            _print_collection_summary(cat, sources, verbose=verbose)

    @classmethod
    def delete_loaded_collection(cls, cat):
        sources_to_delete = cls._collections.pop(cat)

        for source_name in sources_to_delete:
            cls.delete(source_name)

    @classmethod
    def dump(cls, filename):
        """Dumps the contents of the catalog to a file.

        Parameters
        ----------
        filename : str
            Valid path name.  Should have .json or .pickle extension.

        """

        # Parse filename
        filename = os.path.expandvars(os.path.expanduser(filename))
        path = os.path.abspath(os.path.dirname(filename))
        ext = os.path.splitext(filename)[1]

        if ext not in [".pickle", ".json"]:
            raise ValueError(f"Do not recognize '{ext}' extension.")

        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

        # Sort based on _sort_id
        keys = cls.sort()

        # Place a dictionary with the information
        # stored in each object into an OrderedDict
        output = OrderedDict()
        for key in keys:
            output[key] = cls._entries[key].to_dict()

        # Dump this dictionary to file
        with open(filename, "w") as fp:
            if ext == ".json":
                json.dump(output, fp, cls=NumpyEncoder, indent=4)
            elif ext == ".pickle":
                pickle.dump(output, fp)

    @classmethod
    def load(cls, filename, overwrite=0, set_globals=False, verbose=False):
        """
        Load the contents of a file into the catalog.

        Parameters
        ----------
        filename : str
            Valid path name.  Should have .json or .pickle extension.
        overwrite : int between 0 and 2
            Action to take in the event that this source is already in the catalog:
            - 0 - Return the existing entry.
            - 1 - Add any measurements to the existing entry.
            - 2 - Overwrite the existing entry.
            Default is 0.
        set_globals : bool
            If True, this creates a variable in the global space
            for each source in the file.  Default is False.
        verbose : bool
            If True, print some basic info about the contents of
            the file as it is loaded. Default is False.
        """

        # Parse filename
        # Define collection name as basename of file without extension
        filename = os.path.expandvars(os.path.expanduser(filename))
        collection_name, ext = os.path.splitext(os.path.basename(filename))

        # Check if the file actually exists and has the correct extension
        if not os.path.isfile(filename):
            raise ValueError(f"{filename} does not exist.")

        if ext not in [".pickle", ".json"]:
            raise ValueError(f"Do not recognize '{ext}' extension.")

        # Load contents of file into a dictionary
        with open(filename) as fp:
            if ext == ".json":
                collection_dict = json.load(fp, object_hook=json_numpy_obj_hook)
            elif ext == ".pickle":
                collection_dict = pickle.load(fp)

        return cls.load_dict(
            collection_dict, collection_name, overwrite, set_globals, verbose
        )

    @classmethod
    def load_dict(
        cls,
        collection_dict,
        collection_name,
        overwrite=0,
        set_globals=False,
        verbose=False,
    ):
        """
        Load the contents of a dict into the catalog.

        Parameters
        ----------
        collection_dict : dict
            keys are source names, values are the sources
        collection_name : str
            Name of the collection
        overwrite : int between 0 and 2
            Action to take in the event that this source is already in the catalog:
            - 0 - Return the existing entry.
            - 1 - Add any measurements to the existing entry.
            - 2 - Overwrite the existing entry.
            Default is 0.
        set_globals : bool
            If True, this creates a variable in the global space
            for each source in the file.  Default is False.
        verbose : bool
            If True, print some basic info about the contents of
            the file as it is loaded. Default is False.
        """
        # Add this to the list of files
        cls._collections[collection_name] = list(collection_dict.keys())

        # If requested, print some basic info about the collection
        if verbose:
            _print_collection_summary(cls._collections[collection_name])

        # Loop through dictionary and add each source to the catalog
        for key, value in collection_dict.items():
            # Add overwrite keyword
            value["overwrite"] = overwrite

            # Create object for this source
            obj = cls.from_dict(key, value)

            # If requested, create a variable in the global space
            # containing the object for this source.
            if set_globals:
                varkey = varname(key)
                globals()[varkey] = obj


def get_epoch(date):
    """Return the epoch for a date.

    Parameters
    ----------
    date : datetime.datetime
        Date to calculate epoch

    Returns
    -------
    epoch : float
        The fractional-year epoch
    """

    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year + 1, month=1, day=1)

    yearElapsed = sinceEpoch(date) - sinceEpoch(startOfThisYear)
    yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def varname(name):
    """Create a python variable name from `name`.

    The variable name replaces spaces in `name` with
    underscores and adds a leading underscore if `name`
    starts with a digit.

    Parameters
    ----------
    name : str
        The name to create a variable name for

    Returns
    -------
    varname : str
        The python variable name.
    """
    varname = name.replace(" ", "_")

    if varname[0].isdigit():
        varname = "_" + varname

    return varname


def format_source_name(input_name):
    """Standardise the name of a source.

    Parameters
    ----------
    input_name: str
        The name to format

    Returns
    formatted_name: str
        The name after formatting.
    """
    # Address some common naming conventions.
    if input_name.startswith("NAME "):
        # SIMBAD prefixes common source names with 'NAME '.
        # Remove this.
        output_name = input_name[5:]

    elif not any(char.isdigit() for char in input_name):
        # We have been using PascalCase to denote common source names.
        # Convert from CygA, HerA, PerB ---> Cyg A, Her A, Per B.
        output_name = input_name[0]
        for ii in range(1, len(input_name)):
            if input_name[ii - 1].islower() and input_name[ii].isupper():
                output_name += " " + input_name[ii]
            else:
                output_name += input_name[ii]

    else:
        # No funny business with the input_name in this case.
        output_name = input_name

    # Remove multiple spaces.  Replace single spaces with underscores.
    output_name = "_".join(output_name.split())

    # Return the name in all uppercase.
    return output_name.upper()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags["C_CONTIGUOUS"]:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert cont_obj.flags["C_CONTIGUOUS"]
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return {
                "__ndarray__": data_b64,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = base64.b64decode(dct["__ndarray__"])
        return np.frombuffer(data, dct["dtype"]).reshape(dct["shape"])
    return dct


def _print_collection_summary(collection_name, source_names, verbose=True):
    """This prints out information about a collection of sources
    in a standardized way.

    Parameters
    ----------
    collection_name : str
        Name of the collection.

    source_names : list of str
        Names of the sources in the collection.

    verbose : bool
        If true, then print out all of the source names.
    """

    ncol = 4
    nsrc = len(source_names)

    # Create a header containing the collection name and number of sources
    header = collection_name + "  (%d Sources)" % nsrc
    print(header)

    # Print the sources contained in this collection
    if verbose:
        # Seperator
        print("".join(["-"] * len(header)))

        # Source names
        for ii in range(0, nsrc, ncol):
            jj = min(ii + ncol, nsrc)
            print((" ".join(["%-25s"] * (jj - ii))) % tuple(source_names[ii:jj]))

        # Space
        print("")


# Load the default collections
for col in DEFAULT_COLLECTIONS:
    FluxCatalog.load_dict(catalogs.load(col), col)

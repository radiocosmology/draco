"""
======================================================
Map making tasks (:mod:`~draco.analysis.ringmapmaker`)
======================================================

.. currentmodule:: draco.analysis.ringmapmaker

Tools for beamforming data for arrays with a cartesian layout.

Tasks
=====

.. autosummary::
    :toctree: generated/

    MakeVisGrid
    BeamformNS
    BeamformEW
    RingMapMaker
    TikhonovRingMapMakerAnalytical
    TikhonovRingMapMakerExternal
    WienerRingMapMakerAnalytical
    WienerRingMapMakerExternal
    RADependentWeights
"""
import numpy as np
import scipy.constants
from mpi4py import MPI

from caput import config

from ..core import task
from ..core import io
from ..util import tools
from ..core import containers
from . import transform


class MakeVisGrid(task.SingleTask):
    """Arrange the visibilities onto a 2D grid.

    This will fill out the visibilities in the half plane `x >= 0` where x is the EW
    baseline separation.
    """

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
        """
        self.telescope = io.get_telescope(tel)

    def process(self, sstream):
        """Computes the ringmap.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        rm : containers.RingMap
        """

        if np.all(
            sstream.prodstack.view(np.uint16).reshape(-1, 2)
            != self.telescope.uniquepairs
        ):
            raise ValueError(
                "Products in sstream do not match those in the beam transfers."
            )

        # Calculation the set of polarisations in the data, and which polarisation
        # index every entry corresponds to
        polpair = self.telescope.polarisation[self.telescope.uniquepairs].view("U2")
        pol, pind = np.unique(polpair, return_inverse=True)

        if len(pol) != 4:
            raise RuntimeError(f"Expected to find four polarisations. Got {pol}")

        # Find the mapping from a polarisation index to it's complement with the feeds
        # reversed
        pconjmap = np.unique([pj + pi for pi, pj in pol], return_inverse=True)[1]

        # Determine the layout of the visibilities on the grid. This isn't trivial
        # because of potential rotation of the telescope with respect to NS
        xind, yind, min_xsep, min_ysep = find_grid_indices(self.telescope.baselines)

        # Define several variables describing the baseline configuration.
        nx = np.abs(xind).max() + 1
        ny = 2 * np.abs(yind).max() + 1
        vis_pos_x = np.arange(nx) * min_xsep
        vis_pos_y = np.fft.fftfreq(ny, d=(1.0 / (ny * min_ysep)))

        # Extract the right ascension to initialise the new container with (or
        # calculate from timestamp)
        ra = (
            sstream.ra
            if "ra" in sstream.index_map
            else self.telescope.lsa(sstream.time)
        )

        # Create container for output
        grid = containers.VisGridStream(
            pol=pol, ew=vis_pos_x, ns=vis_pos_y, ra=ra, axes_from=sstream
        )

        # Redistribute over frequency
        sstream.redistribute("freq")
        grid.redistribute("freq")

        # Calculate the redundancy
        redundancy = tools.calculate_redundancy(
            sstream.input_flags[:],
            sstream.index_map["prod"][:],
            sstream.reverse_map["stack"]["stack"][:],
            sstream.vis.shape[1],
        )

        # De-reference distributed arrays outside loop to save repeated MPI calls
        ssv = sstream.vis[:]
        ssw = sstream.weight[:]
        gsv = grid.vis[:]
        gsw = grid.weight[:]
        gsr = grid.redundancy[:]

        gsv[:] = 0.0
        gsw[:] = 0.0

        # Unpack visibilities into new array
        for vis_ind, (p_ind, x_ind, y_ind) in enumerate(zip(pind, xind, yind)):

            # Different behavior for intracylinder and intercylinder baselines.
            gsv[p_ind, :, x_ind, y_ind, :] = ssv[:, vis_ind]
            gsw[p_ind, :, x_ind, y_ind, :] = ssw[:, vis_ind]
            gsr[p_ind, x_ind, y_ind, :] = redundancy[vis_ind]

            if x_ind == 0:
                pc_ind = pconjmap[p_ind]
                gsv[pc_ind, :, x_ind, -y_ind, :] = ssv[:, vis_ind].conj()
                gsw[pc_ind, :, x_ind, -y_ind, :] = ssw[:, vis_ind]
                gsr[pc_ind, x_ind, -y_ind, :] = redundancy[vis_ind]

        return grid


class BeamformNS(task.SingleTask):
    """A simple and quick map-maker that forms a series of beams on the meridian.

    This is designed to run on data after it has been collapsed down to
    non-redundant baselines only.

    Attributes
    ----------
    npix : int
        Number of map pixels in the declination dimension.  Default is 512.

    span : float
        Span of map in the declination dimension. Value of 1.0 generates a map
        that spans from horizon-to-horizon.  Default is 1.0.

    weight : string
        How to weight the non-redundant baselines:
            'inverse_variance' - each baseline weighted by the weight attribute
            'natural' - each baseline weighted by its redundancy (default)
            'uniform' - each baseline given equal weight
            'blackman' - use a Blackman window
            'nutall' - use a Blackman-Nutall window

    scaled : bool
        Scale the window to match the lowest frequency. This should make the
        beams more frequency independent.

    include_auto: bool
        Include autocorrelations in the calculation.  Default is False.
    """

    npix = config.Property(proptype=int, default=512)
    span = config.Property(proptype=float, default=1.0)
    weight = config.enum(
        ["uniform", "natural", "inverse_variance", "blackman", "nuttall"],
        default="natural",
    )
    scaled = config.Property(proptype=bool, default=False)
    include_auto = config.Property(proptype=bool, default=False)

    def process(self, gstream):
        """Computes the ringmap.

        Parameters
        ----------
        sstream : VisGridStream
            The input stream.

        Returns
        -------
        bf : HybridVisStream
        """

        # Redistribute over frequency
        gstream.redistribute("freq")

        gsv = gstream.vis[:]
        gsw = gstream.weight[:]
        gsr = gstream.redundancy[:]

        # Construct phase array
        el = self.span * np.linspace(-1.0, 1.0, self.npix)

        # Create empty ring map
        hv = containers.HybridVisStream(el=el, axes_from=gstream, attrs_from=gstream)
        hv.add_dataset("dirty_beam")
        hv.redistribute("freq")

        # Dereference datasets
        hvv = hv.vis[:]
        hvw = hv.weight[:]
        hvb = hv.dirty_beam[:]

        nspos = gstream.index_map["ns"][:]
        freq = gstream.index_map["freq"]["centre"]

        # Get the largest baseline present across all nodes while accounting for masking
        baselines_present = (
            np.moveaxis(gsw.view(np.ndarray), -2, 0).reshape(len(nspos), -1) > 0
        ).any(axis=1)
        nsmax_local = (
            np.abs(nspos[baselines_present]).max()
            if baselines_present.sum() > 0
            else 0.0
        )
        nsmax = self.comm.allreduce(nsmax_local, op=MPI.MAX)
        self.log.info(f"Maximum NS baseline is {nsmax:.2f}m")

        # Loop over local frequencies and fill ring map
        for lfi, fi in gstream.vis[:].enumerate(1):

            # Get the current frequency and wavelength
            fr = freq[fi]
            wv = scipy.constants.c * 1e-6 / fr

            vpos = nspos / wv

            if self.scaled:
                wvmin = scipy.constants.c * 1e-6 / freq.min()
                vmax = nsmax / wvmin
            else:
                vmax = nsmax / wv

            if self.weight == "inverse_variance":
                gw = gsw.local_array[:, lfi].copy()
            elif self.weight == "natural":
                gw = gsr[:].astype(np.float32)
            else:
                x = 0.5 * (vpos / vmax + 1)
                ns_weight = tools.window_generalised(x, window=self.weight)
                gw = (gsw[:, lfi] > 0) * ns_weight[
                    np.newaxis, np.newaxis, :, np.newaxis
                ]

            # Ensure we skip entries which are flagged out entirely
            gw *= gsw[:, lfi] > 0

            # Remove auto-correlations
            if not self.include_auto:
                gw[..., 0, 0, :] = 0.0

            # Normalize by sum of weights
            norm = np.sum(gw, axis=-2)
            gw *= tools.invert_no_zero(norm)[..., np.newaxis, :]

            # Create array that will be used for the inverse
            # discrete Fourier transform in y-direction
            phase = 2.0 * np.pi * nspos[np.newaxis, :] * el[:, np.newaxis] / wv
            F = np.exp(-1.0j * phase)

            # Calculate the hybrid visibilities
            gv = gsv[:, lfi].view(np.ndarray)
            hvv[:, lfi] = np.matmul(F, gw * gv)

            # Calculate the dirty beam
            hvb[:, lfi] = np.matmul(F, gw * np.ones_like(gv)).real

            # Estimate the weights assuming that the errors are all uncorrelated
            t = np.sum(tools.invert_no_zero(gsw[:, lfi]) * gw ** 2, axis=-2)
            hvw[:, lfi] = tools.invert_no_zero(t)

        return hv


class BeamformEW(task.SingleTask):
    """Final beam forming in the EW direction.

    This is designed to run on data after the NS beam forming.

    Attributes
    ----------
    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.  Default is False.
    single_beam: bool
        Only calculate the map for the central beam. Default is False.
    weight_ew : string
        How to weight the EW baselines? One of:
            'natural' - weight by the redundancy of the EW baselines.
            'uniform' - give each EW baseline uniform weight.
    """

    exclude_intracyl = config.Property(proptype=bool, default=False)
    single_beam = config.Property(proptype=bool, default=False)
    weight_ew = config.enum(["natural", "uniform"], default="natural")

    def process(self, hstream):
        """Computes the ringmap.

        Parameters
        ----------
        hstream : HybridVisStream
            The input stream.

        Returns
        -------
        rm : RingMap
        """

        # Redistribute over frequency
        hstream.redistribute("freq")

        # TODO: figure out how to get around this
        if len(hstream.index_map["pol"]) != 4:
            raise ValueError(
                "We need all 4 polarisation combinations for this to work."
            )

        # Create empty ring map
        n_ew = len(hstream.index_map["ew"])
        nbeam = 1 if self.single_beam else 2 * n_ew - 1

        # Determine the weighting coefficient in the EW direction
        if self.weight_ew == "uniform":
            weight_ew = np.ones(n_ew)
        else:  # self.weight_ew == "natural"
            weight_ew = n_ew - np.arange(n_ew)

        # Exclude the in cylinder baselines if requested (EW = 0)
        if self.exclude_intracyl:
            weight_ew[0] = 0.0

        # Factor to include negative elements in sum single beam sum
        if self.single_beam:
            weight_ew[1:] *= 2

        # Normalise the weights
        weight_ew = weight_ew / weight_ew.sum()

        # TODO: derive these from the actual polarisations found in the input
        pol = np.array(["XX", "reXY", "imXY", "YY"], dtype="U4")

        # Create ring map, copying over axes/attrs and add the optional datasets
        rm = containers.RingMap(
            beam=nbeam, pol=pol, axes_from=hstream, attrs_from=hstream
        )
        rm.add_dataset("rms")
        rm.add_dataset("dirty_beam")

        # Make sure ring map is distributed over frequency
        rm.redistribute("freq")

        # Dereference datasets
        hvv = hstream.vis[:]
        hvw = hstream.weight[:]
        hvb = hstream.dirty_beam[:]
        rmm = rm.map[:]
        rmb = rm.dirty_beam[:]

        # This matrix takes the linear combinations of polarisations required to rotate
        # from XY, YX basis into reXY and imXY
        P = np.array(
            [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, -0.5j, 0.5j, 0], [0, 0, 0, 1]]
        )

        # Loop over local frequencies and fill ring map
        for lfi, fi in hstream.vis[:].enumerate(axis=1):

            # Rotate the polarisations
            v = np.tensordot(P, hvv[:, lfi], axes=(1, 0))
            b = np.tensordot(P, hvb[:, lfi], axes=(1, 0))

            # Apply the EW weighting
            v *= weight_ew[np.newaxis, :, np.newaxis, np.newaxis]
            b *= weight_ew[np.newaxis, :, np.newaxis, np.newaxis]

            # Perform  inverse fast fourier transform in x-direction
            if self.single_beam:
                # Only need the 0th term of the irfft, equivalent to summing in
                # then EW direction
                beamformed_data = np.sum(v.real, axis=1)[:, np.newaxis]
                dirty_beam = np.sum(b, axis=1)[:, np.newaxis]
            else:
                beamformed_data = np.fft.irfft(v, nbeam, axis=1) * nbeam
                dirty_beam = np.fft.irfft(b, nbeam, axis=1) * nbeam

            # Save to container (shifting to the final axis ordering)
            rmm[:, :, lfi] = beamformed_data.transpose(1, 0, 3, 2)
            rmb[:, :, lfi] = dirty_beam.transpose(1, 0, 3, 2)

        # Estimate weights/rms noise in the ring map by propagating estimates of the
        # variance in the visibilities
        rm_var = (tools.invert_no_zero(hvw) * weight_ew[:, np.newaxis] ** 2).sum(axis=2)
        rm.rms[:] = rm_var ** 0.5
        rm.weight[:] = tools.invert_no_zero(rm_var[..., np.newaxis])

        return rm


class RingMapMaker(task.group_tasks(MakeVisGrid, BeamformNS, BeamformEW)):
    """Make a ringmap from the data."""


class DeconvolveHybridMBase(task.SingleTask):
    """Base class for deconvolving ringmapmakers (non-functional).

    Attributes
    ----------
    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.
    save_dirty_beam : bool
        Create a `dirty_beam` dataset in the output container that contains
        the synthesized beam in the EW direction at each declination.
    """

    exclude_intracyl = config.Property(proptype=bool, default=False)
    save_dirty_beam = config.Property(proptype=bool, default=False)

    def process(self, hybrid_vis_m: containers.HybridVisMModes) -> containers.RingMap:
        """Generate a deconvolved ringmap.

        Parameters
        ----------
        hybrid_vis_m : containers.HybridVisMModes
            M-mode transform of hybrid beamformed visibilities.

        Returns
        -------
        ringmap : containers.RingMap
            Deconvolved ring map.
        """
        # Distribute over frequency
        hybrid_vis_m.redistribute("freq")

        # Determine local frequencies
        nfreq = hybrid_vis_m.vis.local_shape[3]
        fstart = hybrid_vis_m.vis.local_offset[3]
        fstop = fstart + nfreq

        local_freq = hybrid_vis_m.freq[fstart:fstop]

        # Number of RA samples in the final output
        m = hybrid_vis_m.index_map["m"]
        mmax = hybrid_vis_m.mmax
        nra = 2 * (mmax + 1)

        # Create ring map, copying over axes/attrs
        rm = containers.RingMap(
            beam=1,
            ra=nra,
            axes_from=hybrid_vis_m,
            attrs_from=hybrid_vis_m,
            distributed=hybrid_vis_m.distributed,
            comm=hybrid_vis_m.comm,
        )
        if self.save_dirty_beam:
            rm.add_dataset("dirty_beam")

        rm.weight[:] = 0.0
        rm.redistribute("freq")

        # Add attributes describing the EW weighting scheme
        rm.attrs["exclude_intracyl"] = self.exclude_intracyl
        if hasattr(self, "weight_ew"):
            rm.attrs["weight_ew"] = self.weight_ew

        # Dereference datasets
        hv = hybrid_vis_m.vis[:].view(np.ndarray)
        hw = hybrid_vis_m.weight[:].view(np.ndarray)

        rmm = rm.map[:]
        rmw = rm.weight[:]
        if self.save_dirty_beam:
            rmb = rm.dirty_beam[:]

        # Loop over frequencies
        for lfi, freq in enumerate(local_freq):

            find = (slice(None),) * 3 + (lfi,)

            hvf = hv[find]

            # Make copy here because we will modify weights if exclude_intracyl is set
            inv_var = hw[find][..., np.newaxis].copy()

            # Get the EW weights using method defined by subclass
            weight = self._get_weight(inv_var) * (inv_var > 0.0)

            # Get the beam m-modes using method defined by subclass
            bvf = self._get_beam_mmodes(freq, hybrid_vis_m)

            # Get the regularisation term, exact prescription is defined by the subclass
            epsilon = self._get_regularisation(freq, m)

            # Calculate the normalization
            sum_weight = (weight * np.abs(bvf) ** 2).sum(axis=(1, -2))

            C_inv = epsilon + sum_weight

            # Solve for the sky m-modes
            map_m = (bvf.conj() * weight * hvf).sum(
                axis=(1, -2)
            ) * tools.invert_no_zero(C_inv)

            # Calculate the dirty beam m-modes
            dirty_beam_m = sum_weight * tools.invert_no_zero(C_inv)

            # Calculate the point source normalization (dirty beam at transit)
            norm = tools.invert_no_zero(dirty_beam_m.mean(axis=0))[:, np.newaxis, :]

            # Fill in the ringmap
            rmm[0, :, lfi] = (
                np.fft.irfft(map_m.transpose(1, 2, 0), axis=-1, n=nra).transpose(
                    0, 2, 1
                )
                * norm
            )

            # Fill in the dirty beam
            if self.save_dirty_beam:
                rmb[0, :, lfi] = (
                    np.fft.irfft(
                        dirty_beam_m.transpose(1, 2, 0), axis=-1, n=nra
                    ).transpose(0, 2, 1)
                    * norm
                )

            # Calculate the expected map noise by propagating the uncertainty on the m's
            var = tools.invert_no_zero(inv_var)
            var = ((weight * np.abs(bvf)) ** 2 * var).sum(
                axis=(1, -2)
            ) * tools.invert_no_zero(C_inv ** 2)
            sum_var_map_m = 0.5 * np.sum(var, axis=0)[:, np.newaxis, :]

            rmw[:, lfi] = (mmax + 1) ** 2 * tools.invert_no_zero(
                norm ** 2 * sum_var_map_m
            )

        return rm

    def _get_beam_mmodes(self, freq, hybrid_vis_m):
        """Return the m-mode transform of the beam model at a particular frequency.

        Any subclass must define this method in order to be a functional
        deconvolving ringmap maker.

        Parameters
        ----------
        freq : float
            The frequency in MHz.
        hybrid_vis_m : containers.HybridVisMModes
            The m-mode transform of the hybrid visiblities.

        Returns
        -------
        hybrid_beam_m : np.ndarray[nm, nmsign, npol, new, nel]
            The m-mode transform of the beam model at the requested frequency.
        """
        raise NotImplementedError(
            f"{self.__class__} must define a _get_beam_mmodes method."
        )

    def _get_weight(self, inv_var):
        """Return the weight to be used when averaging over EW baselines.

        Any subclass must define this method in order to be a functional
        deconvolving ringmap maker.

        Parameters
        ----------
        inv_var : np.ndarray[nm, nmsign, npol, new, nel]
            The inverse variance of the noise in the m-mode transform
            of the hybrid visibilities.

        Returns
        -------
        weight :  np.ndarray[nm, nmsign, npol, new, nel] (or can broadcast against)
            The weight given to each EW baseline.
        """
        raise NotImplementedError(f"{self.__class__} must define a _get_weight method.")

    def _get_regularisation(self, freq, m):
        """Return the parameter used to regularize the deconvolution operation.

        Any subclass must define this method in order to be a functional
        deconvolving ringmap maker.

        Parameters
        ----------
        freq : float
            The frequency in MHz.
        m : np.ndarray[nm,]
            The m-modes.

        Returns
        -------
        epsilon : np.ndarray[nm, npol, nel] (or can broadcast against)
            The regularisation parameter that appears in the denominator of
            the deconvolution equation.
        """
        raise NotImplementedError(
            f"{self.__class__} must define a _get_regularisation method."
        )


class DeconvolveAnalyticalBeam(DeconvolveHybridMBase):
    """Base class for deconvolving the driftscan model of the beam (non-functional)."""

    telescope = None

    def setup(self, telescope: io.TelescopeConvertible):
        """Set the telescope object.

        Parameters
        ----------
        manager
            The telescope object to use.
        """

        self.telescope = io.get_telescope(telescope)

    def _get_beam_mmodes(self, freq, hybrid_vis_m):

        # NOTE: Coefficients taken from Mateus's fits, but adjust to fix the definition
        # of sigma, and be the widths for the "voltage" beam
        def sig_chime_X(freq, dec):
            """EW voltage beam widths (in sigma) for CHIME X pol."""
            return 14.87857614 / freq / np.cos(dec)

        def sig_chime_Y(freq, dec):
            """EW voltage beam widths (in sigma) for CHIME Y pol."""
            return 9.95746878 / freq / np.cos(dec)

        beam_width = {"X": sig_chime_X, "Y": sig_chime_Y}

        def A(phi, sigma):
            """A Gaussian like function on the circle."""
            return np.exp(-((2 * np.tan(phi / 2)) ** 2) / (2 * sigma ** 2))

        def B(phi, u, sigma):
            """Azimuthal beam transfer function."""
            return np.exp(2.0j * np.pi * u * np.sin(phi)) * A(phi, sigma)

        # Deteremine the RA axis from the maximum m-mode in the hybrid visibilities
        nra = 2 * (hybrid_vis_m.mmax + 1)
        ra = np.linspace(0.0, 360.0, nra, endpoint=False)
        phi_arr = np.radians(ra)[np.newaxis, np.newaxis, np.newaxis, :]

        # Calculate the baseline distance in wavelengths
        wv = scipy.constants.c * 1e-6 / freq
        u = hybrid_vis_m.index_map["ew"] / wv

        # Calculate the projected baseline distance
        dec = np.arcsin(hybrid_vis_m.index_map["el"]) + np.radians(
            self.telescope.latitude
        )
        u_dec = u[:, np.newaxis] * np.cos(dec)[np.newaxis, :]
        u_arr = u_dec[np.newaxis, :, :, np.newaxis]

        # Construct an array containing the width of the beam for
        # each polarisation and declination
        pol = hybrid_vis_m.index_map["pol"]
        sig = np.zeros((pol.size, dec.size), dtype=dec.dtype)
        for pi, (pa, pb) in enumerate(pol):

            # Get the effective beamwidth for the polarisation combination
            sig_a = beam_width[pa](freq, dec)
            sig_b = beam_width[pb](freq, dec)
            sig[pi] = sig_a * sig_b / (sig_a ** 2 + sig_b ** 2) ** 0.5

        sig_arr = sig[:, np.newaxis, :, np.newaxis]

        # Calculate the effective beam transfer function
        B_arr = B(phi_arr, u_arr, sig_arr)

        mB = transform._make_marray(B_arr.conj(), mmax=hybrid_vis_m.mmax)

        return mB


class DeconvolveExternalBeam(DeconvolveHybridMBase):
    """Base class for deconvolving an external model of the beam (non-functional)."""

    def process(
        self,
        hybrid_vis_m: containers.HybridVisMModes,
        hybrid_beam_m: containers.HybridVisMModes,
    ) -> containers.RingMap:
        """Generate a deconvolved ringmap using an external beam model.

        Parameters
        ----------
        hybrid_vis_m : containers.HybridVisMModes
            M-mode transform of hybrid beamformed visibilities.

        hybrid_beam_m : containers.HybridVisMModes
            M-mode transform of the beam after converting into
            hybrid beamformed visibilities.

        Returns
        -------
        ringmap
            The deconvolved ring map.
        """
        # Prepare the external beam m-modes and save to class attribute
        hybrid_beam_m.redistribute("freq")

        fstart = hybrid_beam_m.vis.local_offset[3]
        fstop = fstart + hybrid_beam_m.vis.local_shape[3]

        self.beam_freq = hybrid_beam_m.freq[fstart:fstop]
        self.beam_mmodes = hybrid_beam_m.vis[:].view(np.ndarray)

        return super(DeconvolveExternalBeam, self).process(hybrid_vis_m)

    def _get_beam_mmodes(self, freq, hybrid_vis_m):

        ifreq = np.argmin(np.abs(freq - self.beam_freq))

        if np.abs(freq - self.beam_freq[ifreq]) > 0.0:
            raise RuntimeError("Frequency axis of the beam m-modes does not match.")

        return self.beam_mmodes[:, :, :, ifreq]


class TikhonovRingMapMaker(DeconvolveHybridMBase):
    """Base class for making maps using a Tikhonov regularisation scheme (non-functional).

    Attributes
    ----------
    weight_ew : string
        How to weight the EW baselines? One of:
            'natural' - weight by the redundancy of the EW baselines.
            'uniform' - give each EW baseline uniform weight.
            'inverse_variance' - weight by the expected inverse noise variance
                                 saved to the `weight` dataset.

    inv_SN:
        Regularisation parameter.
    """

    weight_ew = config.enum(
        ["natural", "uniform", "inverse_variance"], default="natural"
    )
    inv_SN = config.Property(proptype=float, default=1e-6)

    def _get_weight(self, inv_var):

        if self.weight_ew == "inverse_variance":
            weight_ew = inv_var

        else:
            n_ew = inv_var.shape[-2]

            if self.weight_ew == "uniform":
                weight_ew = np.ones(n_ew)
            else:  # self.weight_ew == "natural"
                weight_ew = n_ew - np.arange(n_ew)

            expand = [None] * inv_var.ndim
            expand[-2] = slice(None)
            weight_ew = weight_ew[tuple(expand)]

        if self.exclude_intracyl:
            weight_ew[..., 0, :] = 0.0

        weight_ew = weight_ew * tools.invert_no_zero(
            np.sum(weight_ew, axis=-2, keepdims=True)
        )

        return weight_ew

    def _get_regularisation(self, *args):

        return self.inv_SN


class WienerRingMapMaker(DeconvolveHybridMBase):
    """Base class for map making using a Wiener regularisation scheme (non-functional).

    Compared to TikhonovRingMapMaker, this task has a frequency and m-mode
    dependent regularisation parameter given by the ratio of the noise spectrum
    to the expected signal spectrum. The noise spectrum is obtained from the
    `weight` dataset and the signal spectrum is obtained from a power-law model
    for the extragalactic point source and diffuse galactic synchrotron emission
    whose parameters can be changed by the user.

    .. math::
        |V_{gal}| = a_{gal} \left(\frac{\nu}{\nu_{0}\right)^{\alpha_{gal}} m^{\beta_{gal}}
        |V_{psrc}| = a_{psrc} \left(\frac{\nu}{\nu_{0}\right)^{\alpha_{psrc}}
        S = |V_{gal}|^2 + |V_{psrc}|^2

    Attributes
    ----------
    gal_amp : float
        Prior for the amplitude of the m-mode transform of the
        diffuse galactic synchrotron emission.
    gal_alpha : float
        Prior for the power-law exponent describing the frequency dependence
        of the m-mode transform of the diffuse galactic synchrotron emission.
    gal_beta : float
        Prior for the power-law exponent describing the m dependence of
        of the m-mode transform of the diffuse galactic synchrotron emission.
    psrc_amp : float
        Prior for the amplitude of the m-mode transform of the
        extra-galactic point source emission.
    psrc_alpha : float
        Prior for the power-law exponent describing the frequency dependence
        of the m-mode transform of the extra-galactic point source emission.
    """

    gal_amp = config.Property(proptype=float, default=1.41)
    gal_alpha = config.Property(proptype=float, default=-1.75)
    gal_beta = config.Property(proptype=float, default=-0.75)

    psrc_amp = config.Property(proptype=float, default=0.045)
    psrc_alpha = config.Property(proptype=float, default=-1.0)

    pivot_freq = 600.0
    weight_ew = "inverse_variance"

    def _get_regularisation(self, freq, m, *args):

        gal = (
            self.gal_amp
            * (freq / self.pivot_freq) ** self.gal_alpha
            * np.where(m > 0.0, m, 1.0) ** self.gal_beta
        )
        psrc = self.psrc_amp * (freq / self.pivot_freq) ** self.psrc_alpha

        spectrum = gal ** 2 + psrc ** 2

        # Expand the array so that it can be broadcast against an
        # array of shape (nm, npol, nel)
        return tools.invert_no_zero(spectrum[:, np.newaxis, np.newaxis])

    def _get_weight(self, inv_var):

        weight_ew = inv_var
        if self.exclude_intracyl:
            weight_ew[..., 0, :] = 0.0

        return weight_ew


class TikhonovRingMapMakerAnalytical(DeconvolveAnalyticalBeam, TikhonovRingMapMaker):
    """Make a ringmap using Tikhonov deconvolution of an analytical beam model."""


class TikhonovRingMapMakerExternal(DeconvolveExternalBeam, TikhonovRingMapMaker):
    """Make a ringmap using Tikhonov deconvolution of an external beam model."""


class WienerRingMapMakerAnalytical(DeconvolveAnalyticalBeam, WienerRingMapMaker):
    """Make a ringmap using Wiener deconvolution of an analytical beam model."""


class WienerRingMapMakerExternal(DeconvolveExternalBeam, WienerRingMapMaker):
    """Make a ringmap using Wiener deconvolution of an external beam model."""


class RADependentWeights(task.SingleTask):
    """Re-establish an RA dependence to the `weight` dataset of a deconvolved ringmap.

    This RA dependence was lost in the round-trip m-mode transform.
    """

    def process(
        self, hybrid_vis: containers.HybridVisStream, ringmap: containers.RingMap
    ) -> containers.RingMap:
        """Scale the ringmap weights by the RA dependence of the hybrid visibility weights.

        Parameters
        ----------
        hybrid_vis : containers.HybridVisStream
            Hybrid beamformed visibilities before deconvolution.
            Used to measure the RA dependence of the `weight` dataset.
        ringmap : containers.RingMap
            RingMap after deconvolution.

        Returns
        -------
        ringmap : containers.RingMap
            The input ringmap container with the `weight` dataset scaled
            by an RA dependent factor determined from hybrid_vis.
        """

        # Determine how the EW baselines were averaged in the ringmap maker
        exclude_intracyl = ringmap.attrs.get("exclude_intracyl", None)
        weight_scheme = ringmap.attrs.get("weight_ew", None)

        if (exclude_intracyl is None) or (weight_scheme is None):
            msg = (
                "The ring map maker must save `weight_ew` and `exclude_intracyl` "
                "config parameters to the container attributes in order to "
                "reconstruct the RA dependence of the noise."
            )
            raise RuntimeError(msg)

        # Extract the variance of the hybrid visibilities from the weight dataset
        var = tools.invert_no_zero(hybrid_vis.weight[:].view(np.ndarray))

        # Calculate the time averaged variance
        var_time_avg = np.mean(var, axis=-1, keepdims=True)

        # Determine the weights that where used to average over the EW baselines
        if weight_scheme == "inverse_variance":

            weight_ew = tools.invert_no_zero(var_time_avg)

        else:

            n_ew = var.shape[-2]

            if weight_scheme == "uniform":
                weight_ew = np.ones(n_ew)
            else:  # weight_scheme == "natural"
                weight_ew = n_ew - np.arange(n_ew)

            expand = [None] * var.ndim
            expand[-2] = slice(None)
            weight_ew = weight_ew[tuple(expand)]

        if exclude_intracyl:
            weight_ew[..., 0, :] = 0.0

        # Use the baseline averaged variance divided by the baseline averaged,
        # time averaged variance as an approximatation for the RA dependence of
        # the noise variance in the deconvolved ringmap.  Note that this is
        # inverted in the equation below since we want to scale the weights.
        ra_dependence = np.sum(
            weight_ew ** 2 * var_time_avg, axis=-2
        ) * tools.invert_no_zero(np.sum(weight_ew ** 2 * var, axis=-2))

        # Scale the ringmap weights by the RA dependence
        ringmap.weight[:] *= ra_dependence[..., np.newaxis]

        return ringmap


def find_basis(baselines):
    """Find the basis unit vectors of the grid of baselines.

    Parameters
    ----------
    baselines : np.ndarray[nbase, 2]
        X and Y displacements of the baselines.

    Returns
    -------
    xhat, yhat : np.ndarray[2]
        Unit vectors pointing in the mostly X and mostly Y directions of the grid.
    """

    # Find the shortest baseline, this should give one of the axes
    bl = np.abs(baselines)
    bl[bl == 0] = 1e30
    ind = np.argmin(bl)

    # Determine the basis vectors and label them xhat and yhat
    e1 = baselines[ind]
    e2 = np.array([e1[1], -e1[0]])

    xh, yh = (e1, e2) if abs(e1[0]) > abs(e2[0]) else (e2, e1)

    xh = xh / np.dot(xh, xh) ** 0.5 * np.sign(xh[0])
    yh = yh / np.dot(yh, yh) ** 0.5 * np.sign(yh[1])

    return xh, yh


def find_grid_indices(baselines):
    """Find the indices of each baseline in the grid, and the spacing.

    Parameters
    ----------
    baselines : np.ndarray[nbase, 2]
        X and Y displacements of the baselines.

    Returns
    -------
    xind, yind : np.ndarray[nbase]
        Indices of the baselines in the grid.
    dx, dy : float
        Spacing of the grid in each direction.
    """

    def _get_inds(s):
        s_abs = np.abs(s)
        d = s_abs[s_abs > 1e-4].min()
        return np.rint(s / d).astype(np.int), d

    xh, yh = find_basis(baselines)

    xind, dx = _get_inds(np.dot(baselines, xh))
    yind, dy = _get_inds(np.dot(baselines, yh))

    return xind, yind, dx, dy

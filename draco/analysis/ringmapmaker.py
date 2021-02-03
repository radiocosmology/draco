"""
========================================================
Map making tasks (:mod:`~draco.analysis.beamform`)
========================================================

.. currentmodule:: draco.analysis.beamform

Tools for beamforming data for arrays with a cartesian layout.

Tasks
=====

.. autosummary::
    :toctree: generated/

    MakeVisGrid
    BeamformNS
"""
import numpy as np
import scipy.constants

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
        hv.redistribute("freq")

        # Dereference datasets
        hvv = hv.vis[:]
        hvw = hv.weight[:]
        hvb = hv.dirty_beam[:]

        nspos = gstream.index_map["ns"][:]
        nsmax = np.abs(nspos).max()
        freq = gstream.index_map["freq"]["centre"]

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

        # Create ring map, copying over axes/attrs and add the optional datasets
        rm = containers.RingMap(beam=nbeam, axes_from=hstream, attrs_from=hstream)
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
        rm.weight[:] = tools.invert_no_zero(rm_var)

        return rm


class RingMapMaker(task.group_tasks(MakeVisGrid, BeamformNS, BeamformEW)):
    """Make a ringmap from the data."""


class DeconvolveHybridM(task.SingleTask):
    """Apply a Wiener deconvolution to calculate a ring map from hybrid visibility data.

    Attributes
    ----------
    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.  Default is False.
    weight_ew : string
        How to weight the EW baselines? One of:
            'natural' - weight by the redundancy of the EW baselines.
            'uniform' - give each EW baseline uniform weight.
    inv_SN : float
        Inverse signal to noise ratio, assuming identity matrices for both. This is
        equivalent to a Tikhonov regularisation.
    """

    exclude_intracyl = config.Property(proptype=bool, default=False)
    weight_ew = config.enum(["natural", "uniform"], default="natural")
    inv_SN = config.Property(proptype=float, default=1e-6)

    telescope = None

    def setup(self, telescope: io.TelescopeConvertible):
        """Set the telescope object.

        Parameters
        ----------
        manager
            The telescope object to use.
        """

        self.telescope = io.get_telescope(telescope)

    def process(self, hybrid_vis_m: containers.HybridVisMModes) -> containers.RingMap:
        """Generate a deconvolved ringmap.

        Parameters
        ----------
        hybrid_vis_m
            The hybrid beamformed m-mode visibilities to use.

        Returns
        -------
        ringmap
            The deconvolved ring map.
        """

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

        sza = hybrid_vis_m.index_map["el"]
        # TODO: I think the sign is incorrect here
        dec = np.arcsin(sza) + np.radians(self.telescope.latitude)

        # Create empty ring map
        d_ew = hybrid_vis_m.index_map["ew"]
        n_ew = len(d_ew)

        # Determine the weighting coefficient in the EW direction
        if self.weight_ew == "uniform":
            weight_ew = np.ones(n_ew)
        else:  # self.weight_ew == "natural"
            weight_ew = n_ew - np.arange(n_ew)

        # Exclude the in cylinder baselines if requested (EW = 0)
        if self.exclude_intracyl:
            weight_ew[0] = 0.0

        # Normalise the weights
        weight_ew = weight_ew / weight_ew.sum()

        # Number of RA samples in the final output
        nra = 2 * hybrid_vis_m.mmax

        # Create ring map, copying over axes/attrs and add the optional datasets
        rm = containers.RingMap(
            beam=1, ra=nra, axes_from=hybrid_vis_m, attrs_from=hybrid_vis_m
        )
        rm.weight[:] = 0.0

        # This matrix takes the linear combinations of polarisations required to rotate
        # from XY, YX basis into reXY and imXY
        # P = np.array(
        #     [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, -0.5j, 0.5j, 0], [0, 0, 0, 1]]
        # )

        # Make sure ring map is distributed over frequency
        hybrid_vis_m.redistribute("freq")
        rm.redistribute("freq")

        # Dereference datasets
        hv = hybrid_vis_m.vis[:]
        hw = hybrid_vis_m.weight[:]
        rmm = rm.map[:]
        rmw = rm.weight[:]

        phi_arr = np.radians(rm.ra)[np.newaxis, np.newaxis, :]

        pol_map = hybrid_vis_m.index_map["pol"]

        # Loop over frequencies
        for lfi, fi in hv.enumerate(3):

            freq = hybrid_vis_m.freq[fi]
            wv = scipy.constants.c * 1e-6 / freq

            u = d_ew / wv
            u_dec = u[:, np.newaxis] * np.cos(dec)[np.newaxis, :]
            u_arr = u_dec[:, :, np.newaxis]

            map_m = np.zeros(
                (hybrid_vis_m.mmax + 1, len(pol_map), len(dec)), dtype=np.complex128
            )

            # TODO: the handling of the polarisations in here is probably very messed
            # up.
            for pi, (pa, pb) in enumerate(pol_map):

                # Get the effective beamwidth for the polarisation combination
                sig_a = beam_width[pa](freq, dec)
                sig_b = beam_width[pb](freq, dec)
                sig = sig_a * sig_b / (sig_a ** 2 + sig_b ** 2) ** 0.5
                sig_arr = sig[np.newaxis, :, np.newaxis]

                # Calculate the effective beam transfer function at each declination
                # and normalise
                B_arr = B(phi_arr, u_arr, sig_arr)
                B_norm = B_arr
                # B_norm = B_arr / np.sum(np.abs(B_arr), axis=-1)[:, :, np.newaxis]

                # Get the transfer function in m-space
                mB = transform._make_marray(B_norm.conj(), mmax=hybrid_vis_m.mmax)

                C_inv = self.inv_SN + (mB.conj() * weight_ew[:, np.newaxis] * mB).sum(
                    axis=(1, -2)
                )

                dirty_m = (hv[:, :, pi, lfi] * weight_ew[:, np.newaxis] * mB).sum(
                    axis=(1, -2)
                )

                map_m[:, pi] = dirty_m / C_inv

                # TODO: do a more intelligent setting of the weights. At the very least
                # we should get the scale right
                rmw[pi, lfi] = hw[:, :, pi, lfi].mean()

            # Fourier transform back to the deconvolved RA, and insert into the ringmap
            rmm[0, :, lfi] = np.fft.irfft(
                map_m.transpose(1, 2, 0), axis=-1, n=nra
            ).transpose(0, 2, 1)

            # Rotate the polarisations
            # v = np.tensordot(P, hvv[:, lfi], axes=(1, 0))

        return rm


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

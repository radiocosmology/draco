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

        if (sstream.prodstack != self.telescope.uniquepairs).all():
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
        ny = np.abs(yind).max() + 1
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
        default="uniform",
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
                gw = gsw[:, lfi].copy()
            elif self.weight == "natural":
                gw = gsr[:].copy()
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
            hvv[:, lfi] = np.dot(F, gw * gsv[:, lfi]).transpose(1, 2, 0, 3)

            # Calculate the dirty beam
            hvb[:, lfi] = np.dot(F, gw * np.one_like(gsv[:, lfi])).transpose(1, 2, 0, 3)

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
    """

    exclude_intracyl = config.Property(proptype=bool, default=False)
    single_beam = config.Property(proptype=bool, default=False)

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

        # Create ring map, copying over axes/attrs and add the optional datasets
        rm = containers.RingMap(beam=nbeam, axes_from=hstream, attrs_from=hstream)
        rm.add_dataset("rms")
        rm.add_dataset("dirty_beam")

        # Make sure ring map is distributed over frequency
        rm.redistribute("freq")

        # Estimate rms noise in the ring map by propagating estimates
        # of the variance in the visibilities
        # TODO: figure out what this should be
        # rm.rms[:] = np.sqrt(np.sum(np.dot(coeff,
        #             tools.invert_no_zero(invvar) * weight**2.0), axis=-1))
        rm.rms[:] = 0.0

        # Dereference datasets
        hvv = hstream.vis[:]
        hvw = hstream.weight[:]
        rmm = rm.map[:]
        rmb = rm.dirty_beam[:]

        # Loop over local frequencies and fill ring map
        for lfi, fi in hstream.vis[:].enumerate(axis=1):

            # TODO: make m a broadcastable rather than full size array
            v = hvv[:, lfi]
            w = hvw[:, lfi]
            m = np.ones_like(w)

            # Exclude the in cylinder baselines if requested (EW = 0)
            if self.exclude_intracyl:
                m[:, 0] = 0.0

            # Perform  inverse fast fourier transform in x-direction
            if self.single_beam:
                # Only need the 0th term of the irfft, equivalent to summing in
                # then EW direction
                m[:, 1:] *= 2.0  # Factor to include negative elements in sum below
                bfm = np.asarray(np.sum(v * m, axis=1)).real[:, np.newaxis, ...]
                sb = np.asarray(np.sum(w * m, axis=1)).real[:, np.newaxis, ...]
            else:
                bfm = np.fft.irfft(v * m, nbeam, axis=1) * nbeam
                sb = np.fft.irfft(w * m, nbeam, axis=1) * nbeam

            # Save to container (shifting to the final axis ordering)
            rmm[:, :, lfi] = bfm.transpose(1, 0, 3, 2)
            rmb[:, :, lfi] = sb.transpose(1, 0, 3, 2)

        return rm


class RingMapMaker(task.group_tasks(MakeVisGrid, BeamformNS, BeamformEW)):
    """Make a ringmap from the data."""

    pass


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
        return np.rint(s / d), d

    xh, yh = find_basis(baselines)

    xind, dx = _get_inds(np.dot(baselines, xh))
    yind, dy = _get_inds(np.dot(baselines, yh))

    return xind, yind, dx, dy

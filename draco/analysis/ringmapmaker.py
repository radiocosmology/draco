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
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
import scipy.constants

from caput import config

from ..core import task
from ..core import io
from ..util import tools
from ..core import containers


class MakeVisGrid(task.SingleTask):
    """Arrange the visibilities onto a 2D grid."""

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

        # Redistribute over frequency
        sstream.redistribute("freq")

        # Extract the right ascension (or calculate from timestamp)
        ra = (
            sstream.ra
            if "ra" in sstream.index_map
            else self.telescope.lsa(sstream.time)
        )

        # Construct mapping from vis array to unpacked 2D grid
        nprod = sstream.prod.shape[0]
        pind = np.zeros(nprod, dtype=np.int)
        xind = np.zeros(nprod, dtype=np.int)
        ysep = np.zeros(nprod, dtype=np.float)
        xsep = np.zeros(nprod, dtype=np.float)

        for pp, (ii, jj) in enumerate(sstream.prod):

            if self.telescope.feedconj[ii, jj]:
                ii, jj = jj, ii

            # fi = self.telescope.feeds[ii]
            # fj = self.telescope.feeds[jj]

            pind[pp] = 2 * self.telescope.beamclass[ii] + self.telescope.beamclass[jj]
            # pind[pp] = 2 * int(fi.pol == 'S') + int(fj.pol == 'S')
            # xind[pp] = np.abs(fi.cyl - fj.cyl)
            ysep[pp] = self.telescope.baselines[pp, 1]
            xsep[pp] = self.telescope.baselines[pp, 0]
            # xsep[pp] = fi.pos[0] - fj.pos[0]

        abs_ysep = np.abs(ysep)
        abs_xsep = np.abs(xsep)
        min_ysep, max_ysep = np.percentile(abs_ysep[abs_ysep > 0.0], [0, 100])
        min_xsep, max_xsep = np.percentile(abs_xsep[abs_xsep > 0.0], [0, 100])

        yind = np.round(ysep / min_ysep).astype(np.int)
        xind = np.round(xsep / min_xsep).astype(np.int)

        grid_index = list(zip(pind, xind, yind))

        # Define several variables describing the baseline configuration.
        nfeed = int(np.round(max_ysep / min_ysep)) + 1
        nvis_1d = 2 * nfeed - 1
        ncyl = np.max(xind) + 1

        # Define polarisation axis
        pol = np.array([x + y for x in ["X", "Y"] for y in ["X", "Y"]])
        vis_pos_ns = np.fft.fftfreq(nvis_1d, d=(1.0 / (nvis_1d * min_ysep)))
        vis_pos_ew = np.arange(ncyl) * min_xsep

        # Create container for output
        grid = containers.VisGridStream(
            pol=pol, ew=vis_pos_ew, ns=vis_pos_ns, ra=ra, axes_from=sstream
        )

        # De-reference distributed arrays outside loop to save repeated MPI calls
        ssv = sstream.vis[:]
        ssw = sstream.weight[:]
        gsv = grid.vis[:]
        gsw = grid.weight[:]

        gsv[:] = 0.0
        gsw[:] = 0.0

        # Unpack visibilities into new array
        for vis_ind, (p_ind, x_ind, y_ind) in enumerate(grid_index):

            # Different behavior for intracylinder and intercylinder baselines.
            gsv[p_ind, :, x_ind, y_ind, :] = ssv[:, vis_ind]
            gsw[p_ind, :, x_ind, y_ind, :] = ssw[:, vis_ind]

            if x_ind == 0:
                gsv[p_ind, :, x_ind, -y_ind, :] = ssv[:, vis_ind].conj()
                gsw[p_ind, :, x_ind, -y_ind, :] = ssw[:, vis_ind]

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
        gsw = gstream.weight[:].copy()

        # Remove auto-correlations
        if not self.include_auto:
            gsw[..., 0, 0] = 0.0

        # Construct phase array
        el = self.span * np.linspace(-1.0, 1.0, self.npix)

        # Create empty ring map
        hv = containers.HybridVisStream(el=el, axes_from=gstream, attrs_from=gstream)
        hv.redistribute("freq")

        # Dereference datasets
        hvv = hv.vis[:]
        hvw = hv.weight[:]

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

            x = 0.5 * (vpos / vmax + 1)
            ns_weight = tools.window_generalised(x, window=self.weight)

            # Create array that will be used for the inverse
            # discrete Fourier transform in y-direction
            phase = 2.0 * np.pi * nspos[np.newaxis, :] * el[:, np.newaxis] / wv
            F = np.exp(-1.0j * phase) * ns_weight[np.newaxis, :]

            # Calculate the hybrid visibilities
            hvv[:, lfi] = np.dot(F, gsv[:, lfi]).transpose(1, 2, 0, 3)

            # Estimate the weights assuming that the errors are all uncorrelated
            t = np.dot(F, tools.invert_no_zero(gsw[:, lfi]) ** 0.5).transpose(
                1, 2, 0, 3
            )
            hvw[:, lfi] = tools.invert_no_zero(t.real) ** 2

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
                m[:, :, 1:] *= 2.0  # Factor to include negative elements in sum below
                bfm = np.sum(v * m, axis=1).real[:, :, np.newaxis, ...]
                sb = np.sum(w * m, axis=1).real[:, :, np.newaxis, ...]
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

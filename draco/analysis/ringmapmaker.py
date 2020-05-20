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
from caput.pipeline import PipelineRuntimeError, PipelineConfigError
from cora.util import coord

from .transform import _make_marray
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

            if self.weight == "inverse_variance":
                gw = gsw[:, lfi]
            elif self.weight == "natural":
                raise NotImplementedError(
                    "Natural weighting hasn't been implemented yet."
                )
            else:
                x = 0.5 * (vpos / vmax + 1)
                ns_weight = tools.window_generalised(x, window=self.weight)
                gw = (gsw[:, lfi] > 0) * ns_weight[
                    np.newaxis, np.newaxis, :, np.newaxis
                ]

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

            # Estimate the weights assuming that the errors are all uncorrelated
            t = np.sum(tools.invert_no_zero(gsw[:, lfi]) * gw ** 2, axis=-2)
            hvw[:, lfi] = tools.invert_no_zero(t[..., np.newaxis, :])

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


class UnbeamformNS(task.SingleTask):
    """Attempt to undo the beamforming in the NS direction.

    Inverts the beamforming matrix and the apodisation window applied by `BeamformNS`
    The weights cannot be restored since we do not track the full covariances.

    Attributes
    ----------

    invert_weight : bool (default False)
        Whether to invert the weighting that was applied to the visibilities.

    weight : string (default 'uniform')
        NS baseline weighting that was used for the beamforming, to be undone.
        See `BeamformNS` for options.

    scaled : bool (default False)
        Whether the beamforming used a window scaled to the longest wavelength.

    pinv : bool (default True)
        Calculate the pseudo inverse of the forward Fourier matrix to invert the
        beamforming. Otherwise just use the conjugate transpose.

    mask : bool (default False)
        Include a massk that was applied to the data in the pseudo-inverse.
        Should be used together with `pinv` only.
    """

    pinv = config.Property(proptype=bool, default=False)
    weight = config.enum(
        ["uniform", "natural", "inverse_variance", "blackman", "nuttall"],
        default="uniform",
    )
    invert_weight = config.Property(proptype=bool, default=True)
    scaled = config.Property(proptype=bool, default=False)
    mask = config.Property(proptype=bool, default=False)

    def setup(self, tel):
        """Get physical array properties.

        Parameters
        ----------

        tel : TransitTelescope
            Telescope.
        """

        # get telescope
        self.tel = io.get_telescope(tel)

        # get the baseline grid properties
        ypos = self.tel.baselines[:, 1]
        self.ysep = np.abs(ypos)[ypos != 0.0].min()
        self.ny = int(np.round(2 * np.abs(ypos).max() / self.ysep)) + 1
        self.log.debug(
            "Got telescope with {:d} NS grid points with separations {:.3f}".format(
                self.ny, self.ysep
            )
        )

    def process(self, bf, mask):
        """Apply the inverse transform.

        Parameters
        ----------

        bf : containers.HybridVisStream or containers.HybridVisMModes
            The beamformed data.

        mask : containers.HybridVisStream or containers.HybridVisMModes
            Mask that was applied to the data to be included in the
            pseudo-inverse calculation. Only used if parameter `mask=True`.

        Returns
        -------

        vg : containers.VisGridStream or containers.VisGridMModes
            The gridded visibility data.
        """

        bf.redistribute("freq")

        el = bf.index_map["el"]
        freq = bf.index_map["freq"]["centre"]

        # reproduce baseline ordering from BeaformNS
        if self.ny is None:
            self.ny = el.shape[0]
        if self.ny % 2 == 0:
            self.ny += 1
        ns = np.fft.fftfreq(self.ny, d=(1.0 / (self.ny * self.ysep)))
        nsmax = np.abs(ns).max()

        # Get the output container and figure out at which position is it's
        # frequency axis
        contmap = {
            containers.HybridVisStream: containers.VisGridStream,
            containers.HybridVisMModes: containers.VisGridMModes,
        }
        out_cont = contmap[bf.__class__]
        freq_axis = out_cont._dataset_spec["vis"]["axes"].index("freq")
        is_m = bf.__class__ is containers.HybridVisMModes

        # create visibility grid container
        vg = out_cont(axes_from=bf, attrs_from=bf, ns=ns, comm=bf.comm)
        vg.redistribute("freq")

        # Dereference datasets
        bfv = bf.vis[:]
        bfw = bf.weight[:]
        vgv = vg.vis[:]
        vgw = vg.weight[:]
        if self.mask:
            bfm = mask.vis[:]

        # Iterate over local frequencies
        for lfi, fi in bf.vis[:].enumerate(freq_axis):

            # scale baselines in units of wavelength
            wv = scipy.constants.c * 1e-6 / freq[fi]
            vpos = ns / wv

            # reconstruct NS weights applied in forward transform
            if self.weight == "inverse_variance":
                raise NotImplementedError(
                    "Inverse variance weighting cannot be inverted."
                )
            elif self.weight == "natural":
                raise NotImplementedError(
                    "Natural weighting hasn't been implemented yet."
                )
            else:
                # evaluate weights over specified span
                wvmin = scipy.constants.c * 1e-6 / freq.min()
                vmax = nsmax / wvmin if self.scaled else nsmax / wv
                x = 0.5 * (vpos / vmax + 1)
                ns_weight = tools.window_generalised(x, window=self.weight)
                # normalize
                ns_weight *= tools.invert_no_zero(ns_weight.sum())
                ns_weight[
                    np.isclose(ns_weight, 0.0, rtol=np.finfo(bf.vis.dtype).resolution)
                ] = 0.0

            # make phase weights
            # TODO: makes more sense to invert order and do trasnpose in the FT case
            phase = 2.0 * np.pi * ns[:, np.newaxis] * el[np.newaxis, :] / wv
            F = np.exp(1.0j * phase)

            # treat m-mode and sstream cases separately
            if not is_m:
                if self.mask:
                    raise NotImplementedError(
                        "Mask is only supported for HybridVisMModes."
                    )
                if self.pinv:
                    try:
                        F = np.linalg.pinv(np.conj(F.T))
                    except:
                        self.log.warning(
                            "Pseudo-inverse failed for frequency {:.2f} MHz.".format(
                                freq[fi]
                            )
                        )
                        F = np.zeros_like(F)
                else:
                    # implicitly using this convention for Fourier transform normalisation
                    F /= F.shape[1]
                # exclude missing data
                weight = (bfw[:, lfi] > 0).astype(float)

                # Transform back into visibility grid
                vgv[:, lfi] = np.dot(F, weight * bfv[:, lfi]).transpose(1, 2, 0, 3)
                # Estimate the noise weights assuming that the errors are all uncorrelated
                # Because we don't track the full covariance matrix we can't recover
                # the pre-transform weights here
                t = np.sum(tools.invert_no_zero(bfw[:, lfi]) * weight ** 2, axis=-2)
                vgw[:, lfi] = tools.invert_no_zero(t[..., np.newaxis, :])

                # invert the forward weights
                if self.invert_weight:
                    vgv[:, lfi] *= tools.invert_no_zero(ns_weight)[:, np.newaxis]
                    vgw[:, lfi] *= (ns_weight ** 2)[:, np.newaxis]
            else:
                # only intended to use mask with pinv
                if self.mask and not self.pinv:
                    raise PipelineConfigError(
                        "If a mask is provided, pinv should be set to True."
                    )
                if self.pinv:
                    if self.mask:
                        F = bfm[:, :, :, lfi, :, np.newaxis, :] * F
                    else:
                        F = np.ones_like(bfw[:, :, :, lfi, :, np.newaxis, :]) * F
                    try:
                        F = np.linalg.pinv(np.conj(np.swapaxes(F, -1, -2)))
                    except:
                        self.log.warning(
                            "Pseudo-inverse failed for frequency {:.2f} MHz.".format(
                                freq[fi]
                            )
                        )
                        F = np.zeros_like(F)
                else:
                    F /= F.shape[1]
                if len(F.shape) == 2:
                    F = F[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
                # exclude missing data
                weight = (bfw[:, :, :, lfi] > 0).astype(float)

                # Transform back into visibility grid
                # Need to treat +/- m separately
                vgv[:, 0, :, lfi] = np.matmul(
                    F[:, 0], (weight[:, 0] * bfv[:, 0, :, lfi])[..., np.newaxis]
                )[..., 0]
                vgv[:, 1, :, lfi] = np.matmul(
                    F[:, 1], (weight[:, 1] * bfv[:, 1, :, lfi].conj())[..., np.newaxis]
                )[..., 0].conj()
                # Estimate the noise weights assuming that the errors are all uncorrelated
                # Because we don't track the full covariance matrix we can't recover
                # the pre-transform weights here
                t = np.sum(
                    tools.invert_no_zero(bfw[:, :, :, lfi]) * weight ** 2, axis=-1
                )
                vgw[:, :, :, lfi] = tools.invert_no_zero(t[..., np.newaxis])

                # invert the forward weights
                if self.invert_weight:
                    vgv[:, :, :, lfi] *= tools.invert_no_zero(ns_weight)
                    vgw[:, :, :, lfi] *= ns_weight ** 2

        return vg


class HybridVisMBeamMask(task.SingleTask):
    """Create a mask isolating the beam in beamformed m-modes.

    Use either a telescope object or a simple model to determine the extent of
    the beam and return a mask.

    Attributes
    ---------

    local_thresh : (default 1e-4)
        Threshold to set the mask compared to the beam maximum at a every
        declination independantly.

    global_thresh : (default 1e-6)
        Threshold to set the mask compared to the beam maximum overall.

    simple_beam : (default False)
        Ignore the thresholds and use a simple beam model.

    n_beam : float (default 4.)
        The width of the unmasked region in units of the beam scale.
        This is based on the cylinder separation and scaled in declination
        by the projection onto celestial coordinates.
        Ignored if simple_beam is False.

    min_width : int (default 50)
        The minimum width of the unmasked region in units of m.
        Ignored if simple_beam is False.

    max_width : int (default 50)
        The minimum width of the unmasked region in units of m.
        Ignored if simple_beam is False.

    taper : float (int 50)
        Span of the taper at the edges of the mask.
    """

    local_thresh = config.Property(proptype=float, default=1e-4)
    global_thresh = config.Property(proptype=float, default=1e-6)
    simple_beam = config.Property(proptype=bool, default=False)
    n_beam = config.Property(proptype=float, default=4.0)
    min_width = config.Property(proptype=int, default=50)
    max_width = config.Property(proptype=int, default=100)
    taper = config.Property(proptype=int, default=50)

    def setup(self, tel):
        """Get physical array properties.

        Parameters
        ----------

        tel : TransitTelescope
            Telescope.
        """

        # get telescope
        self.tel = io.get_telescope(tel)

        # get the baseline grid properties
        xpos = self.tel.baselines[:, 0]
        ypos = self.tel.baselines[:, 1]
        self.cyl_sep = np.abs(xpos)[xpos != 0].min()
        self.ysep = np.abs(ypos)[ypos != 0.0].min()
        self.ny = int(np.round(2 * np.abs(ypos).max() / self.ysep)) + 1
        self.log.debug(
            "Got telescope with {:d} NS grid points with separations {:.3f}".format(
                self.ny, self.ysep
            )
            + " and cylinder separation {:.2f}".format(self.cyl_sep)
        )

    def process(self, mmodes):
        """Calculate the beam mask.

        Parameters
        ----------

        mmodes : containers.HybridVisMModes
            Only used to obtain the axes on which to calculate the mask.

        Returns
        -------

        mask : containers.HybridVisMModes
            The resulting mask
        """

        mmodes.redistribute("freq")

        # axes
        el = mmodes.index_map["el"][:]
        freq = mmodes.index_map["freq"]["centre"]
        m = mmodes.index_map["m"][:]
        m_signed = np.array((m, -m)).T

        # get angular coordinates
        phi = (np.arange(2 * len(m) - 1) - len(m) + 1) * 2 * np.pi / (2 * len(m) - 1)
        theta = np.arcsin(np.where(np.abs(el) <= 1.0, el, 0.0))
        dec = theta + np.radians(self.tel.latitude)

        # set the coordinates of the telescope model
        t_grid, p_grid = np.meshgrid(np.pi / 2 - dec, phi)
        self.tel._angpos = np.vstack((t_grid.flatten(), p_grid.flatten())).T

        # output container
        mask = containers.HybridVisMModes(
            axes_from=mmodes, attrs_from=mmodes, comm=mmodes.comm
        )
        mask.weight[:] = 0

        mmv = mask.vis[:]

        for lfi, fi in mmv.enumerate(3):

            wv = scipy.constants.c / freq[fi] / 1e6

            # calculate the beam model
            if not self.simple_beam:
                feed_x, feed_y = None, None
                for feed_ind in range(self.tel.nfeed):
                    if self.tel.beamclass[feed_ind] == 0:
                        feed_x = feed_ind
                    elif self.tel.beamclass[feed_ind] == 1:
                        feed_y = feed_ind
                    if feed_x is not None and feed_y is not None:
                        break

                if feed_x is None or feed_y is None:
                    raise (
                        Exception(
                            "Could not find feed for polarisation {}.".format(
                                "X" if feed_x is None else "Y"
                            )
                        )
                    )

                beam_x = self.tel.beam(feed_x, fi).reshape(len(phi), len(theta), -1)
                beam_y = self.tel.beam(feed_y, fi).reshape(len(phi), len(theta), -1)

                # final beam model. [pol, theta, phi]
                # TODO: should get pol order from container
                beam = np.array(
                    (
                        (np.abs(beam_x) ** 2).sum(axis=-1).T,
                        (beam_x * beam_y.conj()).sum(axis=-1).T,
                        (beam_y * beam_x.conj()).sum(axis=-1).T,
                        (np.abs(beam_y) ** 2).sum(axis=-1).T,
                    )
                )

                # apodize in prep for m-mode transform
                beam *= np.hanning(beam.shape[-1])
            else:
                # otherwise just model the scaling of the beam with declination
                beam_width = wv / self.cyl_sep
                ew_scale = np.abs(
                    np.cos(dec)
                    / np.sqrt(1 - np.where(np.abs(el) < 1.0, el, 0.0) ** 2)
                    / beam_width
                )
                ew_scale *= self.n_beam
                ew_scale = np.minimum(ew_scale, self.max_width)
                beam = None

            # ensure beam is zero beyond horizon
            horizon = (
                (
                    np.dot(
                        coord.sph_to_cart(self.tel._angpos),
                        coord.sph_to_cart(self.tel.zenith),
                    )
                    > 0.0
                )
                .astype(float)
                .reshape(len(phi), len(theta))
                .T
            )
            beam *= horizon
            beam *= (np.abs(el) <= 1.0)[np.newaxis, :, np.newaxis]

            # for every cylinder separation
            for ci in range(mmodes.index_map["ew"].shape[0]):

                # calculate the beam model in m-space
                if not self.simple_beam:
                    # transform beam model into m
                    beam_phase = (
                        ci * self.cyl_sep / wv * np.cos(dec)[:, np.newaxis] * phi
                    )
                    beam_m = _make_marray(
                        np.exp(-2j * np.pi * beam_phase) * beam, mmax=m.max(),
                    )

                    # null below thresholds
                    beam_m *= (
                        (
                            np.abs(beam_m)
                            > (np.abs(beam_m).max(axis=0) * self.local_thresh)
                        )
                        * (np.abs(beam_m) > (np.abs(beam_m).max() * self.global_thresh))
                    ).astype(float)
                else:
                    # calculate m with max sensitivity
                    m0 = (
                        -2
                        * np.pi
                        * ci
                        * self.cyl_sep
                        / wv
                        * np.cos(dec)[np.newaxis, np.newaxis, :]
                    )

                    # define sensitivity region around this value
                    beam_m = (
                        np.abs(m_signed[..., np.newaxis] - m0)
                        < np.maximum(ew_scale, self.min_width) / 2.0
                    ).astype(float)

                    # add polarisation axis
                    # just repeat them. this makes it easier to iterate later
                    beam_m = np.tile(beam_m, (1, 1, mmv.shape[2], 1))

                # taper the edges
                if self.taper > 0:
                    win = np.hanning(self.taper)
                    for i in range(beam_m.shape[-1]):
                        for pi in range(beam_m.shape[2]):
                            beam_m_cnt = np.concatenate(
                                (beam_m[:0:-1, 1, pi, i], beam_m[:, 0, pi, i]), axis=0
                            )
                            beam_m_cnt = np.convolve(beam_m_cnt, win, "same")
                            beam_m_max = beam_m_cnt.max()
                            beam_m_cnt /= beam_m_max if beam_m_max != 0.0 else 1.0
                            beam_m[:, :, pi, i] = np.vstack(
                                (beam_m_cnt[len(m) - 1 :], beam_m_cnt[len(m) - 1 :: -1])
                            ).T

                mmv[:, :, :, lfi, ci] = beam_m

        return mask


class ApplyMask(task.SingleTask):

    overwrite = config.Property(proptype=bool, default=False)

    def process(self, data, mask):

        if data.vis.shape != mask.vis.shape:
            raise PipelineRuntimeError(
                "Mask and data do not have same shape: {}, {}.".format(
                    data.vis.shape, mask.vis.shape
                )
            )

        if self.overwrite:
            data_out = data
        else:
            data_out = data.__class__(axes_from=data, attrs_from=data, comm=data.comm)

        data_out.vis[:] = data.vis[:] * mask.vis[:]
        data_out.weight[:] = data.weight[:] * tools.invert_no_zero(
            np.abs(mask.vis[:]) ** 2
        )

        return data_out

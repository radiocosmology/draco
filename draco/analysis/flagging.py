"""Tasks for flagging out bad or unwanted data.

This includes data quality flagging on timestream data; sun excision on sidereal
data; and pre-map making flagging on m-modes.

Tasks
=====

.. autosummary::
    :toctree: generated/

    DayMask
    SunClean
    MaskData
"""
import numpy as np

from caput import config
from ch_util import tools

from ..core import task


class DayMask(task.SingleTask):
    """Crudely simulate a masking out of the daytime data.

    Attributes
    ----------
    start, end : float
        Start and end of masked out region.
    width : float
        Use a smooth transition of given width between the fully masked and
        unmasked data. This is interior to the region marked by start and end.
    zero_data : bool, optional
        Zero the data in addition to modifying the noise weights
        (default is True).
    remove_average : bool, optional
        Estimate and remove the mean level from each visibilty. This estimate
        does not use data from the masked region.
    """

    start = config.Property(proptype=float, default=90.0)
    end = config.Property(proptype=float, default=270.0)

    width = config.Property(proptype=float, default=60.0)

    zero_data = config.Property(proptype=bool, default=True)
    remove_average = config.Property(proptype=bool, default=True)

    def process(self, sstream):
        """Apply a day time mask.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Unmasked sidereal stack.

        Returns
        -------
        mstream : containers.SiderealStream
            Masked sidereal stream.
        """

        sstream.redistribute('freq')

        ra_shift = (sstream.ra[:] - self.start) % 360.0
        end_shift = (self.end - self.start) % 360.0

        # Crudely mask the on and off regions
        mask_bool = ra_shift > end_shift

        # Put in the transition at the start of the day
        mask = np.where(ra_shift < self.width,
                        0.5 * (1 + np.cos(np.pi * (ra_shift / self.width))),
                        mask_bool)

        # Put the transition at the end of the day
        mask = np.where(np.logical_and(ra_shift > end_shift - self.width, ra_shift <= end_shift),
                        0.5 * (1 + np.cos(np.pi * ((ra_shift - end_shift) / self.width))),
                        mask)

        if self.remove_average:
            # Estimate the mean level from unmasked data
            import scipy.stats

            nanvis = sstream.vis[:] * np.where(mask_bool, 1.0, np.nan)[np.newaxis, np.newaxis, :]
            average = scipy.stats.nanmedian(nanvis, axis=-1)[:, :, np.newaxis]
            sstream.vis[:] -= average

        # Apply the mask to the data
        if self.zero_data:
            sstream.vis[:] *= mask

        # Modify the noise weights
        sstream.weight[:] *= mask**2

        return sstream


class SunClean(task.SingleTask):
    """Clean the sun from data by projecting out signal from its location.

    Optionally flag out all data around transit, and sunrise/sunset.

    Attributes
    ----------
    flag_time : float, optional
        Flag out time around sun rise/transit/set. Should be set in degrees. If
        :obj:`None` (default), then don't flag at all.
    """

    flag_time = config.Property(proptype=float, default=None)

    def setup(self, inputmap):
        self.inputmap = inputmap

    def process(self, sstream):
        """Apply a day time mask.

        Parameters
        ----------
        sstream : containers.SiderealStream
            Unmasked sidereal stack.

        Returns
        -------
        mstream : containers.SiderealStream
            Masked sidereal stream.
        """

        inputmap = self.inputmap

        from ch_util import ephemeris
        import ephem

        sstream.redistribute('freq')

        def ra_dec_of(body, time):
            obs = ephemeris._get_chime()
            obs.date = ephemeris.unix_to_ephem_time(time)

            body.compute(obs)

            return body.ra, body.dec, body.alt

        # Get array of CSDs for each sample
        ra = sstream.index_map['ra'][:]
        csd = sstream.attrs['csd'] + ra / 360.0

        # Get position of sun at every time sample
        times = ephemeris.csd_to_unix(csd)
        sun_pos = np.array([ra_dec_of(ephem.Sun(), t) for t in times])

        # Get hour angle and dec of sun, in radians
        ha = 2 * np.pi * (ra / 360.0) - sun_pos[:, 0]
        dec = sun_pos[:, 1]
        el = sun_pos[:, 2]

        # Construct lengths for each visibility and determine what polarisation combination they are
        feed_pos = tools.get_feed_positions(inputmap)
        vis_pos = np.array([ feed_pos[ii] - feed_pos[ij] for ii, ij in sstream.index_map['prod'][:]])

        feed_list = [ (inputmap[fi], inputmap[fj]) for fi, fj in sstream.index_map['prod'][:]]
        pol_ind = np.array([ 2 * tools.is_chime_y(fi) + tools.is_chime_y(fj) for fi, fj in feed_list])

        # Initialise new container
        sscut = sstream.__class__(axes_from=sstream, attrs_from=sstream)
        sscut.redistribute('freq')

        wv = 3e2 / sstream.index_map['freq']['centre']

        # Iterate over frequencies and polarisations to null out the sun
        for lfi, fi in sstream.vis[:].enumerate(0):

            # Get the baselines in wavelengths
            u = vis_pos[:, 0] / wv[fi]
            v = vis_pos[:, 1] / wv[fi]

            # Calculate the phase that the sun would have using the fringestop routine
            fsphase = tools.fringestop_phase(ha[np.newaxis, :], np.radians(ephemeris.CHIMELATITUDE),
                                             dec[np.newaxis, :], u[:, np.newaxis], v[:, np.newaxis])

            # Calculate the visibility vector for the sun
            sun_vis = fsphase.conj() * (el > 0.0)

            # Mask out the auto-correlations
            sun_vis *= np.logical_or(u != 0.0, v != 0.0)[:, np.newaxis]

            # Copy over the visiblities and weights
            vis = sstream.vis[fi]
            weight = sstream.weight[fi]
            sscut.vis[fi] = vis
            sscut.weight[fi] = weight

            # Iterate over polarisations to do projection independently for each.
            # This is needed because of the different beams for each pol.
            for pol in range(4):

                # Mask out other polarisations in the visibility vector
                sun_vis_pol = sun_vis * (pol_ind == pol)[:, np.newaxis]

                # Calculate various projections
                vds = (vis * sun_vis_pol.conj() * weight).sum(axis=0)
                sds = (sun_vis_pol * sun_vis_pol.conj() * weight).sum(axis=0)
                isds = tools.invert_no_zero(sds)

                # Subtract sun contribution from visibilities and place in new array
                sscut.vis[fi] -= sun_vis_pol * vds * isds

        # If needed mask out the regions around sun rise, set and transit
        if self.flag_time is not None:

            # Find the RAs of each event
            transit_ra = ephemeris.transit_RA(ephemeris.solar_transit(times[0], times[-1]))
            rise_ra = ephemeris.transit_RA(ephemeris.solar_rising(times[0], times[-1]))
            set_ra = ephemeris.transit_RA(ephemeris.solar_setting(times[0], times[-1]))

            # Construct a mask for each
            rise_mask = ((ra - rise_ra) % 360.0) > self.flag_time
            set_mask = ((ra - set_ra + self.flag_time) % 360.0) > self.flag_time
            transit_mask = ((ra - transit_ra + self.flag_time / 2) % 360.0) > self.flag_time

            # Combine the masks and apply to data
            mask = np.logical_and(rise_mask, np.logical_and(set_mask, transit_mask))
            sscut.weight[:] *= mask

        return sscut


class MaskData(task.SingleTask):
    """Mask out data ahead of map making.

    Attributes
    ----------
    auto_correlations : bool
        Exclude auto correlations if set (default=False).
    m_zero : bool
        Ignore the m=0 mode (default=False).
    positive_m : bool
        Include positive m-modes (default=True).
    negative_m : bool
        Include negative m-modes (default=True).
    """

    auto_correlations = config.Property(proptype=bool, default=False)
    m_zero = config.Property(proptype=bool, default=False)
    positive_m = config.Property(proptype=bool, default=True)
    negative_m = config.Property(proptype=bool, default=True)

    def process(self, mmodes):
        """Mask out unwanted datain the m-modes.

        Parameters
        ----------
        mmodes : containers.MModes

        Returns
        -------
        mmodes : containers.MModes
        """

        # Exclude auto correlations if set
        if not self.auto_correlations:
            for pi, (fi, fj) in enumerate(mmodes.index_map['prod']):
                if fi == fj:
                    mmodes.weight[..., pi] = 0.0

        # Apply m based masks
        if not self.m_zero:
            mmodes.weight[0] = 0.0

        if not self.positive_m:
            mmodes.weight[1:, 0] = 0.0

        if not self.negative_m:
            mmodes.weight[1:, 1] = 0.0

        return mmodes

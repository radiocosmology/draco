"""
===============================================================
Tasks for Flagging Data (:mod:`~ch_pipeline.analysis.flagging`)
===============================================================

.. currentmodule:: ch_pipeline.analysis.flagging

Tasks for calculating RFI and data quality masks for timestream data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    RFIFilter
    ChannelFlagger
    BadNodeFlagger
    DayMask
    SunClean
"""
import numpy as np

from caput import config
from caput import mpiutil
from ch_util import rfi, data_quality, tools

from . import task


class RFIFilter(task.SingleTask):
    """Filter RFI from a Timestream.

    This task works on the parallel
    :class:`~ch_pipeline.containers.TimeStream` objects.

    Attributes
    ----------
    threshold_mad : float
        Threshold above which we mask the data.
    """

    threshold_mad = config.Property(proptype=float, default=5.0)

    flag1d = config.Property(proptype=bool, default=False)

    def process(self, data):

        if mpiutil.rank0:
            print "RFI filtering %s" % data.attrs['tag']

        # Construct RFI mask
        mask = rfi.flag_dataset(data, only_autos=False, threshold=self.threshold_mad, flag1d=self.flag1d)

        data.weight[:] *= (1 - mask)  # Switch from mask to inverse noise weight

        # Redistribute across frequency
        data.redistribute('freq')

        return data


class ChannelFlagger(task.SingleTask):
    """Mask out channels that appear weird in some way.

    Parameters
    ----------
    test_freq : list
        Frequencies to test the data at.
    """

    test_freq = config.Property(proptype=list, default=[610.0])

    ignore_fit = config.Property(proptype=bool, default=False)
    ignore_noise = config.Property(proptype=bool, default=False)
    ignore_gains = config.Property(proptype=bool, default=False)

    known_bad = config.Property(proptype=list, default=[])

    def process(self, timestream, inputmap):
        """Flag bad channels in timestream.

        Parameters
        ----------
        timestream : andata.CorrData
            Timestream to flag.

        Returns
        -------
        timestream : andata.CorrData
            Returns the same timestream object with a modified weight dataset.
        """

        # Redistribute over the frequency direction
        timestream.redistribute('freq')

        # Find the indices for frequencies in this timestream nearest
        # to the given physical frequencies
        freq_ind = [np.argmin(np.abs(timestream.freq - freq)) for freq in self.test_freq]

        # Create a global channel weight (channels are bad by default)
        chan_mask = np.zeros(timestream.ninput, dtype=np.int)

        # Mark any CHIME channels as good
        for i in range(timestream.ninput):
            if isinstance(inputmap[i], tools.CHIMEAntenna):
                chan_mask[i] = 1

        # Calculate start and end frequencies
        sf = timestream.vis.local_offset[0]
        ef = sf + timestream.vis.local_shape[0]

        # Iterate over frequencies and find bad channels
        for fi in freq_ind:

            # Only run good_channels if frequency is local
            if fi >= sf and fi < ef:

                # Run good channels code and unpack arguments
                res = data_quality.good_channels(timestream, test_freq=fi, inputs=inputmap, verbose=False)
                good_gains, good_noise, good_fit, test_channels = res

                print ("Frequency %i bad channels: blank %i; gains %i; noise %i; fit %i %s" %
                       ( fi, np.sum(chan_mask == 0), np.sum(good_gains == 0), np.sum(good_noise == 0),
                         np.sum(good_fit == 0), '[ignored]' if self.ignore_fit else ''))

                if good_noise is None:
                    good_noise = np.ones_like(test_channels)

                # Construct the overall channel mask for this frequency
                if not self.ignore_gains:
                    chan_mask[test_channels] *= good_gains
                if not self.ignore_noise:
                    chan_mask[test_channels] *= good_noise
                if not self.ignore_fit:
                    chan_mask[test_channels] *= good_fit

        # Gather the channel flags from all nodes, and combine into a
        # single flag (checking that all tests pass)
        chan_mask_all = np.zeros((timestream.comm.size, timestream.ninput), dtype=np.int)
        timestream.comm.Allgather(chan_mask, chan_mask_all)
        chan_mask = np.prod(chan_mask_all, axis=0)

        # Mark already known bad channels
        for ch in self.known_bad:
            chan_mask[ch] = 0.0

        # Apply weights to files weight array
        chan_mask = chan_mask[np.newaxis, :, np.newaxis]
        weight = timestream.datasets['vis_weight'][:]
        tools.apply_gain(weight, chan_mask, out=weight)

        return timestream


class BadNodeFlagger(task.SingleTask):
    """Flag out bad GPU nodes by giving zero weight to their frequencies.

    Parameters
    ----------
    nodes : list of ints
        Indices of bad nodes to flag.
    flag_freq_zero : boolean, optional
        Whether to flag out frequency zero.
    """

    nodes = config.Property(proptype=list, default=[])

    flag_freq_zero = config.Property(proptype=bool, default=True)

    def process(self, timestream):
        """Flag out bad nodes by giving them zero weight.

        Parameters
        ----------
        timestream : andata.CorrData or containers.SiderealStream

        Returns
        -------
        flagged_timestream : same type as timestream
        """

        timestream.redistribute(['prod', 'input'])

        if self.flag_freq_zero:
            timestream.datasets['vis_weight'][0] = 0.0

        for node in self.nodes:
            if node < 0 or node >= 16:
                raise RuntimeError('Node index (%i) is invalid (should be 0-15).' % node)

            timestream.datasets['vis_weight'][node::16] = 0.0

        timestream.redistribute('freq')

        return timestream


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

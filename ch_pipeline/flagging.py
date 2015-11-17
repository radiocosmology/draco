"""
======================================================
Tasks for Flagging Data (:mod:`~ch_pipeline.flagging`)
======================================================

.. currentmodule:: ch_pipeline.flagging

Tasks for calculating RFI and data quality masks for timestream data.

Tasks
=====

.. autosummary::
    :toctree: generated/

    RFIFilter
    ChannelFlagger
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

        # Create a global channel weight
        chan_mask = np.ones(timestream.ninput, dtype=np.int)

        # Mark any Blank channels as bad
        for i in range(timestream.ninput):
            if isinstance(inputmap[i], tools.Blank):
                chan_mask[i] = 0

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

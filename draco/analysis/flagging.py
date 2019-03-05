"""Tasks for flagging out bad or unwanted data.

This includes data quality flagging on timestream data; sun excision on sidereal
data; and pre-map making flagging on m-modes.

Tasks
=====

.. autosummary::
    :toctree:

    DayMask
    MaskData
    MaskBaselines
    RadiometerWeight
"""
import numpy as np
from scipy.ndimage import median_filter

from caput import config

from ..core import task, containers, io
from ..util import tools


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


class MaskBaselines(task.SingleTask):
    """Mask out baselines from a dataset.

    Attributes
    ----------
    mask_long_ns : float
        Mask out baselines longer than a given distance in the N/S direction.
    mask_short : float
        Mask out baselines shorter than a given distance.
    """

    mask_long_ns = config.Property(proptype=float, default=None)
    mask_short = config.Property(proptype=float, default=None)

    def setup(self, telescope):
        """Set the telescope model.

        Parameters
        ----------
        telescope : TransitTelescope
        """

        self.telescope = io.get_telescope(telescope)

    def process(self, ss):
        """Apply the mask to data.

        Parameters
        ----------
        ss : SiderealStream or TimeStream
            Data to mask. Applied in place.
        """

        ss.redistribute('freq')

        baselines = self.telescope.baselines

        if self.mask_long_ns is not None:
            long_ns_mask = np.abs(baselines[:, 1]) < self.mask_long_ns
            ss.weight[:] *= long_ns_mask[np.newaxis, :, np.newaxis]

        if self.mask_short is not None:
            short_mask = np.sum(baselines**2, axis=1) > self.mask_short
            ss.weight[:] *= short_mask[np.newaxis, :, np.newaxis]

        return ss


class RadiometerWeight(task.SingleTask):
    """Update vis_weight according to the radiometer equation:

    .. math::

        \text{weight}_{ij} = N_\text{samp} / V_{ii} V_{jj}

    Attributes
    ----------
    replace : bool, optional
        Replace any existing weights (default). If `False` then we multiply the
        existing weights by the radiometer values.

    """

    replace = config.Property(proptype=bool, default=True)

    def process(self, stream):
        """Change the vis weight.

        Parameters
        ----------
        stream : SiderealStream or TimeStream
            Data to be weighted. This is done in place.

        Returns
        --------
        stream : SiderealStream or TimeStream
        """

        from caput.time import STELLAR_S

        # Redistribute over the frequency direction
        stream.redistribute('freq')

        ninput = len(stream.index_map['input'])
        nprod = len(stream.index_map['prod'])

        if nprod != (ninput * (ninput + 1) / 2):
            raise RuntimeError('Must have a input stream with the full correlation triangle.')

        freq_width = np.median(stream.index_map['freq']['width'])

        if isinstance(stream, containers.SiderealStream):
            RA_S = 240 * STELLAR_S  # SI seconds in 1 deg of RA change
            int_time = np.median(np.abs(np.diff(stream.ra))) / RA_S
        else:
            int_time = np.median(np.abs(np.diff(stream.index_map['time'])))

        if self.replace:
            stream.weight[:] = 1.0

        # Construct and set the correct weights in place
        nsamp = (1e6 * freq_width * int_time)
        autos = tools.extract_diagonal(stream.vis[:]).real
        weight_fac = nsamp**0.5 / autos
        tools.apply_gain(stream.weight[:], weight_fac, out=stream.weight[:])

        # Return timestream with updated weights
        return stream


class SmoothVisWeight(task.SingleTask):
    """Smooth the visibility weights with a median filter.

    Attributes
    ----------
    kernel_size : int
        Size of the kernel for the median filter in time points.
        Default is 31, corresponding to ~5 minutes window for 10s cadence data.

    """

    # 31 time points correspond to ~ 5min in 10s cadence
    kernel_size = config.Property(proptype=int, default=31)

    def process(self, data):
        """Smooth the weights with a median filter.

        Parameters
        ----------
        data : :class:`andata.CorrData` or :class:`containers.TimeStream` object
            Data containing the weights to be smoothed

        Returns
        -------
        data : :class:`andata.CorrData` object
            Data object containing the same data as the input, but with the
            weights substituted by the smoothed ones.
        """

        # Ensure data is distributed in frequency:
        data.redistribute('freq')
        # Full slice reutrns an MPIArray
        weight = data.weight[:]
        # Data will be distributed in frequency.
        # So a frequency loop will not be too large.
        for lfi, gfi in weight.enumerate(axis=0):

            # MPIArray takes the local index, returns a local np.ndarray
            # Find values equal to zero to preserve them in final weights
            zeromask = (weight[lfi] == 0.)
            # Median filter. Mode='nearest' to prevent steps close to
            # the end from being washed
            weight[lfi] = median_filter(
                weight[lfi], size=(1, self.kernel_size), mode='nearest')
            # Ensure zero values are zero
            weight[lfi][zeromask] = 0.

        return data


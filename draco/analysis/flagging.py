"""Tasks for flagging out bad or unwanted data.

This includes data quality flagging on timestream data; sun excision on sidereal
data; and pre-map making flagging on m-modes.

Tasks
=====

.. autosummary::
    :toctree:

    DayMask
    MaskData
"""
import numpy as np

from caput import config

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

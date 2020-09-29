"""Delay space spectrum estimation and filtering.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import time

import numpy as np

from caput import config
from cora.util import units

from ..core import task, io


class DayenuDelayFilter(task.SingleTask):
    """Apply DAYENU filter along the frequency axis.

    See https://arxiv.org/abs/2004.11397.

    Attributes
    ----------
    za_cut : float
        Sine of the maximum zenith angle included in
        baseline-dependent delay filtering. Default is 1
        which corresponds to the horizon (ie: filters
        out all zenith angles). Setting to zero turns off
        baseline dependent cut.
    telescope_orientation : one of ('NS', 'EW', 'none')
        Determines if the baseline-dependent delay cut is based on
        the north-south component, the east-west component or the full
        baseline length. For cylindrical telescopes oriented in the
        NS direction (like CHIME) use 'NS'. The default is 'NS'.
    epsilon : float
        Rejection in power.
    tauw : float
        Delay cutoff in nanoseconds.
    single_mask : bool
        Apply a single frequency mask for all times.  Only includes frequencies
        where the weights are nonzero for all times.  Otherwise will construct
        a filter for all unique single time frequency masks (much slower).
    """

    za_cut = config.Property(proptype=float, default=1.0)
    telescope_orientation = config.enum(["NS", "EW", "none"], default="NS")
    epsilon = config.Property(proptype=float, default=1e-10)
    tauw = config.Property(proptype=float, default=100.0)
    single_mask = config.Property(proptype=bool, default=True)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
        """
        self.telescope = io.get_telescope(telescope)

        self.log.info("Initial delay cut set to %0.1f nanosec." % self.tauw)

    def process(self, stream):
        """Filter out delays from a SiderealStream or TimeStream.

        Parameters
        ----------
        stream : containers.SiderealStream
            Data to filter.

        Returns
        -------
        stream_filt : containers.SiderealStream
            Filtered dataset.
        """
        # Distribute over products
        stream.redistribute(["input", "prod", "stack"])

        # Extract the required axes
        freq = stream.freq[:]

        nprod = stream.vis.local_shape[1]
        sp = stream.vis.local_offset[1]
        ep = sp + nprod

        prod = stream.prodstack[sp:ep]

        # Determine the baseline dependent cutoff
        cutoff = self._get_cut(prod)

        # Dereference the required datasets
        vis = stream.vis[:].view(np.ndarray)
        weight = stream.weight[:].view(np.ndarray)

        # Loop over products
        for bb, bcut in enumerate(cutoff):

            t0 = time.time()

            # Flag frequencies and times with zero weight
            flag = weight[:, bb, :] > 0.0

            if self.single_mask:
                flag = np.all(flag, axis=-1, keepdims=True)

            weight[:, bb] *= flag.astype(weight.dtype)

            if not np.any(flag):
                continue

            self.log.info(
                "Filtering baseline %d of %d. [%0.1f nsec]" % (bb, nprod, bcut)
            )

            # Construct the filter
            NF = highpass_delay_filter(freq, bcut, flag, epsilon=self.epsilon)

            # Apply the filter
            if self.single_mask:
                vis[:, bb] = np.matmul(NF[:, :, 0], vis[:, bb])
            else:
                vis[:, bb] = np.sum(NF * vis[np.newaxis, :, bb], axis=1)

            self.log.info("Took %0.2f seconds." % (time.time() - t0,))

        return stream

    def _get_cut(self, prod):

        baselines = (
            self.telescope.feedpositions[prod["input_a"], :]
            - self.telescope.feedpositions[prod["input_b"], :]
        )

        if self.telescope_orientation == "NS":
            baselines = abs(baselines[:, 1])  # Y baseline
        elif self.telescope_orientation == "EW":
            baselines = abs(baselines[:, 0])  # X baseline
        else:
            baselines = np.sqrt(np.sum(baselines ** 2, axis=-1))  # Norm

        baseline_delay_cut = 1e9 * self.za_cut * baselines / units.c

        return baseline_delay_cut + self.tauw


class DayenuMFilter(task.SingleTask):
    """Apply Dayenu bandpass filter along the time axis.

    Attributes
    ----------
    dec: float
        The bandpass filter is centered on the m corresponding to the
        fringe rate of a source at the meridian at this declination.
    epsilon : float
        Rejection in power.
    fkeep_intra : float
        Width of the bandpass filter for intracylinder baselines in terms
        of the fraction of the telescope cylinder width.
    fkeep_inter : float
        Width of the bandpass filter for intercylinder baselines in terms
        of the fraction of the telescope cylinder width.
    """

    dec = config.Property(proptype=float, default=40.0)
    epsilon = config.Property(proptype=float, default=1e-8)
    fkeep_intra = config.Property(proptype=float, default=0.75)
    fkeep_inter = config.Property(proptype=float, default=0.75)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, stream):
        """Filter out m-modes from a SiderealStream or TimeStream.

        Parameters
        ----------
        stream : containers.SiderealStream
            Data to filter.

        Returns
        -------
        stream_filt : containers.SiderealStream
            Filtered dataset.
        """
        # Distribute over products
        stream.redistribute("freq")

        # Extract the required axes
        ra = np.radians(stream.ra[:])

        nfreq = stream.vis.local_shape[0]
        sf = stream.vis.local_offset[0]
        ef = sf + nfreq

        freq = stream.freq[sf:ef]

        # Calculate unique E-W baselines
        prod = stream.prodstack[:]
        baselines = (
            self.telescope.feedpositions[prod["input_a"], 0]
            - self.telescope.feedpositions[prod["input_b"], 0]
        )
        baselines = (
            np.round(baselines / self.telescope.cylinder_spacing)
            * self.telescope.cylinder_spacing
        )

        uniqb, indexb = np.unique(baselines, return_inverse=True)
        nuniq = uniqb.size

        db = 0.5 * self.telescope.cylinder_spacing

        # Dereference the required datasets
        vis = stream.vis[:].view(np.ndarray)
        weight = stream.weight[:].view(np.ndarray)

        # Loop over frequencies
        for ff, nu in enumerate(freq):

            t0 = time.time()

            # Flag frequencies and times with zero weight
            flag = weight[ff, :, :] > 0.0

            gb = np.flatnonzero(np.any(flag, axis=-1))

            if gb.size == 0:
                continue

            flag = np.sum(flag[gb, :], axis=0, keepdims=True) > (0.90 * float(gb.size))

            weight[ff] *= flag.astype(weight.dtype)

            if not np.any(flag):
                continue

            self.log.info("Filtering freq %d of %d." % (ff, nfreq))

            # Construct the filters
            m_cut = np.abs(self._get_cut(nu, db))

            m_center_intra = 0.5 * (2.0 - self.fkeep_intra) * m_cut
            m_cut_intra = 0.5 * self.fkeep_intra * m_cut

            m_cut_inter = self.fkeep_inter * m_cut

            INTRA = bandpass_mmode_filter(
                ra, m_center_intra, m_cut_intra, flag, epsilon=self.epsilon
            )
            INTER = lowpass_mmode_filter(ra, m_cut_inter, flag, epsilon=self.epsilon)

            # Loop over E-W baselines
            for uu, ub in enumerate(uniqb):

                iub = np.flatnonzero(indexb == uu)

                # Construct the filter
                if np.abs(ub) < db:
                    vis[ff, iub, :] = np.matmul(INTRA, vis[ff, iub, :, np.newaxis])[
                        :, :, 0
                    ]

                else:
                    m_center = self._get_cut(nu, ub)
                    mixer = np.exp(-1.0j * m_center * ra)[np.newaxis, :]
                    vis_mixed = vis[ff, iub, :] * mixer

                    vis[ff, iub, :] = (
                        np.matmul(INTER, vis_mixed[:, :, np.newaxis])[:, :, 0]
                        * mixer.conj()
                    )

            self.log.info("Took %0.2f seconds." % (time.time() - t0,))

        return stream

    def _get_cut(self, freq, xsep):

        lmbda = units.c / (freq * 1e6)
        u = xsep / lmbda
        m = instantaneous_m(
            0.0, np.radians(self.telescope.latitude), np.radians(self.dec), u, 0.0
        )

        return m


def highpass_delay_filter(freq, tau_cut, flag, epsilon=1e-8):

    ishp = flag.shape
    nfreq = freq.size
    assert ishp[0] == nfreq

    oshp = (nfreq,) + ishp

    cov = np.eye(nfreq, dtype=np.float64)
    cov += (
        np.sinc(2.0 * tau_cut * (freq[:, np.newaxis] - freq[np.newaxis, :]) * 1e-3)
        / epsilon
    )

    uflag, uindex = np.unique(flag.reshape(nfreq, -1), return_inverse=True, axis=-1)
    uflag = uflag.T
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    pinv = np.swapaxes(pinv, 0, 2)[:, :, uindex].reshape(oshp)

    return pinv


def bandpass_mmode_filter(ra, m_center, m_cut, flag, epsilon=1e-8):

    ishp = flag.shape
    nra = ra.size
    assert ishp[-1] == nra

    oshp = ishp + (nra,)

    a = np.median(np.abs(np.diff(ra))) * m_cut / np.pi
    aeps = a * epsilon

    dra = ra[:, np.newaxis] - ra[np.newaxis, :]

    cov = np.eye(nra, dtype=np.float64) / aeps
    cov += (
        2
        * a
        * (1.0 - 1.0 / aeps)
        * np.sinc(m_cut * dra / np.pi)
        * np.cos(m_center * dra)
    )

    uflag, uindex = np.unique(flag.reshape(-1, nra), return_inverse=True, axis=0)
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    pinv = pinv[uindex, :, :].reshape(oshp)

    return pinv


def lowpass_mmode_filter(ra, m_cut, flag, epsilon=1e-8):

    ishp = flag.shape
    nra = ra.size
    assert ishp[-1] == nra

    oshp = ishp + (nra,)

    a = np.median(np.abs(np.diff(ra))) * m_cut / np.pi
    aeps = a * epsilon

    dra = ra[:, np.newaxis] - ra[np.newaxis, :]

    cov = np.eye(nra, dtype=np.float64) / aeps
    cov += a * (1.0 - 1.0 / aeps) * np.sinc(m_cut * dra / np.pi)

    uflag, uindex = np.unique(flag.reshape(-1, nra), return_inverse=True, axis=0)
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    pinv = pinv[uindex, :, :].reshape(oshp)

    return pinv


def highpass_mmode_filter(ra, m_cut, flag, epsilon=1e-8):

    ishp = flag.shape
    nra = ra.size
    assert ishp[-1] == nra

    oshp = ishp + (nra,)

    dra = ra[:, np.newaxis] - ra[np.newaxis, :]

    cov = np.eye(nra, dtype=np.float64)
    cov += np.sinc(m_cut * dra / np.pi) / epsilon

    uflag, uindex = np.unique(flag.reshape(-1, nra), return_inverse=True, axis=0)
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    pinv = pinv[uindex, :, :].reshape(oshp)

    return pinv


def instantaneous_m(ha, lat, dec, u, v, w=0.0):

    deriv = u * (-1 * np.cos(dec) * np.cos(ha))
    deriv += v * (np.sin(lat) * np.cos(dec) * np.sin(ha))
    deriv += w * (-1 * np.cos(lat) * np.cos(dec) * np.sin(ha))

    return 2.0 * np.pi * deriv

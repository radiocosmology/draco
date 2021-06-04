"""DAYENU delay and m-mode filtering.

See https://arxiv.org/abs/2004.11397 for a description.
"""

import time

import numpy as np
import scipy.interpolate

from caput import config, memh5
from cora.util import units

from ..core import task, io
from ..util import tools


class DayenuDelayFilter(task.SingleTask):
    """Apply a DAYENU high-pass delay filter to visibility data.

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
        The stop-band rejection of the filter.
    tauw : float
        Delay cutoff in micro-seconds.
    single_mask : bool
        Apply a single frequency mask for all times.  Only includes
        frequencies where the weights are nonzero for all times.
        Otherwise will construct a filter for all unique single-time
        frequency masks (can be significantly slower).
    """

    za_cut = config.Property(proptype=float, default=1.0)
    telescope_orientation = config.enum(["NS", "EW", "none"], default="NS")
    epsilon = config.Property(proptype=float, default=1e-10)
    tauw = config.Property(proptype=float, default=0.100)
    single_mask = config.Property(proptype=bool, default=True)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
        """
        self.telescope = io.get_telescope(telescope)

        self.log.info("Instrumental delay cut set to %0.3f micro-sec." % self.tauw)

    def process(self, stream):
        """Filter out delays from a SiderealStream or TimeStream.

        Parameters
        ----------
        stream : SiderealStream
            Data to filter.

        Returns
        -------
        stream_filt : SiderealStream
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

            var = tools.invert_no_zero(weight[:, bb])

            self.log.info(
                "Filtering baseline %d of %d. [%0.3f micro-sec]" % (bb, nprod, bcut)
            )

            # Construct the filter
            NF, index = highpass_delay_filter(freq, bcut, flag, epsilon=self.epsilon)

            # Apply the filter
            if self.single_mask:
                vis[:, bb] = np.matmul(NF[:, :, 0], vis[:, bb])
                weight[:, bb] = tools.invert_no_zero(np.matmul(NF[:, :, 0] ** 2, var))
            else:
                self.log.info("There are %d unique masks/filters." % len(index))
                for ii, ind in enumerate(index):
                    vis[:, bb, ind] = np.matmul(NF[:, :, ii], vis[:, bb, ind])
                    weight[:, bb, ind] = tools.invert_no_zero(
                        np.matmul(NF[:, :, ii] ** 2, var[:, ind])
                    )

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

        baseline_delay_cut = 1e6 * self.za_cut * baselines / units.c

        return baseline_delay_cut + self.tauw


class DayenuDelayFilterMap(task.SingleTask):
    """Apply a DAYENU high-pass delay filter to ringmap data.

    Attributes
    ----------
    epsilon : float
        The stop-band rejection of the filter.
    filename : str
        The name of an hdf5 file containing a DelayCutoff container.
        If a filename is provided, then it will be loaded during setup
        and the `cutoff` dataset will be interpolated to determine
        the cutoff of the filter based on the el coordinate of the map.
        If a filename is not provided, then a single cutoff given by the
        tauw property will be used for all el.
    tauw : float
        Delay cutoff in micro-seconds.
    single_mask : bool
        Apply a single frequency mask for all times.  Only includes
        frequencies where the weights are nonzero for all times.
        Otherwise will construct a filter for all unique single-time
        frequency masks (can be significantly slower).
    """

    epsilon = config.Property(proptype=float, default=1e-12)
    filename = config.Property(proptype=str, default=None)
    tauw = config.Property(proptype=float, default=0.100)
    single_mask = config.Property(proptype=bool, default=True)

    _ax_dist = "el"

    def setup(self):
        """Create the function used to determine the delay cutoff."""

        if self.filename is not None:

            fcut = containers.DelayCutoff.from_file(self.filename, distributed=False)
            kind = fcut.attrs.get("kind", "linear")

            self.log.info(
                f"Using {kind} interpolation of the delay cut in the file: "
                f"{self.filename}"
            )

            self._cut_interpolator = {}
            for pp, pol in enumerate(fcut.pol):

                self._cut_interpolator[pol] = scipy.interpolate.interp1d(
                    fcut.el,
                    fcut.cutoff[pp],
                    kind=kind,
                    bounds_error=False,
                    fill_value=self.tauw,
                )

        else:
            self._cut_interpolator = None

    def process(self, ringmap):
        """Filter out delays from a RingMap.

        Parameters
        ----------
        ringmap : RingMap
            Data to filter.

        Returns
        -------
        ringmap_filt : RingMap
            Filtered data.
        """
        # Distribute over el
        ringmap.redistribute(self._ax_dist)

        # Extract the required axes
        axes = list(ringmap.map.attrs["axis"])
        ax_freq = axes.index("freq")
        ax_dist = axes.index(self._ax_dist)

        lshp = ringmap.map.local_shape[0:ax_freq]

        freq = ringmap.freq[:]

        nel = ringmap.map.local_shape[ax_dist]
        sel = ringmap.map.local_offset[ax_dist]
        eel = sel + nel

        els = ringmap.index_map[self._ax_dist][sel:eel]

        # Dereference the required datasets
        rm = ringmap.map[:].view(np.ndarray)
        weight = ringmap.weight[:].view(np.ndarray)

        # Loop over beam and polarisation
        for ind in np.ndindex(*lshp):

            wind = ind[1:]

            kwargs = {ax: ringmap.index_map[ax][ii] for ax, ii in zip(axes, ind)}

            for ee, el in enumerate(els):

                t0 = time.time()

                slc = ind + (slice(None), slice(None), ee)
                wslc = slc[1:]

                # Flag frequencies and times with zero weight
                flag = weight[wslc] > 0.0

                if self.single_mask:
                    flag = np.all(flag, axis=-1, keepdims=True)
                    weight[wslc] *= flag.astype(weight.dtype)

                if not np.any(flag):
                    continue

                # Determine the delay cutoff
                ecut = self._get_cut(el, **kwargs)

                self.log.info(
                    "Filtering el %d of %d. [%0.3f micro-sec]" % (ee, nel, ecut)
                )

                erm = rm[slc]
                evar = tools.invert_no_zero(weight[wslc])

                # Construct the filter
                NF, index = highpass_delay_filter(
                    freq, ecut, flag, epsilon=self.epsilon
                )

                # Apply the filter
                if self.single_mask:

                    rm[slc] = np.matmul(NF[:, :, 0], erm)
                    weight[wslc] = tools.invert_no_zero(
                        np.matmul(NF[:, :, 0] ** 2, evar)
                    )

                else:

                    self.log.info("There are %d unique masks/filters." % len(index))

                    for ii, rr in enumerate(index):
                        rm[ind][:, rr, ee] = np.matmul(NF[:, :, ii], erm[:, rr])
                        weight[wind][:, rr, ee] = tools.invert_no_zero(
                            np.matmul(NF[:, :, ii] ** 2, evar[:, rr])
                        )

                self.log.info("Took %0.2f seconds." % (time.time() - t0,))

        return ringmap

    def _get_cut(self, el, pol=None, **kwargs):
        """Return the delay cutoff in micro-seconds."""

        if self._cut_interpolator is None:
            return self.tauw

        elif pol in self._cut_interpolator:
            return self._cut_interpolator[pol](el)

        else:
            # The file does not contain this polarisation (likely XY or YX).
            # Use the maximum value over the polarisations that we do have.
            return np.max([func(el) for func in self._cut_interpolator.values()])


class DayenuMFilter(task.SingleTask):
    """Apply a DAYENU bandpass m-mode filter.

    Attributes
    ----------
    dec: float
        The bandpass filter is centered on the m corresponding to the
        fringe rate of a source at the meridian at this declination.
    epsilon : float
        The stop-band rejection of the filter.
    fkeep_intra : float
        Width of the bandpass filter for intracylinder baselines in terms
        of the fraction of the telescope cylinder width.
    fkeep_inter : float
        Width of the bandpass filter for intercylinder baselines in terms
        of the fraction of the telescope cylinder width.
    """

    dec = config.Property(proptype=float, default=40.0)
    epsilon = config.Property(proptype=float, default=1e-10)
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
        stream : SiderealStream
            Data to filter.

        Returns
        -------
        stream_filt : SiderealStream
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


def highpass_delay_filter(freq, tau_cut, flag, epsilon=1e-10):
    """Construct a high-pass delay filter.

    The stop band will range from [-tau_cut, tau_cut].

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequency in MHz.
    tau_cut : float
        The half width of the stop band in micro-seconds.
    flag : np.ndarray[nfreq, ntime]
        Boolean flag that indicates what frequencies are valid
        as a function of time.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-10.

    Returns
    -------
    pinv : np.ndarray[nfreq, nfreq, ntime_uniq]
        High pass filter for each set of unique frequency flags.
    index : list of length nuniq_time
        Maps the last axis of pinv to the original time axis.
        Apply pinv[:, :, i] to the time samples at index[i].
    """

    ishp = flag.shape
    nfreq = freq.size
    assert ishp[0] == nfreq
    assert len(ishp) == 2

    cov = np.eye(nfreq, dtype=np.float64)
    cov += (
        np.sinc(2.0 * tau_cut * (freq[:, np.newaxis] - freq[np.newaxis, :])) / epsilon
    )

    uflag, uindex = np.unique(flag.reshape(nfreq, -1), return_inverse=True, axis=-1)
    uflag = uflag.T
    uflag = uflag[:, np.newaxis, :] & uflag[:, :, np.newaxis]
    uflag = uflag.astype(np.float64)

    ucov = uflag * cov[np.newaxis, :, :]

    pinv = np.linalg.pinv(ucov, hermitian=True) * uflag
    pinv = np.swapaxes(pinv, 0, 2)

    index = [np.flatnonzero(uindex == ii) for ii in range(pinv.shape[-1])]

    return pinv, index


def bandpass_mmode_filter(ra, m_center, m_cut, flag, epsilon=1e-10):
    """Construct a bandpass m-mode filter.

    The pass band will range from [m_center - m_cut, m_center + m_cut].

    Parameters
    ----------
    ra : np.ndarray[nra,]
        Righ ascension in radians.
    m_center : float
        The center of the pass band.
    m_cut : float
        The half width of the pass band.
    flag : np.ndarray[nfreq, nra]
        Boolean flag that indicates what right ascensions are valid
        as a function of frequency.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-10.

    Returns
    -------
    pinv : np.ndarray[nfreq, nra, nra]
        Bandpass m-mode filter for each frequency.
    """
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


def lowpass_mmode_filter(ra, m_cut, flag, epsilon=1e-10):
    """Construct a low-pass m-mode filter.

    The pass band will range from [-m_cut, m_cut].

    Parameters
    ----------
    ra : np.ndarray[nra,]
        Righ ascension in radians.
    m_cut : float
        The half width of the pass band.
    flag : np.ndarray[nfreq, nra]
        Boolean flag that indicates what right ascensions are valid
        as a function of frequency.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-10.

    Returns
    -------
    pinv : np.ndarray[nfreq, nra, nra]
        Low-pass m-mode filter for each frequency.
    """
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


def highpass_mmode_filter(ra, m_cut, flag, epsilon=1e-10):
    """Construct a high-pass m-mode filter.

    The stop band will range from [-m_cut, m_cut].

    Parameters
    ----------
    ra : np.ndarray[nra,]
        Righ ascension in radians.
    m_cut : float
        The half width of the stop band.
    flag : np.ndarray[nfreq, nra]
        Boolean flag that indicates what right ascensions are valid
        as a function of frequency.
    epsilon : float
        The stop-band rejection of the filter.  Defaults to 1e-10.

    Returns
    -------
    pinv : np.ndarray[nfreq, nra, nra]
        High-pass m-mode filter for each frequency.
    """
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
    """Calculate the instantaneous fringe-rate.

    Parameters
    ----------
    ha : float
        Hour angle in radians.
    dec : float
        Declination in radians.
    u : float
        EW baseline distance in wavelengths.
    v : float
        NS baseline distance in wavelengths.
    w : float
        Vertical baseline distance in wavelengths.

    Returns
    -------
    m : float
        The fringe-rate of the requested location on the sky
        as measured by the requested baseline.
    """

    deriv = u * (-1 * np.cos(dec) * np.cos(ha))
    deriv += v * (np.sin(lat) * np.cos(dec) * np.sin(ha))
    deriv += w * (-1 * np.cos(lat) * np.cos(dec) * np.sin(ha))

    return 2.0 * np.pi * deriv

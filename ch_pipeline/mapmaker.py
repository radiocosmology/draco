"""
===============================================
Map making tasks (:mod:`~ch_pipeline.mapmaker`)
===============================================

.. currentmodule:: ch_pipeline.mapmaker

Tools for map makign from CHIME data using the m-mode formalism.

Tasks
=====

.. autosummary::
    :toctree: generated/

    FrequencyRebin
    SelectProducts
    MModeTransform
    MapMaker
"""
import numpy as np
from caput import pipeline, mpidataset, config, mpiutil

from ch_util import tools

from . import containers


def _make_marray(ts):

    mmodes = np.fft.fft(ts, axis=-1) / ts.shape[-1]

    marray = _pack_marray(mmodes)

    return marray


def _pack_marray(mmodes, mmax=None):

    if mmax is None:
        mmax = mmodes.shape[-1] / 2

    shape = mmodes.shape[:-1]

    marray = np.zeros((mmax+1, 2) + shape, dtype=np.complex128)

    marray[0, 0] = mmodes[..., 0]

    mlimit = min(mmax, mmodes.shape[-1] / 2)  # So as not to run off the end of the array
    for mi in range(1, mlimit - 1):
        marray[mi, 0] = mmodes[..., mi]
        marray[mi, 1] = mmodes[..., -mi].conj()

    return marray


def pinv_svd(M, acond=1e-4, rcond=1e-3):

    import scipy.linalg as la

    u, sig, vh = la.svd(M, full_matrices=False)

    rank = np.sum(np.logical_and(sig > rcond * sig.max(), sig > acond))

    psigma_diag = 1.0 / sig[: rank]

    B = np.transpose(np.conjugate(np.dot(u[:, : rank] * psigma_diag, vh[: rank])))

    return B


class FrequencyRebin(pipeline.TaskBase):
    """Rebin neighbouring frequency channels.

    Parameters
    ----------
    channel_bin : int
        Number of channels to in together.
    """

    channel_bin = config.Property(proptype=int, default=1)

    def next(self, ss):
        """Take the input dataset and rebin the frequencies.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sb : SiderealStream
        """

        if len(ss.freq) % self.channel_bin != 0:
            raise Exception("Binning must exactly divide the number of channels.")

        # Get all frequencies onto same node
        ss.redistribute(1)

        # Calculate the new frequency centres and widths
        fc = ss.freq['centre'].reshape(-1, self.channel_bin).mean(axis=-1)
        fw = ss.freq['width'].reshape(-1, self.channel_bin).sum(axis=-1)

        freq_map = np.empty(fc.shape[0], dtype=ss.freq.dtype)
        freq_map['centre'] = fc
        freq_map['width'] = fw

        # Rebin the weight array
        rshape = ss.vis.shape[1:]
        wt_rebin = ss.weight.view(np.ndarray).reshape((-1, self.channel_bin) + rshape).sum(axis=1)
        wt_rebin = mpidataset.MPIArray.wrap(wt_rebin, axis=1)

        # Rebin the visibility array
        vis_rebin = (ss.vis * ss.weight).view(np.ndarray).reshape((-1, self.channel_bin) + rshape).sum(axis=1)
        vis_rebin = np.where(wt_rebin == 0, np.zeros_like(vis_rebin), vis_rebin / wt_rebin)
        vis_rebin = mpidataset.MPIArray.wrap(vis_rebin, axis=1)

        # Construct the container for the rebinned timestream
        sb = containers.SiderealStream(ss.vis.global_shape[-1], freq_map, 1)
        sb.distributed['vis'] = vis_rebin
        sb.distributed['weight'] = wt_rebin
        sb.common['input'] = ss.input
        sb.attrs['tag'] = ss.attrs['tag']

        sb.redistribute(0)

        return sb


class SelectProducts(pipeline.TaskBase):
    """Extract and order the correlation products for map-making.

    The task will take a sidereal task and format the products that are needed
    or the map-making. It uses a BeamTransfer instance to figure out what these
    products are, and how they should be ordered. It similarly selects only the
    required frequencies.
    """

    def setup(self, bt):
        """Set the BeamTransfer instance to use.

        Parameters
        ----------
        bt : BeamTransfer
        """

        self.beamtransfer = bt
        self.telescope = bt.telescope

    def next(self, ss):
        """Select and reorder the products.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sp : SiderealStream
            Dataset containing only the required products.
        """

        ss.redistribute(0)

        ss_keys = ss.input['correlator_input']

        # Figure the mapping between inputs for the beam transfers and the file
        bt_feeds = self.telescope.feeds
        bt_keys = [ f.input_sn for f in bt_feeds ]

        input_ind = [np.nonzero(ss_keys == bk)[0][0] for bk in bt_keys]

        # Figure out mapping between the frequencies
        bt_freq = self.telescope.frequencies
        ss_freq = ss.freq['centre']

        freq_ind = [np.nonzero(ss_freq == bf)[0][0] for bf in bt_freq]

        if mpiutil.rank0:
            print input_ind
            print freq_ind

        sp_freq = ss.freq[freq_ind]
        sp_input = ss.input[input_ind]

        nfreq = len(sp_freq)
        nfeed = len(sp_input)

        sp = containers.SiderealStream(len(ss.ra), sp_freq, nfeed * (nfeed + 1) / 2, comm=ss.comm)

        if ss.weight is not None:
            sp.distributed['weight'] = mpidataset.MPIArray.wrap(np.zeros_like(sp.vis, dtype=np.float64), axis=0, comm=ss.comm)

        # Iterate over the selected frequencies and inputs and pull out the correct data
        for lfi, fi in sp.vis.enumerate(axis=0):

            lf = freq_ind[fi]

            for ii in range(nfeed):

                li = input_ind[ii]

                for ij in range(ii, nfeed):

                    lj = input_ind[ij]

                    sp_pi = tools.cmap(ii, ij, nfeed)
                    ss_pi = tools.cmap(li, lj, len(ss_keys))

                    if lj >= li:
                        sp.vis[lfi, sp_pi] = ss.vis[lf, ss_pi]
                    else:
                        sp.vis[lfi, sp_pi] = ss.vis[lf, ss_pi].conj()

                    if sp.weight is not None:
                        sp.weight[lfi, sp_pi] = ss.weight[lf, ss_pi]

        sp.common['input'] = sp_input

        return sp


class MModeTransform(pipeline.TaskBase):
    """Transform a sidereal stream to m-modes.

    Currently ignores any noise weighting.
    """

    def next(self, sstream):
        """Perform the m-mode transform.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        mmodes : containers.MModes
        """

        sstream.redistribute(axis=0)

        marray = _make_marray(sstream.vis)
        marray = mpidataset.MPIArray.wrap(marray, axis=2, comm=sstream.comm)

        ma = containers.MModes(1, sstream.freq, 1, comm=sstream.comm)

        ma._distributed['vis'] = marray
        ma.redistribute(axis=0)

        return ma


class MapMaker(pipeline.TaskBase):
    """Rudimetary m-mode map maker.

    Attributes
    ----------
    nside : int
        Resolution of output Healpix map.
    maptype : one of ['dirty', 'ml' 'wiener']
        What sort of map to make.
    baseline_mask : one of [ None, 'no_auto', 'no_intra']
        Whether to exclude any baselines in the estimation.
    prior_amp : float
        An amplitude prior to use for the Wiener filter map maker. In Kelvin.
    prior_tilt : float
        Power law index prior for the power spectrum, again for the Wiener filter.
    """

    nside = config.Property(proptype=int, default=256)
    maptype = config.Property(proptype=str, default='dirty')

    baseline_mask = config.Property(proptype=str, default=None)
    pol_mask = config.Property(proptype=str, default=None)
    m_mask = config.Property(proptype=str, default='no_m_zero')

    prior_amp = config.Property(proptype=float, default=1.0)
    prior_tilt = config.Property(proptype=float, default=0.5)


    def setup(self, bt):
        """Set the beamtransfer matrices to use.

        Parameters
        ----------
        bt : beamtransfer.BeamTransfer
            Beam transfer manager object containing all the pre-generated beam
            transfer matrices.
        """

        self.beamtransfer = bt

    def _noise_weight(self, m):
        # Construct the noise weighting for the data. Returns an estimate of
        # the inverse noise for each baseline and frequency (assumes no
        # correlations), this is used to apply the masking of unwanted
        # baselines.

        tel = self.beamtransfer.telescope
        nw = 1.0 / tel.noisepower(np.arange(tel.nbase)[np.newaxis, :],
                                  np.arange(tel.nfreq)[:, np.newaxis], ndays=1)

        mask = np.ones(tel.nbase)

        # Mask out auto correlations
        if self.baseline_mask == 'no_auto':
            for fi in range(tel.nfeed):
                mask[tools.cmap(fi, fi, tel.nfeed)] = 0

        # Mask out intracylinder correlations
        elif self.baseline_mask == 'no_intra':
            for pi in range(tel.nbase):

                fi, fj = tools.icmap(pi, tel.nfeed)

                if tel.feeds[fi].cyl == tel.feeds[fj].cyl:
                    mask[pi] = 0

        if self.pol_mask == 'x_only':
            for pi in range(tel.nbase):

                fi, fj = tools.icmap(pi, tel.nfeed)

                if tools.is_chime_y(tel.feeds[fi]) or tools.is_chime_y(tel.feeds[fj]):
                    mask[pi] = 0

        elif self.pol_mask == 'y_only':
            for pi in range(tel.nbase):

                fi, fj = tools.icmap(pi, tel.nfeed)

                if tools.is_chime_x(tel.feeds[fi]) or tools.is_chime_x(tel.feeds[fj]):
                    mask[pi] = 0

        if ((self.m_mask == 'no_m_zero' and m == 0) or
            (self.m_mask == 'positive_only' and m <= 0) or
            (self.m_mask == 'negative_only' and m > 0)):
            nw[:] = 0.0

        nw = nw * mask[np.newaxis, :]

        # Concatenate the noise weight to take into account positivie and negative m's.
        nw = np.concatenate([nw, nw], axis=1)

        return nw

    def _dirty_proj(self, m):

        bt = self.beamtransfer
        nw = self._noise_weight(m)

        bm = bt.beam_m(m).reshape(bt.nfreq, bt.ntel, bt.nsky)
        db = bm.transpose((0, 2, 1)).conj() * nw[:, np.newaxis, :]

        return db

    def _ml_proj(self, m):

        bt = self.beamtransfer
        nw = self._noise_weight(m)

        bm = bt.beam_m(m).reshape(bt.nfreq, bt.ntel, bt.nsky)

        ib = np.zeros((bt.nfreq, bt.nsky, bt.ntel), dtype=np.complex128)

        for fi in range(bt.nfreq):
            nh = nw[fi]**0.5
            ib[fi] = pinv_svd(bm[fi] * nh[:, np.newaxis]) * nh[np.newaxis, :]

        #ib = bt.invbeam_m(m).reshape(bt.nfreq, bt.nsky, bt.ntel)

        return ib

    def _wiener_proj_cl(self, m):

        import scipy.linalg as la
        bt = self.beamtransfer
        nw = self._noise_weight(m)
        nh = nw**0.5

        bmt = bt.beam_m(m).reshape(bt.nfreq, bt.ntel, bt.nsky) * nh[:, :, np.newaxis]
        bth = bmt.transpose((0, 2, 1)).conj()

        wb = np.zeros((bt.nfreq, bt.nsky, bt.ntel), dtype=np.complex128)

        l = np.arange(bt.telescope.lmax + 1)
        l[0] = 1
        cl_TT = self.prior_amp**2 * l**(-self.prior_tilt)
        S = np.concatenate([cl_TT] * 4)

        for fi in range(bt.nfreq):

            if bt.ntel > bt.nsky:
                mat = np.diag(1.0 / S) + np.dot(bth[fi], bmt[fi])
                print la.eigvalsh(mat)
                wb[fi] = np.dot(la.inv(mat), bth[fi] * nh[fi, np.newaxis, :])
            else:
                mat = np.identity(bt.ntel) + np.dot(bmt[fi] * S[np.newaxis, :], bth[fi])
                wb[fi] = S[:, np.newaxis] * np.dot(bth[fi], la.inv(mat)) * nh[fi, np.newaxis, :]

        #ib = bt.invbeam_m(m).reshape(bt.nfreq, bt.nsky, bt.ntel)

        return wb

    def _proj(self, m):
        # Return approproate projection matrix depending on value of maptype

        proj_calltable = {'dirty': self._dirty_proj,
                          'ml': self._ml_proj,
                          'wiener': self._wiener_proj_cl}

        if self.maptype not in proj_calltable.keys():
            raise Exception("Map type not known.")

        return proj_calltable[self.maptype](m)

    def next(self, mmodes):
        """Make a map from the given m-modes.

        Parameters
        ----------
        mmodes : containers.MModes

        Returns
        -------
        map : containers.Map
        """

        from cora.util import hputil

        # Fetch various properties
        bt = self.beamtransfer
        lmax = bt.telescope.lmax
        mmax = bt.telescope.mmax
        nfreq = bt.telescope.nfreq

        # Trim off excess m-modes
        mmodes.redistribute(axis=2)
        mmodes.distributed['vis'] = mpidataset.MPIArray.wrap(mmodes.vis[:(mmax+1)], axis=2, comm=mmodes.comm)
        mmodes.redistribute(axis=0)

        # Create array to store alms in.
        alm = mpidataset.MPIArray((nfreq, 4, lmax+1, mmax+1), dtype=np.complex128, axis=3, comm=mmodes.comm)
        alm[:] = 0.0

        # Calculate which m's are local
        ml = mmodes.vis.local_offset[0]  # Lowest m
        mh = ml + mmodes.vis.local_shape[0]  # Highest m

        # Loop over all m's and project from m-mode visibilities to alms.
        for mi, m in enumerate(range(ml, mh)):

            pm = self._proj(m)

            for fi in range(nfreq):
                alm[fi, ..., mi] = np.dot(pm[fi], mmodes.vis[mi, :, fi].flatten()).reshape(4, lmax+1)

        # Redistribute back over frequency
        alm = alm.redistribute(axis=0)

        # Copy into square alm array for transform
        almt = mpidataset.MPIArray((nfreq, 4, lmax+1, lmax+1), dtype=np.complex128, axis=0, comm=mmodes.comm)
        almt[..., :(mmax+1)] = alm
        alm = almt

        # Perform spherical harmonic transform to map space
        maps = hputil.sphtrans_inv_sky(alm, self.nside)
        maps = mpidataset.MPIArray.wrap(maps, axis=0)

        m = containers.Map(comm=mmodes.comm)
        m._distributed['map'] = maps

        return m

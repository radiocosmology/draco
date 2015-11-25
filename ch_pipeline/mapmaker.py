"""
===============================================
Map making tasks (:mod:`~ch_pipeline.mapmaker`)
===============================================

.. currentmodule:: ch_pipeline.mapmaker

Tools for map making from CHIME data using the m-mode formalism.

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
from caput import mpiarray, config

from ch_util import tools, andata

from . import containers, task


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


class FrequencyRebin(task.SingleTask):
    """Rebin neighbouring frequency channels.

    Parameters
    ----------
    channel_bin : int
        Number of channels to in together.
    """

    channel_bin = config.Property(proptype=int, default=1)

    def process(self, ss):
        """Take the input dataset and rebin the frequencies.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sb : SiderealStream
        """

        if 'freq' not in ss.index_map:
            raise RuntimeError('Data does not have a frequency axis.')

        if len(ss.freq) % self.channel_bin != 0:
            raise RuntimeError("Binning must exactly divide the number of channels.")

        # Get all frequencies onto same node
        #ss.redistribute(['prod', 'input'])
        ss.redistribute('time')

        # Calculate the new frequency centres and widths
        fc = ss.index_map['freq']['centre'].reshape(-1, self.channel_bin).mean(axis=-1)
        fw = ss.index_map['freq']['width'].reshape(-1, self.channel_bin).sum(axis=-1)

        freq_map = np.empty(fc.shape[0], dtype=ss.index_map['freq'].dtype)
        freq_map['centre'] = fc
        freq_map['width'] = fw

        # Create new container for rebinned stream
        if isinstance(ss, containers.ContainerBase):
            sb = ss.__class__(freq=freq_map, axes_from=ss)
        elif isinstance(ss, andata.CorrData):
            sb = containers.make_empty_corrdata(freq=freq_map, axes_from=ss, distributed=True,
                                                distributed_axis=2, comm=ss.comm)
        else:
            raise RuntimeError("I don't know how to deal with data type %s" % ss.__class__.__name__)

        # Get all frequencies onto same node
        sb.redistribute('time')

        # Copy over the tag attribute
        sb.attrs['tag'] = ss.attrs['tag']

        # Rebin the arrays, do this with a loop to save memory
        for fi in range(len(ss.freq)):

            # Calculate rebinned index
            ri = fi / self.channel_bin

            sb.vis[ri] += ss.vis[fi] * ss.weight[fi]
            sb.gain[ri] += ss.gain[fi] / self.channel_bin  # Don't do weighted average for the moment

            sb.weight[ri] += ss.weight[fi]

            # If we are on the final sub-channel then divide the arrays through
            if (fi + 1) % self.channel_bin == 0:
                sb.vis[ri] *= tools.invert_no_zero(sb.weight[ri])

        sb.redistribute('freq')

        return sb


class SelectProducts(task.SingleTask):
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

    def process(self, ss):
        """Select and reorder the products.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sp : SiderealStream
            Dataset containing only the required products.
        """

        ss_keys = ss.index_map['input'][:]

        # Figure the mapping between inputs for the beam transfers and the file
        # bt_feeds = self.telescope.feeds
        # bt_keys = [ f.input_sn for f in bt_feeds ]
        try:
            bt_keys = self.telescope.feed_index
        except AttributeError:
            bt_keys = np.arange(self.telescope.nfeed)

        input_ind = [np.nonzero(ss_keys == bk)[0][0] for bk in bt_keys]

        # Figure out mapping between the frequencies
        bt_freq = self.telescope.frequencies
        ss_freq = ss.freq['centre']

        freq_ind = [np.nonzero(ss_freq == bf)[0][0] for bf in bt_freq]

        sp_freq = ss.freq[freq_ind]
        sp_input = ss.input[input_ind]

        nfreq = len(sp_freq)
        nfeed = len(sp_input)

        sp = containers.SiderealStream(freq=sp_freq, input=sp_input, axes_from=ss,
                                       distributed=True, comm=ss.comm)

        # Ensure all frequencies and products are on each node
        ss.redistribute('ra')
        sp.redistribute('ra')

        # Iterate over the selected frequencies and inputs and pull out the correct data
        for fi in range(nfreq):

            lf = freq_ind[fi]

            for ii in range(nfeed):

                li = input_ind[ii]

                for ij in range(ii, nfeed):

                    lj = input_ind[ij]

                    sp_pi = tools.cmap(ii, ij, nfeed)
                    ss_pi = tools.cmap(li, lj, len(ss_keys))

                    if lj >= li:
                        sp.vis[fi, sp_pi] = ss.vis[lf, ss_pi]
                    else:
                        sp.vis[fi, sp_pi] = ss.vis[lf, ss_pi].conj()

                    if sp.weight is not None:
                        sp.weight[fi, sp_pi] = ss.weight[lf, ss_pi]

        # Switch back to frequency distribution
        ss.redistribute('freq')
        sp.redistribute('freq')

        return sp


class SelectProductsRedundant(task.SingleTask):
    """Extract and order the correlation products for map-making.

    The task will take a sidereal task and format the products that are needed
    or the map-making. It uses a BeamTransfer instance to figure out what these
    products are, and how they should be ordered. It similarly selects only the
    required frequencies.

    It is important to note that while the input
    :class:`SiderealStream` can contain more feeds and frequencies
    than are contained in the BeamTransfers, the converse is not
    true. That is, all the frequencies and feeds that are in the
    BeamTransfers must be found in the timestream object.
    """

    def setup(self, bt):
        """Set the BeamTransfer instance to use.

        Parameters
        ----------
        bt : BeamTransfer
        """

        self.beamtransfer = bt
        self.telescope = bt.telescope

    def process(self, ss):
        """Select and reorder the products.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sp : SiderealStream
            Dataset containing only the required products.
        """

        ss_keys = ss.index_map['input'][:]

        # Figure the mapping between inputs for the beam transfers and the file
        try:
            bt_keys = self.telescope.feed_index
        except AttributeError:
            bt_keys = np.arange(self.telescope.nfeed)

        def find_key(key_list, key):
            try:
                return map(tuple, list(key_list)).index(tuple(key))
            except TypeError:
                return list(key_list).index(key)
            except ValueError:
                return None

        input_ind = [ find_key(bt_keys, sk) for sk in ss_keys]

        # Figure out mapping between the frequencies
        bt_freq = self.telescope.frequencies
        ss_freq = ss.freq['centre']

        freq_ind = [ find_key(ss_freq, bf) for bf in bt_freq]

        sp_freq = ss.freq[freq_ind]
        # sp_input = ss.input[input_ind

        # sp = containers.SiderealStream(freq=sp_freq, input=sp_input, prod=self.telescope.uniquepairs,
        sp = containers.SiderealStream(freq=sp_freq, input=len(bt_keys), prod=self.telescope.uniquepairs,
                                       axes_from=ss, attrs_from=ss, distributed=True, comm=ss.comm)

        # Ensure all frequencies and products are on each node
        ss.redistribute('ra')
        sp.redistribute('ra')

        sp.vis[:] = 0.0
        sp.weight[:] = 0.0

        # Iterate over products in the sidereal stream
        for ss_pi in range(len(ss.index_map['prod'])):

            #if ss.vis.comm.rank == 0:
            #if ss_pi % 100 == 0:
            #    print "Progress", ss.vis.comm.rank, ss_pi

            # Get the feed indices for this product
            ii, ij = ss.index_map['prod'][ss_pi]

            # Map the feed indices into ones for the Telescope class
            bi, bj = input_ind[ii], input_ind[ij]

            # If either feed is not in the telescope class, skip it.
            if bi is None or bj is None:
                continue

            sp_pi = self.telescope.feedmap[bi, bj]
            feedconj = self.telescope.feedconj[bi, bj]

            # Skip if product index is not valid
            if sp_pi < 0:
                continue

            # Accumulate visibilities, conjugating if required
            if not feedconj:
                sp.vis[:, sp_pi] += ss.weight[freq_ind, ss_pi] * ss.vis[freq_ind, ss_pi]
            else:
                sp.vis[:, sp_pi] += ss.weight[freq_ind, ss_pi] * ss.vis[freq_ind, ss_pi].conj()

            # Accumulate weights
            sp.weight[:, sp_pi] += ss.weight[freq_ind, ss_pi]

        # Divide through by weights to get properly weighted visibility average
        sp.vis[:] *= tools.invert_no_zero(sp.weight[:])

        # Switch back to frequency distribution
        ss.redistribute('freq')
        sp.redistribute('freq')

        return sp


class SelectFreq(task.SingleTask):
    """Select a subset of frequencies from the data.

    Attributes
    ----------
    frequencies : list
        List of frequency indices.
    """

    frequencies = config.Property(proptype=list)

    def process(self, data):
        """Selet a subset of the frequencies.

        Parameters
        ----------
        data : containers.ContainerBase
            A data container with a frequency axis.

        Returns
        -------
        newdata : containers.ContainerBase
            New container with trimmed frequencies.
        """

        freq_map = data.index_map['freq'][self.frequencies]
        data.redistribute(['ra', 'time'])

        newdata = data.__class__(freq=freq_map, axes_from=data)
        newdata.redistribute(['ra', 'time'])

        for name, dset in data.datasets.items():

            if 'freq' in dset.attrs['axis']:
                newdata.datasets[name][:] = data.datasets[name][self.frequencies, ...]
            else:
                newdata.datasets[name][:] = data.datasets[name][:]

        return newdata


class MModeTransform(task.SingleTask):
    """Transform a sidereal stream to m-modes.

    Currently ignores any noise weighting.
    """

    def process(self, sstream):
        """Perform the m-mode transform.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        mmodes : containers.MModes
        """

        sstream.redistribute('freq')

        marray = _make_marray(sstream.vis[:])
        marray = mpiarray.MPIArray.wrap(marray[:], axis=2, comm=sstream.comm)

        mmax = marray.shape[0] - 1

        ma = containers.MModes(mmax=mmax, axes_from=sstream, comm=sstream.comm)
        ma.redistribute('freq')

        ma.vis[:] = marray
        ma.redistribute('m')

        return ma


class MapMaker(task.SingleTask):
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
            for pi in range(tel.nbase):

                fi, fj = tel.uniquepairs[pi]

                if fi == fj:
                    mask[pi] = 0

        # Mask out intracylinder correlations
        elif self.baseline_mask == 'no_intra':
            for pi in range(tel.nbase):

                fi, fj = tel.uniquepairs[pi]

                if tel.feeds[fi].cyl == tel.feeds[fj].cyl:
                    mask[pi] = 0

        if self.pol_mask == 'x_only':
            for pi in range(tel.nbase):

                fi, fj = tel.uniquepairs[pi]

                if tools.is_chime_y(tel.feeds[fi]) or tools.is_chime_y(tel.feeds[fj]):
                    mask[pi] = 0

        elif self.pol_mask == 'y_only':
            for pi in range(tel.nbase):

                fi, fj = tel.uniquepairs[pi]

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

    def _dirty_proj(self, m, f):

        bt = self.beamtransfer
        nw = self._noise_weight(m)

        bm = bt.beam_m(m, fi=f).reshape(bt.ntel, bt.nsky)
        db = bm.T.conj() * nw[f, np.newaxis, :]

        return db

    def _ml_proj(self, m, f):

        bt = self.beamtransfer
        nw = self._noise_weight(m)

        bm = bt.beam_m(m, fi=f).reshape(bt.ntel, bt.nsky)

        nh = nw[f]**0.5
        ib = pinv_svd(bm * nh[:, np.newaxis]) * nh[np.newaxis, :]

        return ib

    def _wiener_proj_cl(self, m, f):

        import scipy.linalg as la
        bt = self.beamtransfer
        nw = self._noise_weight(m)
        nh = nw**0.5

        bmt = bt.beam_m(m, fi=f).reshape(bt.ntel, bt.nsky) * nh[f, :, np.newaxis]
        bth = bmt.T.conj()

        wb = np.zeros((bt.nsky, bt.ntel), dtype=np.complex128)

        l = np.arange(bt.telescope.lmax + 1)
        l[0] = 1
        cl_TT = self.prior_amp**2 * l**(-self.prior_tilt)
        S = np.concatenate([cl_TT] * 4)

        if bt.ntel > bt.nsky:
            mat = np.diag(1.0 / S) + np.dot(bth, bmt)
            wb = np.dot(la.inv(mat), bth * nh[f, np.newaxis, :])
        else:
            mat = np.identity(bt.ntel) + np.dot(bmt * S[np.newaxis, :], bth)
            wb = S[:, np.newaxis] * np.dot(bth, la.inv(mat)) * nh[f, np.newaxis, :]

        return wb

    # def _wiener_proj_cl(self, m, f):
    #
    #     import scipy.linalg as la
    #     bt = self.beamtransfer
    #     nw = self._noise_weight(m)
    #     nh = nw**0.5
    #
    #     bmt = bt.beam_m(m).reshape(bt.nfreq, bt.ntel, bt.nsky) * nh[:, :, np.newaxis]
    #     bth = bmt.transpose((0, 2, 1)).conj()
    #
    #     wb = np.zeros((bt.nfreq, bt.nsky, bt.ntel), dtype=np.complex128)
    #
    #     l = np.arange(bt.telescope.lmax + 1)
    #     l[0] = 1
    #     cl_TT = self.prior_amp**2 * l**(-self.prior_tilt)
    #     S = np.concatenate([cl_TT] * 4)
    #
    #     for fi in range(bt.nfreq):
    #
    #         if bt.ntel > bt.nsky:
    #             mat = np.diag(1.0 / S) + np.dot(bth[fi], bmt[fi])
    #             print la.eigvalsh(mat)
    #             wb[fi] = np.dot(la.inv(mat), bth[fi] * nh[fi, np.newaxis, :])
    #         else:
    #             mat = np.identity(bt.ntel) + np.dot(bmt[fi] * S[np.newaxis, :], bth[fi])
    #             wb[fi] = S[:, np.newaxis] * np.dot(bth[fi], la.inv(mat)) * nh[fi, np.newaxis, :]
    #
    #     #ib = bt.invbeam_m(m).reshape(bt.nfreq, bt.nsky, bt.ntel)
    #
    #     return wb

    def _proj(self, *args):
        # Return approproate projection matrix depending on value of maptype

        proj_calltable = {'dirty': self._dirty_proj,
                          'ml': self._ml_proj,
                          'wiener': self._wiener_proj_cl}

        if self.maptype not in proj_calltable.keys():
            raise Exception("Map type not known.")

        return proj_calltable[self.maptype](*args)

    def process(self, mmodes):
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
        mmax = min(bt.telescope.mmax, len(mmodes.index_map['m']) - 1)
        nfreq = bt.telescope.nfreq

        # Trim off excess m-modes
        mmodes.redistribute('freq')
        m_array = mmodes.vis[:(mmax+1)]
        m_array = m_array.redistribute(axis=0)

        # Create array to store alms in.
        alm = mpiarray.MPIArray((nfreq, 4, lmax+1, mmax+1), dtype=np.complex128, axis=3, comm=mmodes.comm)
        alm[:] = 0.0

        # Loop over all m's and project from m-mode visibilities to alms.
        for mi, m in m_array.enumerate(axis=0):

            for fi in range(nfreq):
                pm = self._proj(m, fi)
                alm[fi, ..., mi] = np.dot(pm, m_array[mi, :, fi].flatten()).reshape(4, lmax+1)

        # Redistribute back over frequency
        alm = alm.redistribute(axis=0)

        # Copy into square alm array for transform
        almt = mpiarray.MPIArray((nfreq, 4, lmax+1, lmax+1), dtype=np.complex128, axis=0, comm=mmodes.comm)
        almt[..., :(mmax+1)] = alm
        alm = almt

        # Perform spherical harmonic transform to map space
        maps = hputil.sphtrans_inv_sky(alm, self.nside)
        maps = mpiarray.MPIArray.wrap(maps, axis=0)

        m = containers.Map(nside=self.nside, axes_from=mmodes, comm=mmodes.comm)
        m.map[:] = maps

        return m


class RingMapMaker(task.SingleTask):
    """A simple and quick map-maker that forms a series of beams on the meridian.

    This is designed to run on data after it has been collapsed down to
    non-redundant baselines only.

    Attributes
    ----------
    weighting : one of ['natural']
    """

    def setup(self, bt):
        """Set the beamtransfer matrices to use.

        Parameters
        ----------
        bt : beamtransfer.BeamTransfer
            Beam transfer manager object. This does not need to have
            pre-generated matrices as they are not needed.
        """

        self.beamtransfer = bt

    def process(self, sstream):
        """Perform the m-mode transform.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        bfmaps : containers.RingMap
        """

        tel = self.beamtransfer.telescope

        # Redistribute over frequency
        sstream.redistribute('freq')

        nfreq = sstream.vis.local_shape[0]
        nra = len(sstream.ra)
        nfeed = 64  # Fixed for pathfinder
        ncyl = 2
        sp = 0.3048
        nvis_1d = 2 * nfeed - 1
        n_el = 1024

        # Construct mapping from vis array to unpacked 2D grid
        feed_list = [ (tel.feeds[fi], tel.feeds[fj]) for fi, fj in sstream.index_map['prod'][:]]
        feed_ind = [ ( 2 * int(fi.pol == 'S') + int(fj.pol == 'S'),
                       fi.cyl - fj.cyl, int(np.round((fi.pos - fj.pos) / sp))) for fi, fj in feed_list]

        # Empty array for output
        vdr = np.zeros((nfreq, 4, nra, ncyl, nvis_1d), dtype=np.complex128)

        # Unpack visibilities into new array
        for vis_ind, ind in enumerate(feed_ind):

            p_ind, x_ind, y_ind = ind

            w = tel.redundancy[vis_ind]

            if x_ind == 0:
                vdr[:, p_ind, :, x_ind, y_ind] = w * sstream.vis[:, vis_ind]
                vdr[:, p_ind, :, x_ind, -y_ind] = w * sstream.vis[:, vis_ind].conj()
            else:
                vdr[:, p_ind, :, x_ind, y_ind] = w * sstream.vis[:, vis_ind]

        # Remove auto-correlations
        vdr[..., 0, 0] = 0.0

        # Construct phase array
        sin_el = np.linspace(-1.0, 1.0, n_el)
        vis_pos_1d = np.fft.fftfreq(nvis_1d, d=(1.0 / (nvis_1d * sp)))

        # Create empty ring map
        rm = containers.RingMap(beam=(2 * ncyl - 1), el=n_el, polarisation=True, axes_from=sstream)
        rm.redistribute('freq')

        print rm.map

        for lfi, fi in sstream.vis[:].enumerate(0):

            # Get the current freq (this try... except clause can be removed,
            # its just to workaround a now fixed bug in SelectProductsRedundant)
            try:
                fr = sstream.freq['centre'][fi]
            except:
                fr = np.linspace(800.0, 400.0, 1024, endpoint=True).reshape(-1, 4).mean(axis=1)[sstream.freq[fi]]

            wv = 3e2 / fr

            pa = np.exp(-2.0J * np.pi * vis_pos_1d[np.newaxis, :] * sin_el[:, np.newaxis] / wv)

            bfm = np.fft.irfft(np.dot(vdr[lfi], pa.T.conj()), 2 * ncyl - 1, axis=2)
            rm.map[fi] = bfm

        return rm

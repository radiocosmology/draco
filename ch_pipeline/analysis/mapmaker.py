"""
========================================================
Map making tasks (:mod:`~ch_pipeline.analysis.mapmaker`)
========================================================

.. currentmodule:: ch_pipeline.anaysis.mapmaker

Tools for map making from CHIME data using the m-mode formalism.

Tasks
=====

.. autosummary::
    :toctree: generated/

    FrequencyRebin
    CollateProducts
    MModeTransform
    MaskData
    MaskCHIMEData
    DirtyMapMaker
    MaximumLikelihoodMapMaker
    WienerMapMaker
    RingMapMaker
"""
import numpy as np
from caput import mpiarray, config

from ch_util import tools, andata

from ..core import containers, task


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


class CollateProducts(task.SingleTask):
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
            bt_keys = self.telescope.feeds
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

        sp = containers.SiderealStream(
            freq=sp_freq, input=len(bt_keys), prod=self.telescope.uniquepairs,
            axes_from=ss, attrs_from=ss, distributed=True, comm=ss.comm
        )

        # Ensure all frequencies and products are on each node
        ss.redistribute('ra')
        sp.redistribute('ra')

        sp.vis[:] = 0.0
        sp.weight[:] = 0.0

        # Iterate over products in the sidereal stream
        for ss_pi in range(len(ss.index_map['prod'])):

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

        newdata = data.__class__(freq=freq_map, axes_from=data, attrs_from=data)
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

        # Sum the noise variance over time samples, this will become the noise
        # variance for the m-modes
        weight_sum = sstream.weight[:].sum(axis=-1)

        # Construct the array of m-modes
        marray = _make_marray(sstream.vis[:])
        marray = mpiarray.MPIArray.wrap(marray[:], axis=2, comm=sstream.comm)

        # Create the container to store the modes in
        mmax = marray.shape[0] - 1
        ma = containers.MModes(mmax=mmax, axes_from=sstream, comm=sstream.comm)
        ma.redistribute('freq')

        # Assign the visibilities and weights into the container
        ma.vis[:] = marray
        ma.weight[:] = weight_sum[:, :, np.newaxis]

        ma.redistribute('m')

        return ma


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


class MaskCHIMEData(task.SingleTask):
    """Mask out data ahead of map making.

    Attributes
    ----------
    intra_cylinder : bool
        Include baselines within the same cylinder (default=True).
    xx_pol : bool
        Include X-polarisation (default=True).
    yy no_pol : bool
        Include Y-polarisation (default=True).
    cross_pol : bool
        Include cross-polarisation (default=True).
    """

    intra_cylinder = config.Property(proptype=bool, default=True)

    xx_pol = config.Property(proptype=bool, default=True)
    yy_pol = config.Property(proptype=bool, default=True)
    cross_pol = config.Property(proptype=bool, default=True)

    def setup(self, tel):
        """Setup the task.

        Parameters
        ----------
        tel : :class:`ch_pipeline.core.pathfinder.CHIMEPathfinder`
            CHIME telescope class to use to get feed information.
        """
        self.telescope = tel

    def process(self, mmodes):
        """Mask out unwanted datain the m-modes.

        Parameters
        ----------
        mmodes : containers.MModes

        Returns
        -------
        mmodes : containers.MModes
        """

        tel = self.telescope

        for pi, (fi, fj) in enumerate(mmodes.index_map['prod']):

            oi, oj = tel.feeds[fi], tel.feeds[fj]

            # Check if baseline is intra-cylinder
            if not self.intra_cylinder and (oi.cyl == oj.cyl):
                mmodes.weight[..., pi] = 0.0

            # Check all the polarisation states
            is_xx = tools.is_chime_x(oi) and tools.is_chime_x(oj)
            is_yy = tools.is_chime_y(oi) and tools.is_chime_y(oj)

            if not self.xx_pol and is_xx:
                mmodes.weight[..., pi] = 0.0

            if not self.yy_pol and is_yy:
                mmodes.weight[..., pi] = 0.0

            if not self.cross_pol and not (is_xx or is_yy):
                mmodes.weight[..., pi] = 0.0

        return mmodes


class BaseMapMaker(task.SingleTask):
    """Rudimetary m-mode map maker.

    Attributes
    ----------
    nside : int
        Resolution of output Healpix map.
    """

    nside = config.Property(proptype=int, default=256)

    def setup(self, bt):
        """Set the beamtransfer matrices to use.

        Parameters
        ----------
        bt : beamtransfer.BeamTransfer
            Beam transfer manager object containing all the pre-generated beam
            transfer matrices.
        """

        self.beamtransfer = bt

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
        nfreq = len(mmodes.index_map['freq'])  # bt.telescope.nfreq

        def find_key(key_list, key):
            try:
                return map(tuple, list(key_list)).index(tuple(key))
            except TypeError:
                return list(key_list).index(key)
            except ValueError:
                return None

        # Figure out mapping between the frequencies
        bt_freq = self.beamtransfer.telescope.frequencies
        mm_freq = mmodes.index_map['freq']['centre']

        freq_ind = [ find_key(bt_freq, mf) for mf in mm_freq]

        # Trim off excess m-modes
        mmodes.redistribute('freq')
        m_array = mmodes.vis[:(mmax + 1)]
        m_array = m_array.redistribute(axis=0)

        m_weight = mmodes.weight[:(mmax + 1)]
        m_weight = m_weight.redistribute(axis=0)

        # Create array to store alms in.
        alm = mpiarray.MPIArray((nfreq, 4, lmax + 1, mmax + 1), axis=3,
                                dtype=np.complex128, comm=mmodes.comm)
        alm[:] = 0.0

        # Loop over all m's and solve from m-mode visibilities to alms.
        for mi, m in m_array.enumerate(axis=0):

            for fi in range(nfreq):
                v = m_array[mi, :, fi]
                a = alm[fi, ..., mi].view(np.ndarray)
                Ni = m_weight[mi, :, fi]

                a[:] = self._solve_m(m, fi, v, Ni)

        # Redistribute back over frequency
        alm = alm.redistribute(axis=0)

        # Copy into square alm array for transform
        almt = mpiarray.MPIArray((nfreq, 4, lmax + 1, lmax + 1), dtype=np.complex128, axis=0, comm=mmodes.comm)
        almt[..., :(mmax + 1)] = alm
        alm = almt

        # Perform spherical harmonic transform to map space
        maps = hputil.sphtrans_inv_sky(alm, self.nside)
        maps = mpiarray.MPIArray.wrap(maps, axis=0)

        m = containers.Map(nside=self.nside, axes_from=mmodes, comm=mmodes.comm)
        m.map[:] = maps

        return m

    def _solve_m(self, m, f, v, Ni):
        """Solve for the a_lm's.

        This implementation is blank. Must be overriden.

        Parameters
        ----------
        m : int
            Which m-mode are we solving for.
        f : int
            Frequency we are solving for.
        v : np.ndarray[2, nbase]
            Visibility data.
        Ni : np.ndarray[2, nbase]
            Inverse of noise variance. Used as the noise matrix for the solve.

        Returns
        -------
        a : np.ndarray[npol, lmax+1]
        """
        pass


class DirtyMapMaker(BaseMapMaker):
    """Generate a dirty map.

    Notes
    -----

    The dirty map is produced by generating a set of :math:`a_{lm}` coefficients
    using

    .. math:: \hat{\mathbf{a}} = \mathbf{B}^\dagger \mathbf{N}^{-1} \mathbf{v}

    and then performing the spherical harmonic transform to get the sky intensity.
    """

    def _solve_m(self, m, f, v, Ni):

        bt = self.beamtransfer

        # Massage the arrays into shape
        v = v.reshape(bt.ntel)
        Ni = Ni.reshape(bt.ntel)
        bm = bt.beam_m(m, fi=f).reshape(bt.ntel, bt.nsky)

        # Solve for the dirty map alms
        a = np.dot(bm.T.conj(), Ni * v)

        # Reshape to the correct output
        a = a.reshape(bt.npol, bt.lmax + 1)

        return a


class MaximumLikelihoodMapMaker(BaseMapMaker):
    """Generate a Maximum Likelihood map using the Moore-Penrose pseudo-inverse.

    Notes
    -----

    The dirty map is produced by generating a set of :math:`a_{lm}` coefficients
    using

    .. math:: \hat{\mathbf{a}} = \left( \mathbf{N}^{-1/2 }\mathbf{B} \right)^+ \mathbf{N}^{-1/2} \mathbf{v}

    where the superscript :math:`+` denotes the pseudo-inverse.
    """

    def _solve_m(self, m, f, v, Ni):

        bt = self.beamtransfer

        # Massage the arrays into shape
        v = v.reshape(bt.ntel)
        Ni = Ni.reshape(bt.ntel)
        bm = bt.beam_m(m, fi=f).reshape(bt.ntel, bt.nsky)

        Nh = Ni**0.5

        # Construct the beam pseudo inverse
        ib = pinv_svd(bm * Nh[:, np.newaxis])

        # Solve for the ML map alms
        a = np.dot(ib, Nh * v)

        # Reshape to the correct output
        a = a.reshape(bt.npol, bt.lmax + 1)

        return a


class WienerMapMaker(BaseMapMaker):
    """Generate a Wiener filtered map assuming that the signal is a Gaussian
    random field described by a power-law power spectum.

    Attributes
    ----------
    prior_amp : float
        An amplitude prior to use for the map maker. In Kelvin.
    prior_tilt : float
        Power law index prior for the power spectrum.

    Notes
    -----

    The Wiener map is produced by generating a set of :math:`a_{lm}` coefficients
    using

    .. math::
        \hat{\mathbf{a}} = \left( \mathbf{S}^{-1} + \mathbf{B}^\dagger
        \mathbf{N}^{-1} \mathbf{B} \right)^{-1} \mathbf{B}^\dagger \mathbf{N}^{-1} \mathbf{v}

    where the signal covariance matrix :math:`\mathbf{S}` is assumed to be
    governed by a power law power spectrum for each polarisation component.
    """

    prior_amp = config.Property(proptype=float, default=1.0)
    prior_tilt = config.Property(proptype=float, default=0.5)

    def _solve_m(self, m, f, v, Ni):

        import scipy.linalg as la

        bt = self.beamtransfer

        # Massage the arrays into shape
        v = v.reshape(bt.ntel)
        Ni = Ni.reshape(bt.ntel)
        Nh = Ni**0.5

        # Get the beam transfer matrix, but trim off any l < m.
        bm = bt.beam_m(m, fi=f)[..., m:].reshape(bt.ntel, -1)  # No

        # Construct pre-wightened beam and beam-conjugated matrices
        bmt = bm * Nh[:, np.newaxis]
        bth = bmt.T.conj()

        # Pre-wighten the visibilities
        vt = Nh * v

        # Construct the signal covariance matrix
        l = np.arange(bt.telescope.lmax + 1)
        l[0] = 1  # Change l=0 to get around singularity
        l = l[m:]  # Trim off any l < m
        cl_TT = self.prior_amp**2 * l**(-self.prior_tilt)
        S_diag = np.concatenate([cl_TT] * 4)

        # For large ntel it's quickest to solve in the standard Wiener filter way
        if bt.ntel > bt.nsky:
            Ci = np.diag(1.0 / S_diag) + np.dot(bth, bmt)  # Construct the inverse covariance
            a_dirty = np.dot(bth, vt)  # Find the dirty map
            a_wiener = la.solve(Ci, a_dirty, sym_pos=True)  # Solve to find C vt

        # If not it's better to rearrange using the results for blockwise matrix inversion
        else:
            pCi = np.identity(bt.ntel) + np.dot(bmt * S_diag[np.newaxis, :], bth)
            v_int = la.solve(pCi, vt, sym_pos=True)
            a_wiener = S_diag * np.dot(bth, v_int)

        # Copy the solution into a correctly shaped array output
        a = np.zeros((bt.npol, bt.lmax + 1), dtype=v.dtype)
        a[:, m:] = a_wiener.reshape(bt.npol, -1)

        return a


class RingMapMaker(task.SingleTask):
    """A simple and quick map-maker that forms a series of beams on the meridian.

    This is designed to run on data after it has been collapsed down to
    non-redundant baselines only.

    Attributes
    ----------
    weighting : one of ['natural']
    """

    npix = config.Property(proptype=int, default=512)

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
        sin_el = np.linspace(-1.0, 1.0, self.npix)
        vis_pos_1d = np.fft.fftfreq(nvis_1d, d=(1.0 / (nvis_1d * sp)))

        # Create empty ring map
        rm = containers.RingMap(beam=(2 * ncyl - 1), el=self.npix, polarisation=True, axes_from=sstream)
        rm.redistribute('freq')

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


def _make_marray(ts):
    # Construct an array of m-modes from a sidereal time stream
    mmodes = np.fft.fft(ts, axis=-1) / ts.shape[-1]
    marray = _pack_marray(mmodes)

    return marray


def _pack_marray(mmodes, mmax=None):
    # Pack an FFT into the correct format for the m-modes (i.e. [m, freq, +/-,
    # baseline])

    if mmax is None:
        mmax = mmodes.shape[-1] / 2

    shape = mmodes.shape[:-1]

    marray = np.zeros((mmax + 1, 2) + shape, dtype=np.complex128)

    marray[0, 0] = mmodes[..., 0]

    mlimit = min(mmax, mmodes.shape[-1] / 2)  # So as not to run off the end of the array
    for mi in range(1, mlimit - 1):
        marray[mi, 0] = mmodes[..., mi]
        marray[mi, 1] = mmodes[..., -mi].conj()

    return marray


def pinv_svd(M, acond=1e-4, rcond=1e-3):
    # Generate the pseudo-inverse from an svd
    # Not really clear why I'm not just using la.pinv2 instead,

    import scipy.linalg as la

    u, sig, vh = la.svd(M, full_matrices=False)

    rank = np.sum(np.logical_and(sig > rcond * sig.max(), sig > acond))

    psigma_diag = 1.0 / sig[: rank]

    B = np.transpose(np.conjugate(np.dot(u[:, : rank] * psigma_diag, vh[: rank])))

    return B

"""
========================================================
Map making tasks (:mod:`~ch_pipeline.analysis.mapmaker`)
========================================================

.. currentmodule:: ch_pipeline.analysis.mapmaker

Tools for map making from CHIME data using the m-mode formalism.

Tasks
=====

.. autosummary::
    :toctree: generated/

    DirtyMapMaker
    MaximumLikelihoodMapMaker
    WienerMapMaker
    RingMapMaker
"""
import numpy as np
from caput import mpiarray, config

from ..core import containers, task


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
                v = m_array[mi, :, fi].view(np.ndarray)
                a = alm[fi, ..., mi].view(np.ndarray)
                Ni = m_weight[mi, :, fi].view(np.ndarray)

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
        a = a.reshape(bt.telescope.num_pol_sky, bt.telescope.lmax + 1)

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
        a = a.reshape(bt.telescope.num_pol_sky, bt.telescope.lmax + 1)

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
        a = np.zeros((bt.telescope.num_pol_sky, bt.telescope.lmax + 1), dtype=v.dtype)
        a[:, m:] = a_wiener.reshape(bt.telescope.num_pol_sky, -1)

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


def pinv_svd(M, acond=1e-4, rcond=1e-3):
    # Generate the pseudo-inverse from an svd
    # Not really clear why I'm not just using la.pinv2 instead,

    import scipy.linalg as la

    u, sig, vh = la.svd(M, full_matrices=False)

    rank = np.sum(np.logical_and(sig > rcond * sig.max(), sig > acond))

    psigma_diag = 1.0 / sig[: rank]

    B = np.transpose(np.conjugate(np.dot(u[:, : rank] * psigma_diag, vh[: rank])))

    return B

"""Map making from driftscan data using the m-mode formalism.

Tasks
=====

.. autosummary::
    :toctree:

    DirtyMapMaker
    MaximumLikelihoodMapMaker
    WienerMapMaker
    RingMapMaker
"""
import numpy as np
from caput import mpiarray, config

from ..core import containers, task, io


class BaseMapMaker(task.SingleTask):
    """Rudimetary m-mode map maker.

    Attributes
    ----------
    nside : int
        Resolution of output Healpix map.
    """

    nside = config.Property(proptype=int, default=256)
    basis = config.enum(['mmodes', 'svdmodes'], default='mmodes')
    use_weights = config.Property(proptype=bool, default=True)

    bt_cache = None

    def setup(self, bt):
        """Set the beamtransfer matrices to use.

        Parameters
        ----------
        bt : beamtransfer.BeamTransfer or manager.ProductManager
            Beam transfer manager object (or ProductManager) containing all the
            pre-generated beam transfer matrices.
        """

        self.beamtransfer = io.get_beamtransfer(bt)

    def process(self, modes):
        """Make a map from the given m-modes.

        Parameters
        ----------
        modes : containers.MModes or containers.SVDModes, as specified in
                 config.

        Returns
        -------
        map : containers.Map
        """

        from cora.util import hputil

        # Fetch various properties
        bt = self.beamtransfer
        lmax = bt.telescope.lmax
        mmax = min(bt.telescope.mmax, len(modes.index_map['m']) - 1)
        nfreq = bt.telescope.nfreq

        def find_key(key_list, key):
            try:
                return map(tuple, list(key_list)).index(tuple(key))
            except TypeError:
                return list(key_list).index(key)
            except ValueError:
                return None

        if self.basis == 'mmodes':
            # Figure out mapping between the frequencies, there need to be more.
            bt_freq = self.beamtransfer.telescope.frequencies
            mm_freq = modes.index_map['freq']['centre']
            freq_ind = [find_key(bt_freq, mf) for mf in mm_freq]

            modes.redistribute('freq')
            # Trim off excess m-modes
            m_array = modes.vis[:(mmax + 1)]
            m_array = m_array.redistribute(axis=0)

            m_weight = modes.weight[:(mmax + 1)]
            m_weight = m_weight.redistribute(axis=0)

        if self.basis == 'svdmodes':
            modes.redistribute('m')
            m_array = modes.vis[:]
            m_weight = modes.weight[:]

        # Create array to store alms in.
        alm = mpiarray.MPIArray((nfreq, 4, lmax + 1, mmax + 1), axis=3,
                                dtype=np.complex128, comm=modes.comm)
        alm[:] = 0.0

        # Loop over all m's and solve from m-mode visibilities to alms.
        for mi, m in m_array.enumerate(axis=0):

            self.log.debug("Processing m=%i (local %i/%i)",
                           m, mi + 1, m_array.local_shape[0])
            # Get and cache the beam transfer matrix, but trim off any l < m.
            # if self.bt_cache is None:
            #   self.bt_cache = (m, bt.beam_m(m))
            #   self.log.debug("Cached beamtransfer for m=%i", m)

            for fi in range(nfreq):
                if self.basis == 'svdmodes':
                    v = m_array[mi, :].view(np.ndarray)
                    Ni = m_weight[mi, :].view(np.ndarray)
                    f = fi
                else:
                    v = m_array[mi, :, fi].view(np.ndarray)
                    Ni = m_weight[mi, :, fi].view(np.ndarray)
                    f = freq_ind[fi]
                a = alm[fi, ..., mi].view(np.ndarray)

                a[:] = self._solve_m(m, f, v, Ni)

            self.bt_cache = None

        # Redistribute back over frequency
        alm = alm.redistribute(axis=0)

        # Copy into square alm array for transform
        almt = mpiarray.MPIArray((nfreq, 4, lmax + 1, lmax + 1), dtype=np.complex128, axis=0, comm=modes.comm)
        almt[..., :(mmax + 1)] = alm
        alm = almt

        # Perform spherical harmonic transform to map space
        maps = hputil.sphtrans_inv_sky(alm, self.nside)
        maps = mpiarray.MPIArray.wrap(maps, axis=0)

        # If we are making an svd map we need to create a frequency axis -
        # Get frequencies from telescope object
        if self.basis == 'svdmodes':
            freqmap = np.zeros(nfreq,
                               dtype=[('centre', np.float64), ('width', np.float64)])
            freqmap['centre'][:] = bt.telescope.frequencies
            freqmap['width'][:] = np.abs(np.diff(bt.telescope.frequencies)[0])

            m = containers.Map(nside=self.nside, freq=freqmap, comm=modes.comm)

        if self.basis == 'mmodes':
            m = containers.Map(nside=self.nside, axes_from=modes, comm=modes.comm)

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
            Frequency we are solving for. This is the index for the beam transfers.
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
    r"""Generate a Wiener filtered map assuming that the signal is a Gaussian
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

    bt_cache = None

    def _solve_m(self, m, f, v, Ni):

        bt = self.beamtransfer

        # Get transfer for this m and f
        if self.bt_cache is not None and self.bt_cache[0] == m:
            bm = self.bt_cache[1][f]
        else:
            bm = bt.beam_m(m, fi=f)
        bm = bm[..., m:].reshape(bt.ntel, -1)

        # Massage the arrays into shape
        v = v.reshape(bt.ntel)
        Ni = Ni.reshape(bt.ntel)
        Nh = Ni**0.5

        # Construct pre-wightened beam and beam-conjugated matrices
        bmt = bm * Nh[:, np.newaxis]
        bth = bmt.T.conj()
        # Pre-wighten the visibilities
        vt = Nh * v

        a_wiener = self._minimize(m, bmt, bth, vt)

        # Copy the solution into a correctly shaped array output
        a = np.zeros((bt.telescope.num_pol_sky, bt.telescope.lmax + 1), dtype=v.dtype)
        a[:, m:] = a_wiener.reshape(bt.telescope.num_pol_sky, -1)

        return a

    def _minimize(self, m, beam, beam_conj, vec):

        import scipy.linalg as la

        bt = self.beamtransfer

        # Construct the signal covariance matrix
        l = np.arange(bt.telescope.lmax + 1)
        l[0] = 1  # Change l=0 to get around singularity
        l = l[m:]  # Trim off any l < m
        cl_TT = self.prior_amp**2 * l**(-self.prior_tilt)
        S_diag = np.concatenate([cl_TT] * 4)

        # For large ntel it's quickest to solve in the standard Wiener filter way
        if bt.ntel > bt.nsky:
            Ci = np.diag(1.0 / S_diag) + np.dot(beam_conj, beam)  # Construct the inverse covariance
            a_dirty = np.dot(beam_conj, vec)  # Find the dirty map
            a_wiener = la.solve(Ci, a_dirty, sym_pos=True)  # Solve to find C vt

        # If not it's better to rearrange using the results for blockwise matrix inversion
        else:
            pCi = np.identity(vec.shape[0]) + np.dot(beam * S_diag[np.newaxis, :], beam_conj)
            v_int = la.solve(pCi, vec, sym_pos=True)
            a_wiener = S_diag * np.dot(beam_conj, v_int)

        return a_wiener


class DirtyMapMakerSVD(BaseMapMaker):
    """Generate a dirty map in SVD basis.

    Notes
    -----

    The dirty map is produced by generating a set of :math:`a_{lm}` coefficients
    using

    .. math:: \hat{\mathbf{a}} = \bar{\mathbf{B}^{+}}  \bar{\mathbf{v}}

    and then performing the spherical harmonic transform to get the sky intensity.
    """

    def _solve_m(self, m, f, vec, Ni):

        bt = self.beamtransfer

        # Get the svd beamtransfer matrix for a specific m and f
        beam = bt.beam_svd(m, f)
        # Create alm array
        a = np.zeros((bt.telescope.num_pol_sky, bt.telescope.lmax + 1), dtype=vec.dtype)
        # Get the significant svd modes and the frequency bounds
        svnum, svbounds = bt._svd_num(m)

        if svnum[f] < 1:
            return a

        # Get the svd vector for this frequency
        fvec = vec[svbounds[f]:svbounds[f+1]]
        # Get the inverse beam with the significant svd modes but trim off any l < m.
        beam_svnum = beam[:svnum[f], :, m:]
        beam_svnum = beam_svnum.reshape(svnum[f], -1)

        # Construct beam-conjugated matrices
        beam_svnum_conj = beam_svnum.T.conj()

        # Solve for the dirty map alms
        a = np.dot(beam_svnum_conj, fvec)
        a = a.reshape(bt.telescope.num_pol_sky, -1)

        return a


class WienerMapMakerSVD(WienerMapMaker):
    r"""Generate a Wiener filtered map from the SVD basis assuming that the signal is a Gaussian
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

    def _solve_m(self, m, f, vec, Ni):

        bt = self.beamtransfer

        # Get the beam transfer matrix for an m and f
        beam = bt.beam_svd(m, f)
        # Create alm array
        a = np.zeros((bt.telescope.num_pol_sky, bt.telescope.lmax + 1), dtype=vec.dtype)
        # Get the significant svd modes and the frequency bounds
        svnum, svbounds = bt._svd_num(m)

        if svnum[f] < 1:
            return a

        # Get the svd vector for this frequency
        fvec = vec[svbounds[f]:svbounds[f+1]]
        # Get the inverse beam with the significant svd modes but trim off any l < m.
        beam_svnum = beam[:svnum[f], :, m:]
        beam_svnum = beam_svnum.reshape(svnum[f], -1)

        # Construct beam-conjugated matrices
        beam_svnum_conj = beam_svnum.T.conj()

        a_wiener = self._minimize(m, beam_svnum, beam_svnum_conj, fvec)

        # Copy the solution into a correctly shaped array output
        a[:, m:] = a_wiener.reshape(bt.telescope.num_pol_sky, -1)

        return a


def pinv_svd(M, acond=1e-4, rcond=1e-3):
    # Generate the pseudo-inverse from an svd
    # Not really clear why I'm not just using la.pinv2 instead,

    import scipy.linalg as la

    u, sig, vh = la.svd(M, full_matrices=False)

    rank = np.sum(np.logical_and(sig > rcond * sig.max(), sig > acond))

    psigma_diag = 1.0 / sig[: rank]

    B = np.transpose(np.conjugate(np.dot(u[:, : rank] * psigma_diag, vh[: rank])))

    return B

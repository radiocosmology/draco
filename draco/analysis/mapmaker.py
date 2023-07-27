"""Map making from driftscan data using the m-mode formalism."""

import numpy as np
from caput import mpiarray, config

from ..core import containers, task, io
from ..util import tools


class BaseMapMaker(task.SingleTask):
    """Rudimetary m-mode map maker.

    Attributes
    ----------
    nside : int
        Resolution of output Healpix map.
    multifreq : bool
        If this should be a multi-frequency map or not (default: False).
    nmaps : int
        How many multi-frequency maps to output.
    """

    nside = config.Property(proptype=int, default=256)
    multifreq = config.Property(proptype=bool, default=False)
    nmaps = config.Property(proptype=int, default=4)

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
        mmax = min(bt.telescope.mmax, len(mmodes.index_map["m"]) - 1)
        nfreq = len(mmodes.index_map["freq"])  # bt.telescope.nfreq

        # Figure out mapping between the frequencies
        bt_freq = self.beamtransfer.telescope.frequencies
        # mm_freq might need be an attribute of class so BTM can be scaled in loop
        mm_freq = mmodes.index_map["freq"]["centre"][:]

        freq_ind = tools.find_keys(bt_freq, mm_freq, require_match=True)

        # Trim off excess m-modes
        mmodes.redistribute("freq")
        m_array = mmodes.vis[: (mmax + 1)]
        m_array = m_array.redistribute(axis=0)

        m_weight = mmodes.weight[: (mmax + 1)]
        m_weight = m_weight.redistribute(axis=0)

        # Frequency axis is the number of maps for a multi-frequency map or the number of total frequencies
        nfreq2 = self.nmaps if self.multifreq else nfreq

        if self.multifreq:
            freqmap = np.zeros(self.nmaps, dtype=[("centre", float), ("width", float)])
        else:
            freqmap = mmodes.index_map["freq"][:]

        # Create array to store alms in.
        alm = mpiarray.MPIArray(
            (nfreq2, 4, lmax + 1, mmax + 1),
            axis=3,
            dtype=np.complex128,
            comm=mmodes.comm,
        )
        alm[:] = 0.0

        # Loop over all m's and solve from m-mode visibilities to alms.
        for mi, m in m_array.enumerate(axis=0):
            self.log.debug(
                "Processing m=%i (local %i/%i)", m, mi + 1, m_array.local_shape[0]
            )

            # Get and cache the beam transfer matrix, but trim off any l < m.
            # if self.bt_cache is None:
            #     self.bt_cache = (m, bt.beam_m(m))
            #     self.log.debug("Cached beamtransfer for m=%i", m)

            # Multi-frequency iterator
            j = int(nfreq / self.nmaps)

            for fi in range(nfreq2):
                fj = slice(fi * j, (fi + 1) * j) if self.multifreq else fi

                # If multi-frequency map, create a frequency map
                if self.multifreq:
                    freqmap["centre"][fi] = np.mean(mm_freq[fj])
                    freqmap["width"][fi] = abs(mm_freq[fj][0] - mm_freq[fj][-1])

                v = m_array.local_array[mi, :, fj]
                a = alm.local_array[fi, ..., mi]
                Ni = m_weight.local_array[mi, :, fj]
                a[:] = self._solve_m(m, freq_ind[fj], v, Ni)

        self.bt_cache = None

        # Redistribute back over frequency
        alm = alm.redistribute(axis=0)

        # Copy into square alm array for transform
        almt = mpiarray.MPIArray(
            (nfreq2, 4, lmax + 1, lmax + 1),
            dtype=np.complex128,
            axis=0,
            comm=mmodes.comm,
        )
        almt[..., : (mmax + 1)] = alm
        alm = almt

        # Perform spherical harmonic transform to map space
        maps = hputil.sphtrans_inv_sky(alm, self.nside)
        maps = mpiarray.MPIArray.wrap(maps, axis=0)

        m = containers.Map(
            nside=self.nside, freq=freqmap, axes_from=mmodes, comm=mmodes.comm
        )
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
    r"""Generate a dirty map.

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


class DirtyMapMakerMultiFreq(BaseMapMaker):
    r"""Generate a dirty map.

    Notes
    -----

    The dirty map is produced by generating a set of :math:`a_{lm}` coefficients
    using

    .. math:: \hat{\mathbf{a}} = \mathbf{B}^\dagger \mathbf{N}^{-1} \mathbf{v}

    and then performing the spherical harmonic transform to get the sky intensity.
    """

    spectral_mean = config.Property(proptype=float, default=-2.7)

    def _solve_m(self, m, f, v, Ni):
        bt = self.beamtransfer
        bt_freq = bt.telescope.frequencies
        a_dirty = np.zeros((len(f), bt.nsky), dtype=np.complex128)

        f0 = np.mean(bt_freq[f])

        # Massage the arrays into shape
        for i in range(len(f)):
            self.log.debug("Processing frequency" + str(bt_freq[f[i]]))
            vi = v[:, i].reshape(bt.ntel)
            Nii = Ni[:, i].reshape(bt.ntel)
            # BTM needs to be scaled to mean frequency (\nu_0 / nu_i)^{\spectra_mean}
            bm = bt.beam_m(m, fi=f[i]).reshape(bt.ntel, bt.nsky) * (
                f0 / bt_freq[f[i]]
            ) ** (self.spectral_mean)
            a_dirty[i] = np.dot(bm.T.conj(), Nii * vi)

        # Solve for the dirty map alms
        a = np.sum(a_dirty, axis=0)

        # Reshape to the correct output
        a = a.reshape(bt.telescope.num_pol_sky, bt.telescope.lmax + 1)

        return a


class MaximumLikelihoodMapMaker(BaseMapMaker):
    r"""Generate a Maximum Likelihood map using the Moore-Penrose pseudo-inverse.

    Notes
    -----

    The dirty map is produced by generating a set of :math:`a_{lm}` coefficients
    using

    .. math:: \hat{\mathbf{a}} = \left( \mathbf{N}^{-1/2 }\mathbf{B} \right) ^+ \mathbf{N}^{-1/2} \mathbf{v}

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
        import scipy.linalg as la

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

        # Construct the signal covariance matrix
        l = np.arange(bt.telescope.lmax + 1)
        l[0] = 1  # Change l=0 to get around singularity
        l = l[m:]  # Trim off any l < m
        cl_TT = self.prior_amp**2 * l ** (-self.prior_tilt)
        S_diag = np.concatenate([cl_TT] * 4)

        # For large ntel it's quickest to solve in the standard Wiener filter way
        if bt.ntel > bt.nsky:
            Ci = np.diag(1.0 / S_diag) + np.dot(
                bth, bmt
            )  # Construct the inverse covariance
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


class WienerMapMakerMultiFreq(BaseMapMaker):
    r"""Generate a multifrequency Wiener Map"""

    prior_amp = config.Property(proptype=float, default=1.0)
    prior_tilt = config.Property(proptype=float, default=0.5)
    spectral_mean = config.Property(proptype=float, default=-2.7)

    bt_cache = None

    def _solve_m(self, m, f, v, Ni):
        import scipy.linalg as la

        bt = self.beamtransfer
        bt_freq = bt.telescope.frequencies

        f0 = np.mean(bt_freq[f])

        # Construct the signal covariance matrix
        l = np.arange(bt.telescope.lmax + 1)
        l[0] = 1  # Change l=0 to get around singularity
        l = l[m:]  # Trim off any l < m
        cl_TT = self.prior_amp**2 * l ** (-self.prior_tilt)
        S_diag = np.concatenate([cl_TT] * 4)
        s = S_diag.shape[0]

        # Allocate an array for dirty map and covariance
        a_dirty = np.zeros(s, dtype=np.complex128)
        Ci = np.zeros((s, s), dtype=np.complex128)

        # Massage the arrays into shape
        for i in range(len(f)):
            self.log.debug("Processing frequency" + str(bt_freq[f[i]]))
            # Get transfer for this m and f
            if self.bt_cache is not None and self.bt_cache[0] == m:
                bm = self.bt_cache[1][f[i]]
            else:
                bm = bt.beam_m(m, fi=f[i])
            # BTM needs to be scaled by frequency (\nu_0 / nu_i)^{spectral_mean}
            bm = bm[..., m:].reshape(bt.ntel, -1) * (f0 / bt_freq[f[i]]) ** (
                self.spectral_mean
            )

            # Massage the arrays into shape
            vi = v[:, i].reshape(bt.ntel)
            Nii = Ni[:, i].reshape(bt.ntel)
            Nh = Nii**0.5

            # Construct pre-whitened beam and beam-conjugated matrices
            bmt = bm * Nh[:, np.newaxis]
            bth = bmt.T.conj()

            # Pre-whiten the visibilities
            vt = Nh * vi

            # For bt.ntel > bt.nsky
            Ci += np.diag(1.0 / S_diag) + np.dot(bth, bmt)
            a_dirty += np.dot(bth, vt)  # Find the dirty map
            # TO DO: Is there a better way to rearrange for bt.ntel > bt.nsky?

        # Solve to find the multi-frequency a
        a_wiener = la.solve(Ci, a_dirty, sym_pos=True)

        # Copy the solution into a correctly shaped array output
        a = np.zeros((bt.telescope.num_pol_sky, bt.telescope.lmax + 1), dtype=v.dtype)
        a[:, m:] = a_wiener.reshape(bt.telescope.num_pol_sky, -1)

        return a


def pinv_svd(M, acond=1e-4, rcond=1e-3):
    # Generate the pseudo-inverse from an svd
    # Not really clear why I'm not just using la.pinv2 instead,

    import scipy.linalg as la

    u, sig, vh = la.svd(M, full_matrices=False)

    rank = np.sum(np.logical_and(sig > rcond * sig.max(), sig > acond))

    psigma_diag = 1.0 / sig[:rank]

    B = np.transpose(np.conjugate(np.dot(u[:, :rank] * psigma_diag, vh[:rank])))

    return B

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
import healpy as hp
from caput import mpiarray, config
from caput.pipeline import PipelineConfigError
from cora.util import sphfunc

from ..core import containers, task, io
from ..util import tools


class BaseMapMaker(task.SingleTask):
    """Rudimetary m-mode map maker.

    Attributes
    ----------
    nside : int
        Resolution of output Healpix map.
    """

    nside = config.Property(proptype=int, default=256)

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
        mm_freq = mmodes.index_map["freq"]["centre"]

        freq_ind = tools.find_keys(bt_freq, mm_freq, require_match=True)

        # Trim off excess m-modes
        mmodes.redistribute("freq")
        m_array = mmodes.vis[: (mmax + 1)]
        m_array = m_array.redistribute(axis=0)

        m_weight = mmodes.weight[: (mmax + 1)]
        m_weight = m_weight.redistribute(axis=0)

        # Create array to store alms in.
        alm = mpiarray.MPIArray(
            (nfreq, 4, lmax + 1, mmax + 1),
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

            for fi in range(nfreq):
                v = m_array[mi, :, fi].view(np.ndarray)
                a = alm[fi, ..., mi].view(np.ndarray)
                Ni = m_weight[mi, :, fi].view(np.ndarray)

                a[:] = self._solve_m(m, freq_ind[fi], v, Ni)

        self.bt_cache = None

        # Redistribute back over frequency
        alm = alm.redistribute(axis=0)

        # Copy into square alm array for transform
        almt = mpiarray.MPIArray(
            (nfreq, 4, lmax + 1, lmax + 1),
            dtype=np.complex128,
            axis=0,
            comm=mmodes.comm,
        )
        almt[..., : (mmax + 1)] = alm
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

        Nh = Ni ** 0.5

        # Construct the beam pseudo inverse
        ib = pinv_svd(bm * Nh[:, np.newaxis])

        # Solve for the ML map alms
        a = np.dot(ib, Nh * v)

        # Reshape to the correct output
        a = a.reshape(bt.telescope.num_pol_sky, bt.telescope.lmax + 1)

        return a


def _list_float(l):
    if not isinstance(l, (list, tuple)):
        raise ValueError(f"Expected a list, but got {l}.")
    return [float(ll) for ll in l]


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
    spec_ind = config.Property(proptype=float, default=2.8)
    ps_amp = config.Property(proptype=float, default=0.)

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
        Nh = Ni ** 0.5

        # Construct pre-wightened beam and beam-conjugated matrices
        bmt = bm * Nh[:, np.newaxis]
        bth = bmt.T.conj()

        # Pre-wighten the visibilities
        vt = Nh * v

        # Construct the signal covariance matrix
        l = np.arange(bt.telescope.lmax + 1)
        l[0] = 1  # Change l=0 to get around singularity
        l = l[m:]  # Trim off any l < m
        cl_TT = self.prior_amp ** 2 * l ** (-self.prior_tilt)
        # correct amplitude for spectral index
        cl_TT *= (bt.telescope.frequencies[f] / 408.0) ** (-2.0 * self.spec_ind)
        S_diag = np.concatenate([cl_TT] * 4)

        # For large ntel it's quickest to solve in the standard Wiener filter way
        if bt.ntel > bt.nsky:
            # Construct the inverse covariance
            # including flat point-source spectrum
            S_inv = 1.0 / S_diag
            Ci = np.diag(S_inv) + np.dot(BtNi, bmt) * (self.ps_amp * S_inv + 1)
            # Find the dirty map
            a_dirty = np.dot(BtNi, vt)
            # Solve to find C vt
            a_wiener = la.solve(Ci, a_dirty, sym_pos=True)
            del BtNi

        # If not it's better to rearrange using the results for blockwise matrix inversion
        else:
            pCi = np.identity(bt.ntel) + np.dot(bmt * (S_diag + self.ps_amp)[np.newaxis, :], bth)
            v_int = la.solve(pCi, vt, sym_pos=True)
            a_wiener = S_diag * np.dot(bth, v_int)

        # Copy the solution into a correctly shaped array output
        a = np.zeros((bt.telescope.num_pol_sky, bt.telescope.lmax + 1), dtype=v.dtype)
        a[:, m:] = a_wiener.reshape(bt.telescope.num_pol_sky, -1)

        return a


class PointSourceWienerMapMaker(BaseMapMaker):
    r"""Generate a Wiener filtered map assuming that the signal is a Gaussian
    random field described by a power-law power spectum plus a handfull of
    point sources with known positions.

    Attributes
    ----------
    prior_amp : float
        An amplitude prior to use for the map maker. In Kelvin.
    prior_tilt : float
        Power law index prior for the power spectrum.
    spec_ind : float
        Spectral index to assume for the brightness temperature of the diffuse        component, for rescaling the power spectrum.
    kps_amp : float
        Amplitude of the point-source covariance; in Kelvin^2.
    RA : list
        List of the right-ascensions of the point sources; in degrees.
    dec : list
        List of the declinations of the point sources; in degrees.
    ps_amp_file : string
        Name of the file to store the point-source amplitudes in.

    Notes
    -----

    The point source amplitudes are allowed to vary independently. They all
    have the same prior variance, equal across frequencies, whereas the
    diffuse component has a power spectrum that scales with frequency.

    An estimate of the point-source amplitudes is saved to a numpy
    binary (in brightness-temperature units) [TO DO: turn this into a proper
    task output].
    """

    prior_amp = config.Property(proptype=float, default=1.0)
    prior_tilt = config.Property(proptype=float, default=0.5)
    spec_ind = config.Property(proptype=float, default=2.8)
    kps_amp = config.Property(proptype=float, default=1.0)
    RA = config.Property(proptype=list)
    dec = config.Property(proptype=list)
    ps_amp_file = config.Property(proptype=str, default="ps_amps.npy")

    def process(self, mmodes):
        """Overrides the BaseMapMaker routine to allow for input of point source map.

        Parameters
        ----------
        mmodes : containers.MModes

        ps_map : containers.Map

        Returns
        -------
        map : containers.Map
        """

        from cora.util import hputil
        from caput import mpiutil
        from ch_util import ephemeris as eph

        # Fetch various properties
        bt = self.beamtransfer
        tel = bt.telescope
        lmax = bt.telescope.lmax
        mmax = min(bt.telescope.mmax, len(mmodes.index_map["m"]) - 1)
        nfreq = len(mmodes.index_map["freq"])
        npol = bt.telescope.num_pol_sky
        nps = len(self.RA)

        # Save as class variables
        self.tel = tel
        self.lmax = lmax
        self.mmax = mmax
        self.nfreq = nfreq
        self.npol = npol

        def find_key(key_list, key):
            try:
                return list(map(tuple, list(key_list))).index(tuple(key))
            except TypeError:
                return list(key_list).index(key)
            except ValueError:
                return None

        # Figure out mapping between the frequencies
        bt_freq = self.beamtransfer.telescope.frequencies
        mm_freq = mmodes.index_map["freq"]["centre"]

        freq_ind = [find_key(bt_freq, mf) for mf in mm_freq]

        # Trim off excess m-modes
        mmodes.redistribute("freq")
        m_array = mmodes.vis[: (mmax + 1)]
        m_array = m_array.redistribute(axis=0)

        m_weight = mmodes.weight[: (mmax + 1)]
        m_weight = m_weight.redistribute(axis=0)

        # Create array to store final alms in.
        alm = mpiarray.MPIArray(
            (nfreq, 4, lmax + 1, mmax + 1),
            axis=3,
            dtype=np.complex128,
            comm=mmodes.comm,
        )
        alm[:] = 0.0

        # Create array to store projections onto point-source positions in.
        vis_pss = np.zeros((nfreq, nps), dtype=np.complex128)

        # Create array to store matrix in point-source space in.
        M_pss = np.zeros((nfreq, nps, nps), dtype=np.complex128)

        # Create array to store the point source amplitudes.
        ps_amps = np.zeros((nfreq, nps), dtype=np.complex128)

        # Create array to store the point source projection matrix
        U = np.zeros((nfreq, bt.ntel, nps), dtype=np.complex128)

        # Load in the alms of the point sources.
        self._get_ps_alms()

        # Loop over all m's collecting the required components.
        for mi, m in m_array.enumerate(axis=0):

            for fi in range(nfreq):
                v = m_array[mi, :, fi].view(np.ndarray)
                a = alm[fi, ..., mi].view(np.ndarray)
                Ni = m_weight[mi, :, fi].view(np.ndarray)

                # Get the diagonal (in m) piece of the inverse
                self.log.debug(f"Solving Wiener for m={m}, fi={fi}.")
                a[:], Ainv, ps_amps_i = self._solve_m_diagonal(
                    m, mi, fi, bt_freq[fi], v, Ni
                )

                # Get the ingredients for the correction in point-source space
                self.log.debug(f"Solving correction for m={m}, fi={fi}.")
                M_pss_i, vis_pss_i, U_i = self._solve_m_correction(
                    Ainv, m, mi, fi, v, Ni
                )

                # Acumulate over m
                vis_pss[fi] = vis_pss[fi] + vis_pss_i
                M_pss[fi] = M_pss[fi] + M_pss_i
                ps_amps[fi] = ps_amps[fi] + ps_amps_i
                U[fi] += U_i.T

        for fi in range(nfreq):
            self.log.debug(f"Computing point source vis for fi={fi}.")
            # invert the point-source space covariance
            M_pss[fi] = np.linalg.inv(np.identity(nps) / self.kps_amp + M_pss[fi])
            # multiply the point-source-projected visibilities with this matrix
            vis_pss[fi] = np.dot(M_pss[fi], vis_pss[fi])

        # Calculate point source correction projected onto visibilities
        corr = np.dot(U[fi], vis_pss[fi])

        for mi, m in m_array.enumerate(axis=0):

            for fi in range(nfreq):
                a = alm[fi, ..., mi].view(np.ndarray)
                Ni = m_weight[mi, :, fi].view(np.ndarray)

                self.log.debug(f"Assembling results: m={m}, fi={fi}.")

                # corr = self._spread_ps_results(vis_pss[fi], m, mi, fi, Ni)

                acorr, Ainv, pscorr = self._solve_m_diagonal(
                    m, mi, fi, bt_freq[fi], corr, Ni, prewhiten=False
                )

                a[:] = a[:] - acorr
                ps_amps[fi] = ps_amps[fi] - pscorr

        alm = alm.redistribute(axis=0)

        # Copy into square alm array for transform
        almt = mpiarray.MPIArray(
            (nfreq, 4, lmax + 1, lmax + 1),
            dtype=np.complex128,
            axis=0,
            comm=mmodes.comm,
        )
        almt[..., : (mmax + 1)] = alm
        alm = almt

        # Perform spherical harmonic transform to map space
        maps = hputil.sphtrans_inv_sky(alm, self.nside)
        maps = mpiarray.MPIArray.wrap(maps, axis=0)

        m = containers.Map(nside=self.nside, axes_from=mmodes, comm=mmodes.comm)
        m.map[:] = maps

        np.save(self.ps_amp_file, ps_amps)

        return m

    def _solve_m_diagonal(self, m, mi, f, freq, v, Ni, prewhiten=True):
        """perform the regular Wiener inversion (i.e., the part that's diagonal in m)"""

        import scipy.linalg as la

        bt = self.beamtransfer
        lmax = bt.telescope.lmax
        nps = len(self.RA)

        # Massage the arrays into shape
        v = v.reshape(bt.ntel)
        Ni = Ni.reshape(bt.ntel)
        Nh = Ni ** 0.5

        # Get the beam transfer matrix, but trim off any l < m.
        bm = bt.beam_m(m, fi=f)[..., m:].copy()
        bm[np.isnan(bm)] = 0.0

        bm = bm.reshape(bt.ntel, -1)

        # Construct pre-wightened beam and beam-conjugated matrices
        bmt = bm * Nh[:, np.newaxis]
        bth = bmt.T.conj()

        # Pre-wighten the visibilities
        if prewhiten:
            vt = Nh * v
        else:
            vt = v

        # Construct the signal covariance matrix
        l = np.arange(lmax + 1)
        l[0] = 1  # Change l=0 to get around singularity
        l = l[m:]  # Trim off any l < m
        cl_TT = self.prior_amp * l ** (-self.prior_tilt)
        # correct amplitude for spectral index
        cl_TT *= (bt.telescope.frequencies[f] / 408.0) ** (-2.0 * self.spec_ind)
        S_diag = np.concatenate([cl_TT] * 4)

        # Rearrange for blockwise matrix inversion
        pCi = np.identity(bt.ntel) + np.dot(bmt * S_diag[np.newaxis, :], bth)
        pC = la.inv(pCi)
        vis_wiener = np.dot(pC, vt)
        a_wiener = np.dot(bth, vis_wiener)

        # Get the Y_l's for this m, but trim any l<m.
        Yl = np.array([self.col_alm[mi, ii, f][..., m:].ravel() for ii in range(nps)])
        ps_amps = self.kps_amp * np.array(
            [np.dot(Yl[ii].conjugate(), a_wiener) for ii in range(nps)]
        )

        a_wiener = S_diag * a_wiener

        # Copy the solution into a correctly shaped array output
        a = np.zeros((bt.telescope.num_pol_sky, lmax + 1), dtype=v.dtype)
        a[:, m:] = a_wiener.reshape(bt.telescope.num_pol_sky, -1)

        return a, pC, ps_amps

    def _solve_m_correction(self, Ainv, m, mi, f, v, Ni):
        """Get the ingredients for the correction in point-source space"""

        import scipy.linalg as la

        bt = self.beamtransfer
        lmax = bt.telescope.lmax
        nps = len(self.RA)

        # Massage the arrays into shape
        v = v.reshape(bt.ntel)
        Ni = Ni.reshape(bt.ntel)
        Nh = Ni ** 0.5

        # Get the beam transfer matrix, but trim off any l < m.
        bm = bt.beam_m(m, fi=f)[..., m:].copy()
        bm[np.isnan(bm)] = 0.0

        bm = bm.reshape(bt.ntel, -1)

        # Construct pre-wightened beam and beam-conjugated matrices
        bmt = bm * Nh[:, np.newaxis]
        bth = bmt.T.conj()

        # Pre-wighten the visibilities
        vt = Nh * v
        vt = np.dot(Ainv, vt)

        # Get the Y_l's for this m, but trim any l<m.
        Yl = np.array([self.col_alm[mi, ii, f][..., m:].ravel() for ii in range(nps)])

        # Compute A^{-1}U
        Bu = np.array([np.dot(bmt, Yl[ii]) for ii in range(nps)])
        ui = np.array([np.dot(Ainv, Bu[ii]) for ii in range(nps)])

        # Compute the piece of the point-source covariance for this m.
        M_pss_i = np.array(
            [
                [np.dot(Bu[ii].T.conj(), ui[jj]) for jj in range(nps)]
                for ii in range(nps)
            ]
        )

        # Compute the point-source-projected visibilities for this m.
        vis_pss_i = np.array([np.dot(Bu[ii].T.conj(), vt) for ii in range(nps)])

        return M_pss_i, vis_pss_i, Bu

    def _spread_ps_results(self, vis_pss, m, mi, f, Ni):
        """project the results of the point-source correction back into map space"""

        bt = self.beamtransfer
        lmax = bt.telescope.lmax
        nps = len(self.RA)

        # Massage the arrays into shape
        Ni = Ni.reshape(bt.ntel)
        Nh = Ni ** 0.5

        # Get the beam transfer matrix, but trim off any l < m.
        bm = bt.beam_m(m, fi=f)[..., m:].copy()
        bm[np.isnan(bm)] = 0.0

        bm = bm.reshape(bt.ntel, -1)

        # Construct pre-wightened beam and beam-conjugated matrices
        bmt = bm * Nh[:, np.newaxis]
        bth = bmt.T.conj()

        Yl = np.array([self.col_alm[mi, ii, f][..., m:].ravel() for ii in range(nps)])
        Bu = np.array([np.dot(bmt, Yl[ii]) for ii in range(nps)])
        corr = np.zeros(Bu[0].shape, dtype=complex)
        for ii in range(nps):
            corr += vis_pss[ii] * Bu[ii]

        return corr

    def _get_ps_alms(self):
        """Calculate the alms for each point source. At the moment this is done in
        a stupid way by creating a map with one non-zero pixel each and
        calculating the spherical-harmonic transform of those maps.
        TO DO: replace this procedure with a direct evaluation of spherical
        harmonics, using a fast and accurate library (perhaps pyGSL or
        pyshtools)
        """

        from caput import mpiutil
        from cora.util import hputil

        nfreq = self.nfreq
        npol = self.npol
        mmax = self.mmax
        lmax = self.lmax
        tel = self.beamtransfer.telescope
        nps = len(self.RA)

        row_maps = [containers.Map(nside=self.nside, freq=nfreq) for ii in range(nps)]
        for ii in range(nps):
            row_maps[ii].map[:] = 0.0
            theta = np.pi / 2.0 - self.dec[ii] / 180.0 * np.pi
            phi = self.RA[ii] / 180.0 * np.pi
            pix = hputil.healpy.ang2pix(self.nside, theta, phi)
            row_maps[ii].map[:, 0, pix] = 1.0

        # Calculate the alms of input point source map
        row_alm = np.array(
            [
                hputil.sphtrans_sky(row_maps[ii].map[:], lmax=lmax).reshape(
                    (-1, npol * (lmax + 1), lmax + 1)
                )
                for ii in range(nps)
            ]
        )

        # Trim off excess m's and wrap into MPIArray
        row_alm = row_alm[..., : (mmax + 1)]
        row_alm = mpiarray.MPIArray.wrap(row_alm, axis=1)

        # Perform the transposition to distribute different m's across processes.
        col_alm = row_alm.redistribute(axis=3)

        # Transpose and reshape to shift m index first.
        col_alm = col_alm.transpose((3, 0, 1, 2)).reshape(
            (None, nps, nfreq, npol, lmax + 1)
        )

        self.col_alm = col_alm


def pinv_svd(M, acond=1e-4, rcond=1e-3):
    # Generate the pseudo-inverse from an svd
    # Not really clear why I'm not just using la.pinv2 instead,

    import scipy.linalg as la

    u, sig, vh = la.svd(M, full_matrices=False)

    rank = np.sum(np.logical_and(sig > rcond * sig.max(), sig > acond))

    psigma_diag = 1.0 / sig[:rank]

    B = np.transpose(np.conjugate(np.dot(u[:, :rank] * psigma_diag, vh[:rank])))

    return B


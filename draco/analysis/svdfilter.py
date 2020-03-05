"""A set of tasks for SVD filtering the m-modes or telescope-SVD modes.

These tasks perform a separate SVD for each m, representing the visibilities
as a (nfreq,msign*nprod) [e.g., frequency vs. baseline and sign of m] matrix
if acting on m-modes, or a (nfreq,nmode) matrix if acting on telescope-SVD modes.
SVDFilter can return the m-modes/tel-SVD-modes with the largest modes filtered
out, with the option to save the SVD basis (modes and singular values) to disk
for later use.

Tasks
=====

.. autosummary::
    :toctree:

    SVDSpectrumEstimator
    SVDFilter
    SVDFilterFromFile
    svd_em
"""


# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os

import numpy as np
import scipy.linalg as la

import h5py

from caput import config, mpiutil
from drift.util import util

from draco.core import task, containers, io


def in_mode_reshape(m, vis, weight, is_mm, bt):
    """For a given m, reshape visibilities into 2d array.

    For m-modes, reshape to [msign*nprod,freq], while for
    SVD modes, reshape to [min_nmodes,freq], where min_nmodes
    is the minimum number of modes per frequency, taken over all
    frequencies. Taking the minimum is necessary because different
    frequencies will generally have different numbers of SVD modes;
    cutting the lowest ones from some frequencies will result in a
    small S/N hit, but probably quite small.

    Also, this function reshapes the visibility weights in the same way,
    and returns a mask indicating the locations of zero weights.

    Parameters
    ----------
    m : integer
        m value we're working with.
    vis : array
        Array of visibilities for specific m.
    weight : array
        Array of visibility weights for specific m.
    is_mm : bool
        Whether visibilities are in m-mode format (True) or
        telescope-SVD format (False).
    bt : BeamTransfer
        BeamTransfer instance corresponding to telescope.
        This can also take a ProductManager instance.

    Returns
    -------
    vis_m : array
        Reshaped visibilities for specific m, with frequency as
        second index.
    weight_m : array
        Visibility weights in same shape as vis_m.
    """
    nfreq = bt.telescope.nfreq

    # M-mode case
    if is_mm:
        # Transpose vis and weight arrays for a given m from
        # [msign,freq,nprod] to [freq,msign,nprod], reshape into
        # [freq,msign*nprod], and transpose again to [msign*nprod,freq]
        vis_m = (
            vis
            .view(np.ndarray)
            .transpose((1, 0, 2))
            .reshape(nfreq, -1)
            .transpose((1,0))
        )
        weight_m = (
            weight
            .view(np.ndarray)
            .transpose((1, 0, 2))
            .reshape(nfreq, -1)
            .transpose((1,0))
        )

        # Make mask corresponding to zero weights
        mask_m = weight_m == 0.0

    # SVD-mode case
    else:
        # Compute minimum number of tel-SVD modes stored at any frequency
        svd_num = bt._svd_num(m)
        nmodes_min = np.min(svd_num[0])

        # Make empty arrays to hold reshaped visibilities (in tel-SVD basis)
        # and weights
        vis_m = np.zeros((nmodes_min,nfreq), dtype=np.complex128)
        weight_m = np.zeros((nmodes_min,nfreq), dtype=np.complex128)

        # Fill in vis_m and weight_m for each frequency. We only keep the
        # first nmodes_min modes, so that we get an nmodes_min by nfreq
        # matrix that we can SVD.
        for fi in range(nfreq):
            vis_m[:,fi] = vis[svd_num[1][fi] : svd_num[1][fi]+nmodes_min]
            weight_m[:,fi] = weight[svd_num[1][fi] : svd_num[1][fi]+nmodes_min]

        # Make mask corresponding to zero weights
        mask_m = weight_m == 0.0

    return vis_m, mask_m


def in_mode_inv_reshape(m, vis, is_mm, bt):
    """For a given m, reverse the reshaping done by in_mode_reshape().

    Parameters
    ----------
    m : integer
        m value we're working with.
    vis : array
        Array of visibilities for specific m, previously reshaped by
        in_mode_reshape().
    is_mm : bool
        Whether visibilities are in m-mode format (True) or
        telescope-SVD format (False).
    bt : BeamTransfer
        BeamTransfer instance corresponding to telescope.
        This can also take a ProductManager instance.

    Returns
    -------
    vis_m : array
        Visibilites in original m-mode or tel-SVD shape.
    """
    nfreq = bt.telescope.nfreq

    # M-mode case
    if is_mm:
        # Transpose vis array from [msign*nprod,freq] to
        # [msign*nprod,freq], reshape to [freq,msign,nprod],
        # and transpose again to [msign,freq,nprod]
        vis_m = vis.transpose((1,0)).reshape(nfreq, 2, -1).transpose((1, 0, 2))

    # SVD-mode case
    else:
        # Compute minimum number of tel-SVD modes stored at any frequency,
        # as well as the total number of modes over all frequencies
        svd_num = bt._svd_num(m)
        nmodes_min = np.min(svd_num[0])
        nmodes_tot = np.sum(svd_num[0])

        # Create empty 1d array store visibilities, and set it all to zero
        vis_m = np.zeros(nmodes_tot, dtype=np.complex128)

        # Fill in vis_m at each frequency, using starting indices corresponding
        # to each frequency (which are stored in svd_num[1,:]). Note that only
        # the first nmodes_min visibilities at each frequency are retained.
        for fi in range(nfreq):
            vis_m[svd_num[1][fi] : svd_num[1][fi]+nmodes_min] = vis[:,fi]

    return vis_m


class SVDSpectrumEstimator(task.SingleTask):
    """Calculate the SVD spectrum of a set of m-modes.

    Attributes
    ----------
    niter : int
        Number of iterations of EM to perform.
    min_modes_for_svd : int, optional
        If the number of input degrees of freedom per frequency is less
        than this number for a given m, skip the SVD for that m, since there
        is barely any signal there (default: 5)
    """

    niter = config.Property(proptype=int, default=5)
    min_modes_for_svd = config.Property(proptype=int, default=5)

    def setup(self, manager):
        """Set the beamtransfer instance (needed if we are operating
        on telescope-SVD modes).

        Parameters
        ----------
        manager : BeamTransfer
            This can also take a ProductManager instance.
        """
        self.beamtransfer = io.get_beamtransfer(manager)
        self.tel = io.get_telescope(manager)

    def process(self, inmodes):
        """Calculate the spectrum.

        Parameters
        ----------
        inmodes : :class:`containers.MModes` or :class:`containers.SVDModes`
            Input modes to perform the SVD on.

        Returns
        -------
        spectrum : containers.SVDSpectrum
            Spectrum of SVD applied to input modes.
        """
        is_mm = isinstance(inmodes, containers.MModes)

        inmodes.redistribute("m")

        vis = inmodes.vis[:]
        weight = inmodes.weight[:]

        # M-mode case
        if is_mm:
            # The number of SVD modes is the minimum of msign*nprod
            # and nfreq
            nmode = min(vis.shape[1] * vis.shape[3], vis.shape[2])

        # SVD-mode case
        else:
            # The maximum number of SVD modes per m is equal to the minimum of:
            # - the number of frequencies
            # - the maximum over m of the minimum number of tel-SVD modes
            #    over frequencies at each m
            nfreq = self.tel.nfreq

            n_telsvdmodes_min = 0
            for m in range(self.tel.mmax):
                svd_num = self.beamtransfer._svd_num(m)
                n_telsvdmodes_min = max(n_telsvdmodes_min, np.min(svd_num[0]))
            nmode = min(n_telsvdmodes_min, nfreq)

        spec = containers.SVDSpectrum(singularvalue=nmode, axes_from=inmodes)
        spec.spectrum[:] = 0.0

        for mi, m in vis.enumerate(axis=0):
            self.log.debug("Calculating SVD spectrum of m=%i", m)

            vis_m, mask_m = in_mode_reshape(m, vis[mi], weight[mi], is_mm, self.beamtransfer)

            # If there are fewer than min_modes_for_svd modes per frequency,
            # don't do SVD (corresponding SV will be assumed zero)
            if vis_m.shape[0] < self.min_modes_for_svd:
                continue

            # Do SVD. u is matrix of msign+nprod singular vectors,
            # vh is matrix of freq singular vectors
            u, sig, vh = svd_em(vis_m, mask_m, niter=self.niter)

            spec.spectrum[m] = sig

        return spec


class SVDFilter(task.SingleTask):
    """SVD filter the m-modes to remove the most correlated components.

    Attributes
    ----------
    niter : int
        Number of iterations of EM to perform.
    local_threshold : float
        Cut out modes with singular value higher than `local_threshold` times the
        largest mode on each m (default: 0.1).
    global_threshold : float
        Remove modes with singular value higher than `global_threshold` times the
        largest mode on any m (default: 0.1).
    basis_output_dir : string, optional
        Directory to write SVD basis to, if desired
    save_basis_only : bool, optional
        Whether to compute and save SVD basis, without actually filtering
        the input m-modes (default: False)
    min_modes_for_svd : int, optional
        If the number of input degrees of freedom per frequency is less
        than this number for a given m, skip the SVD for that m, since there
        is barely any signal there (default: 5)
    """

    niter = config.Property(proptype=int, default=5)
    global_threshold = config.Property(proptype=float, default=0.1)
    local_threshold = config.Property(proptype=float, default=0.1)
    basis_output_dir = config.Property(proptype=str, default=None)
    save_basis_only = config.Property(proptype=bool, default=False)
    min_modes_for_svd = config.Property(proptype=int, default=5)

    def _svdfile(self, m, mmax):
        """Filename for SVD basis for single m-mode.
        """
        return external_svdfile(self.basis_output_dir, m, mmax)

    def setup(self, manager):
        """Set the beamtransfer instance (needed if we are operating
        on telescope-SVD modes).

        Parameters
        ----------
        manager : BeamTransfer
            This can also take a ProductManager instance.
        """
        self.beamtransfer = io.get_beamtransfer(manager)

    def process(self, inmodes):
        """Filter m-modes or telescope-SVD modes using an SVD.

        Parameters
        ----------
        inmodes : :class:`containers.MModes` or :class:`containers.SVDModes`
            Input modes to perform the SVD on.

        Returns
        -------
        inmodes : :class:`containers.MModes` or :class:`containers.SVDModes`
            Input modes, either filtered with new SVD or not.
        """
        from mpi4py import MPI

        is_mm = isinstance(inmodes, containers.MModes)

        inmodes.redistribute("m")

        vis = inmodes.vis[:]
        weight = inmodes.weight[:]

        # Dirty way to get mmax: number of m's on each rank, times number of
        # ranks. Is there a better way?
        mmax = vis.shape[0]*mpiutil.size

        sv_max = 0.0

        # Make directory for SVD basis files
        if mpiutil.rank0 and self.basis_output_dir is not None:
            if not os.path.exists(self.basis_output_dir):
                os.makedirs(self.basis_output_dir)
                self.log.debug("Created directory %s" % self.basis_output_dir)

        # TODO: this should be changed such that it does all the computation in
        # a single SVD pass.

        if not self.save_basis_only:

            # Do a quick first pass calculation of all the singular values to get the max on this rank.
            for mi, m in vis.enumerate(axis=0):

                vis_m, mask_m = in_mode_reshape(m, vis[mi], weight[mi], is_mm, self.beamtransfer)

                # If there are fewer than min_modes_for_svd modes per frequency,
                # don't do SVD (corresponding SV will be assumed zero)
                if vis_m.shape[0] < self.min_modes_for_svd:
                    continue

                # Do SVD. u is matrix of msign+nprod singular vectors,
                # vh is matrix of freq singular vectors
                u, sig, vh = svd_em(vis_m, mask_m, niter=self.niter)

                sv_max = max(sig[0], sv_max)

            # Reduce to get the global max.
            global_max = inmodes.comm.allreduce(sv_max, op=MPI.MAX)

            self.log.debug("Global maximum singular value=%g", global_max)

        import sys
        sys.stdout.flush()

        # Loop over all m's and remove modes below the combined cut
        for mi, m in vis.enumerate(axis=0):

            vis_m, mask_m = in_mode_reshape(m, vis[mi], weight[mi], is_mm, self.beamtransfer)

            # If there are fewer than min_modes_for_svd modes per frequency,
            # don't do SVD, write SVD basis, or do any projection
            if vis_m.shape[0] < self.min_modes_for_svd:
                self.log.info('Skipping SVD for m = %d: not enough input modes', m)
                continue

            # Do SVD. u is matrix of msign+nprod singular vectors,
            # vh is matrix of freq singular vectors
            u, sig, vh = svd_em(vis_m, mask_m, niter=self.niter, full_matrices=True)

            # If desired, save complete SVD basis to disk.
            # TODO: Should we use HDF5 chunking or compression here?
            if self.basis_output_dir is not None:

                f = h5py.File(self._svdfile(m, mmax), "w")
                dset_u = f.create_dataset('u', data=u)
                dset_sig = f.create_dataset('sig', data=sig)
                dset_vh = f.create_dataset('vh', data=vh)
                f.close()
                self.log.info("Wrote SVD basis for m=%d to disk", m)

            if not self.save_basis_only:

                # Zero out singular values below the combined mode cut
                global_cut = (sig > self.global_threshold * global_max).sum()
                local_cut = (sig > self.local_threshold * sig[0]).sum()
                cut = max(global_cut, local_cut)
                sig[:cut] = 0.0

                # Recombine the matrix
                vis_m = np.dot(u, np.dot(la.diagsvd(sig, u.shape[1], vh.shape[0]), vh))

                # Reshape and write back into the mmodes container.
                # Need to zero-pad at the end, since original vis array is
                # also heavily zero-padded
                vis_m_reshaped = in_mode_inv_reshape(m, vis_m, is_mm, self.beamtransfer)
                vis[mi] = np.pad(
                    vis_m_reshaped,
                    (0, vis[mi].shape[0]-vis_m_reshaped.shape[0]),
                    'constant',
                    constant_values=0.
                )

        return inmodes

class SVDFilterFromFile(task.SingleTask):
    """SVD filter the m-modes to remove the most correlated components.

    As opposed to SVDFilter, this uses an SVD basis that has been previously
    computed and saved to disk.

    Attributes
    ----------
    local_threshold : float
        Cut out modes with singular value higher than `local_threshold` times the
        largest mode on each m (default: 0.1).
    global_threshold : float
        Remove modes with singular value higher than `global_threshold` times the
        largest mode on any m (default: 0.1).
    basis_input_dir : string
        Directory where SVD basis is stored.
    """

    global_threshold = config.Property(proptype=float, default=0.1)
    local_threshold = config.Property(proptype=float, default=0.1)
    basis_input_dir = config.Property(proptype=str, default=None)

    def _svdfile(self, m, mmax):
        """Filename for SVD basis for single m-mode.
        """
        return external_svdfile(self.basis_input_dir, m, mmax)

    def setup(self, manager):
        """Set the beamtransfer instance (needed if we are operating
        on telescope-SVD modes).

        Parameters
        ----------
        manager : BeamTransfer
            This can also take a ProductManager instance.
        """
        self.beamtransfer = io.get_beamtransfer(manager)

    def process(self, inmodes):
        """Filter m-modes or telescope-SVD modes using an SVD.

        Parameters
        ----------
        inmodes : :class:`containers.MModes` or :class:`containers.SVDModes`
            Input modes to perform the SVD on.

        Returns
        -------
        inmodes : :class:`containers.MModes` or :class:`containers.SVDModes`
            Input modes, either filtered with new SVD or not.
        """
        from mpi4py import MPI

        is_mm = isinstance(inmodes, containers.MModes)

        inmodes.redistribute("m")

        vis = inmodes.vis[:]
        weight = inmodes.weight[:]

        # Dirty way to get mmax: number of m's on each rank, times number of
        # ranks. Is there a better way?
        mmax = vis.shape[0]*mpiutil.size

        # Check whether SVD basis directory exists
        if mpiutil.rank0 and not os.path.exists(self.basis_input_dir):
            raise RuntimeError(
                "Directory %s does not exist: " % self.basis_input_dir
            )

        # Do a quick first pass collection of all the singular values to get
        # the max on this rank.
        sv_max = 0.0
        sv_max_rank = 0.
        for mi, m in vis.enumerate(axis=0):
            # Skip this m if SVD basis file doesn't exist.
            if not os.path.exists(self._svdfile(mi, mmax)):
                continue

            fe = h5py.File(self._svdfile(mi, mmax), 'r')
            ext_sig = fe["sig"][:]
            sv_max_rank = max(ext_sig[0], sv_max_rank)
            fe.close()

        sv_max = mpiutil.world.allreduce(sv_max_rank, op=MPI.MAX)
        self.log.debug("Global maximum singular value=%g", sv_max)

        # Loop over all m's and remove modes below the combined cut
        for mi, m in vis.enumerate(axis=0):
            # Skip this m if SVD basis file doesn't exist.
            if not os.path.exists(self._svdfile(mi, mmax)):
                continue

            # Open SVD basis file for this m, and read U and singular values
            fe = h5py.File(self._svdfile(mi, mmax), 'r')
            u = fe["u"][:]
            sig = fe["sig"][:]
            fe.close()

            # Reshape visibilities to have freq as 2nd axis
            vis_m, mask_m = in_mode_reshape(m, vis[mi], weight[mi], is_mm, self.beamtransfer)

            # Compute combined mode cut
            global_cut = (sig > self.global_threshold * sv_max).sum()
            local_cut = (sig > self.local_threshold * sig[0]).sum()
            cut = max(global_cut, local_cut)
            # Construct identity matrix with zeros corresponding to cut modes
            Z_diag = np.ones(u.shape[0])
            Z_diag[:cut] = 0.0
            Z = np.diag(Z_diag)

            # Filter input visibilities by projecting into SVD basis
            # (with U^\dagger), multiplying by Z (which zeros out unwanted
            # modes), and projecting back to original basis (mult. by U)
            vis_m_filt = np.dot(u, np.dot(Z, np.dot(u.T.conj(), vis_m)))

            # Reshape and write back into the mmodes container.
            # Need to zero-pad at the end, since original vis array is
            # also heavily zero-padded
            vis_m_reshaped = in_mode_inv_reshape(m, vis_m_filt, is_mm, self.beamtransfer)
            vis[mi] = np.pad(
                vis_m_reshaped,
                (0, vis[mi].shape[0]-vis_m_reshaped.shape[0]),
                'constant',
                constant_values=0.
            )

        return inmodes


def external_svdfile(dir, m, mmax):
    """Filename of file containing external SVD basis for a given m.

    Parameters
    ----------
    dir : string
        Directory containing files with SVD basis.
    m : int
        m of desired files
    mmax : int
        mmax of telescope (doesn't need to be exact, only the correct
        power of 10)

    Returns
    -------
    filename : string
        Desired filename.
    """
    pat = os.path.join(dir, "m" + util.natpattern(mmax) + ".hdf5")
    return pat % m

def svd_em(A, mask, niter=5, rank=5, full_matrices=False):
    """Perform an SVD with missing entries using Expectation-Maximisation.

    This assumes that the matrix is well approximated by only a few modes in
    order fill the missing entries. This is probably not a proper EM scheme, but
    is not far off.

    Parameters
    ----------
    A : np.ndarray
        Matrix to SVD.
    mask : np.ndarray
        Boolean array of masked values. Missing values are `True`.
    niter : int, optional
        Number of iterations to perform.
    rank : int, optional
        Set the rank of the approximation used to fill the missing values.
    full_matrices : bool, optional
        Return the full span of eigenvectors and values (see `scipy.linalg.svd`
        for a fuller description).

    Returns
    -------
    u, sig, vh : np.ndarray
        The singular values and vectors.
    """

    # Do an initial fill of the missing entries
    A = A.copy()
    A[mask] = np.median(A)

    # Perform cycles of calculating the SVD with the current guess for the
    # missing values, then forming a new estimate of the missing values using a
    # low rank approximation.
    for i in range(niter):

        u, sig, vh = la.svd(A, full_matrices=full_matrices, overwrite_a=False)

        rank = min(rank,sig.shape[0])

        low_rank_A = np.dot(u[:, :rank] * sig[:rank], vh[:rank])
        A[mask] = low_rank_A[mask]

    return u, sig, vh

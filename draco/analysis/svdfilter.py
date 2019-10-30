"""A set of tasks for SVD filtering the m-modes.

These tasks perform a separate SVD for each m, representing the visibilities
as a (nfreq,msign+nprod) [e.g., frequency vs. baseline and sign of m] matrix.
SVDFilter can return the m-modes with the largest modes filtered out, with
the option to save the SVD basis (modes and singular values) to disk for later
use.

Tasks
=====

.. autosummary::
    :toctree:

    SVDSpectrumEstimator
    SVDFilter
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

from draco.core import task, containers


class SVDSpectrumEstimator(task.SingleTask):
    """Calculate the SVD spectrum of a set of m-modes.

    Attributes
    ----------
    niter : int
        Number of iterations of EM to perform.
    """

    niter = config.Property(proptype=int, default=5)

    def process(self, mmodes):
        """Calculate the spectrum.

        Parameters
        ----------
        mmodes : containers.MModes
            MModes to find the spectrum of.

        Returns
        -------
        spectrum : containers.SVDSpectrum
        """

        mmodes.redistribute("m")

        vis = mmodes.vis[:]
        weight = mmodes.weight[:]

        # The number of modes is the minimum of msign*nprod
        # and nfreq
        nmode = min(vis.shape[1] * vis.shape[3], vis.shape[2])

        spec = containers.SVDSpectrum(singularvalue=nmode, axes_from=mmodes)
        spec.spectrum[:] = 0.0

        for mi, m in vis.enumerate(axis=0):
            self.log.debug("Calculating SVD spectrum of m=%i", m)

            # Transpose vis and weight arrays for a given m from
            # [msign,freq,nprod] to [freq,msign,nprod], reshape into
            # [freq,msign+nprod], and transpose again to [msign+nprod,freq]
            vis_m = (
                vis[mi].view(np.ndarray).transpose((1, 0, 2)).reshape(vis.shape[2], -1).transpose((1,0))
            )
            weight_m = (
                weight[mi]
                .view(np.ndarray)
                .transpose((1, 0, 2))
                .reshape(vis.shape[2], -1)
                .transpose((1,0))
            )
            mask_m = weight_m == 0.0

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
        largest mode on each m.
    global_threshold : float
        Remove modes with singular value higher than `global_threshold` times the
        largest mode on any m
    basis_output_dir : string, optional
        Directory to write SVD basis to, if desired
    save_basis_only : bool, optional
        Whether to compute and save SVD basis, without actually filtering
        the input m-modes (default: False)
    """

    niter = config.Property(proptype=int, default=5)
    global_threshold = config.Property(proptype=float, default=1e-3)
    local_threshold = config.Property(proptype=float, default=1e-2)
    basis_output_dir = config.Property(proptype=str, default=None)
    save_basis_only = config.Property(proptype=bool, default=False)

    def _svdfile(self, m, mmax):
        """Filename for SVD basis for single m-mode.
        """
        return external_svdfile(self.basis_output_dir, m, mmax)

    def process(self, mmodes):
        """Filter MModes using an SVD.

        Parameters
        ----------
        mmodes : container.MModes

        Returns
        -------
        mmodes : container.MModes
        """

        from mpi4py import MPI

        mmodes.redistribute("m")

        vis = mmodes.vis[:]
        weight = mmodes.weight[:]

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

                # Transpose vis and weight arrays for a given m from
                # [msign,freq,nprod] to [freq,msign,nprod], reshape into
                # [freq,msign+nprod], and transpose again to [msign+nprod,freq]
                vis_m = (
                    vis[mi].view(np.ndarray).transpose((1, 0, 2)).reshape(vis.shape[2], -1).transpose((1,0))
                )
                weight_m = (
                    weight[mi]
                    .view(np.ndarray)
                    .transpose((1, 0, 2))
                    .reshape(vis.shape[2], -1)
                    .transpose((1,0))
                )
                mask_m = weight_m == 0.0

                # Do SVD. u is matrix of msign+nprod singular vectors,
                # vh is matrix of freq singular vectors
                u, sig, vh = svd_em(vis_m, mask_m, niter=self.niter)

                sv_max = max(sig[0], sv_max)

        # Reduce to get the global max.
        global_max = mmodes.comm.allreduce(sv_max, op=MPI.MAX)

        self.log.debug("Global maximum singular value=%.2g", global_max)
        import sys

        sys.stdout.flush()

        # Loop over all m's and remove modes below the combined cut
        for mi, m in vis.enumerate(axis=0):

            # Transpose vis and weight arrays for a given m from
            # [msign,freq,nprod] to [freq,msign,nprod], reshape into
            # [freq,msign+nprod], and transpose again to [msign+nprod,freq]
            vis_m = (
                vis[mi].view(np.ndarray).transpose((1, 0, 2)).reshape(vis.shape[2], -1).transpose((1,0))
            )
            weight_m = (
                weight[mi]
                .view(np.ndarray)
                .transpose((1, 0, 2))
                .reshape(vis.shape[2], -1)
                .transpose((1,0))
            )
            mask_m = weight_m == 0.0

            # Do SVD. u is matrix of msign+nprod singular vectors,
            # vh is matrix of freq singular vectors
            u, sig, vh = svd_em(vis_m, mask_m, niter=self.niter)

            # If desired, save complete SVD basis to disk.
            # TODO: Should we use HDF5 chunking or compression here?
            if self.basis_output_dir is not None:

                f = h5py.File(self._svdfile(m, mmax), "w")
                dset_u = f.create_dataset('u', data=u)
                dset_sig = f.create_dataset('sig', data=sig)
                dset_vh = f.create_dataset('vh', data=vh)
                f.close()
                self.log.info("Wrote SVD basis for m=%g to disk", m)

            if not self.save_basis_only:

                # Zero out singular values below the combined mode cut
                global_cut = (sig > self.global_threshold * global_max).sum()
                local_cut = (sig > self.local_threshold * sig[0]).sum()
                cut = max(global_cut, local_cut)
                sig[:cut] = 0.0

                # Recombine the matrix
                vis_m = np.dot(u, sig[:, np.newaxis] * vh)

                # Reshape and write back into the mmodes container
                vis[mi] = vis_m.transpose((1,0)).reshape(vis.shape[2], 2, -1).transpose((1, 0, 2))

        return mmodes

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

        low_rank_A = np.dot(u[:, :rank] * sig[:rank], vh[:rank])
        A[mask] = low_rank_A[mask]

    return u, sig, vh

"""A set of tasks for SVD filtering the m-modes."""

import numpy as np
import scipy.linalg as la

from caput import config

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

        nmode = min(vis.shape[1] * vis.shape[3], vis.shape[2])

        spec = containers.SVDSpectrum(singularvalue=nmode, axes_from=mmodes)
        spec.spectrum[:] = 0.0

        for mi, m in vis.enumerate(axis=0):
            self.log.debug("Calculating SVD spectrum of m=%i", m)

            vis_m = vis.local_array[mi].transpose((1, 0, 2)).reshape(vis.shape[2], -1)
            weight_m = (
                weight.local_array[mi].transpose((1, 0, 2)).reshape(vis.shape[2], -1)
            )
            mask_m = weight_m == 0.0

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
    """

    niter = config.Property(proptype=int, default=5)
    global_threshold = config.Property(proptype=float, default=1e-3)
    local_threshold = config.Property(proptype=float, default=1e-2)

    def process(self, mmodes):
        """Filter MModes using an SVD.

        Parameters
        ----------
        mmodes : container.MModes
            MModes to process

        Returns
        -------
        mmodes : container.MModes
        """
        from mpi4py import MPI

        mmodes.redistribute("m")

        vis = mmodes.vis[:]
        weight = mmodes.weight[:]

        sv_max = 0.0

        # TODO: this should be changed such that it does all the computation in
        # a single SVD pass.

        # Do a quick first pass calculation of all the singular values to get the max on this rank.
        for mi, m in vis.enumerate(axis=0):
            vis_m = vis.local_array[mi].transpose((1, 0, 2)).reshape(vis.shape[2], -1)
            weight_m = (
                weight.local_array[mi].transpose((1, 0, 2)).reshape(vis.shape[2], -1)
            )
            mask_m = weight_m == 0.0

            u, sig, vh = svd_em(vis_m, mask_m, niter=self.niter)

            sv_max = max(sig[0], sv_max)

        # Reduce to get the global max.
        global_max = mmodes.comm.allreduce(sv_max, op=MPI.MAX)

        self.log.debug("Global maximum singular value=%.2g", global_max)
        import sys

        sys.stdout.flush()

        # Loop over all m's and remove modes below the combined cut
        for mi, m in vis.enumerate(axis=0):
            vis_m = vis.local_array[mi].transpose((1, 0, 2)).reshape(vis.shape[2], -1)
            weight_m = (
                weight.local_array[mi].transpose((1, 0, 2)).reshape(vis.shape[2], -1)
            )
            mask_m = weight_m == 0.0

            u, sig, vh = svd_em(vis_m, mask_m, niter=self.niter)

            # Zero out singular values below the combined mode cut
            global_cut = (sig > self.global_threshold * global_max).sum()
            local_cut = (sig > self.local_threshold * sig[0]).sum()
            cut = max(global_cut, local_cut)
            sig[:cut] = 0.0

            # Recombine the matrix
            vis_m = np.dot(u, sig[:, np.newaxis] * vh)

            # Reshape and write back into the mmodes container
            vis[mi] = vis_m.reshape(vis.shape[2], 2, -1).transpose((1, 0, 2))

        return mmodes


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
    A[mask] = np.median(A[~mask])

    # Perform cycles of calculating the SVD with the current guess for the
    # missing values, then forming a new estimate of the missing values using a
    # low rank approximation.
    for i in range(niter):
        u, sig, vh = la.svd(A, full_matrices=full_matrices, overwrite_a=False)

        low_rank_A = np.dot(u[:, :rank] * sig[:rank], vh[:rank])
        A[mask] = low_rank_A[mask]

    return u, sig, vh

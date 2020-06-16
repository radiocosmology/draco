"""A set of tasks for SVD filtering the m-modes.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
import scipy.linalg as la

from caput import config
from caput.pipeline import PipelineConfigError

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
        mmodes : containers.MModes or containers.HybridVisMModes
            MModes to find the spectrum of.

        Returns
        -------
        spectrum : containers.SVDSpectrum or containers.HybridVisSVDSpectrum
        """

        contmap = {
            containers.MModes: containers.SVDSpectrum,
            containers.HybridVisMModes: containers.HybridVisSVDSpectrum,
        }

        mmodes.redistribute("m")

        vis = mmodes.vis[:]
        weight = mmodes.weight[:]

        # transform freq against baseline stack / beamformed pixel
        f_ax = list(mmodes.vis.attrs["axis"]).index("freq")
        nmode = min(vis.shape[-1], vis.shape[f_ax])

        out_cont = contmap[mmodes.__class__]
        spec = out_cont(singularvalue=nmode, axes_from=mmodes)
        spec.spectrum[:] = 0.0

        for mi, m in vis.enumerate(axis=0):
            self.log.debug("Calculating SVD spectrum of m=%i", m)

            vis_m = np.moveaxis(
                vis[mi].view(np.ndarray), f_ax - 1, -2
            ).reshape(-1, vis.shape[f_ax], vis.shape[-1])
            weight_m = np.moveaxis(
                weight[mi].view(np.ndarray), f_ax - 1, -2
            ).reshape(-1, vis.shape[f_ax], vis.shape[-1])
            mask_m = weight_m == 0.0

            sig_out = np.zeros((vis_m.shape[0], nmode), dtype=vis_m.dtype)
            for i in range(vis_m.shape[0]):
                u, sig, vh = svd_em(vis_m[i], mask_m[i], niter=self.niter)
                sig_out[i] = sig

            spec.spectrum[m] = sig_out.reshape(spec.spectrum.shape[1:])

        return spec


class SVDFilter(task.SingleTask):
    """SVD filter the m-modes to remove the most correlated components.

    Attributes
    ----------
    niter : int (default 5)
        Number of iterations of EM to perform.
    local_threshold : float (default 1e-2)
        Cut out modes with singular value higher than `local_threshold` times the
        largest mode on each m.
    global_threshold : float (default 1e-3)
        Remove modes with singular value higher than `global_threshold` times the
        largest mode on any m
    fixed_cut : int (default 0)
        If positive, ignore thresholds and just cut a fixed number of modes for
        every m.
    """

    niter = config.Property(proptype=int, default=5)
    global_threshold = config.Property(proptype=float, default=1e-3)
    local_threshold = config.Property(proptype=float, default=1e-2)
    fixed_cut = config.Property(proptype=int, default=0)

    def process(self, mmodes):
        """Filter MModes using an SVD.

        Parameters
        ----------
        mmodes : containers.MModes or containers.HybridVisMModes

        Returns
        -------
        mmodes : containers.MModes or containers.HybridVisMModes
        """

        from mpi4py import MPI

        mmodes.redistribute("m")

        vis = mmodes.vis[:]
        weight = mmodes.weight[:]

        f_ax = list(mmodes.vis.attrs["axis"]).index("freq")

        sv_max = 0.0

        if self.fixed_cut > 0:
            self.log.debug(f"Will cut {self.fixed_cut} modes for every m.")
        elif self.fixed_cut < 0:
            raise PipelineConfigError("'fixed_cut' parameter cannot be negative.")


        # TODO: this should be changed such that it does all the computation in
        # a single SVD pass.

        if self.fixed_cut == 0:
            # Do a quick first pass calculation of all the singular values to get the max on this rank.
            for mi, m in vis.enumerate(axis=0):

                # Reorder array to have frequency and baseline/pixel at the end
                vis_m = np.moveaxis(
                    vis[mi].view(np.ndarray), f_ax - 1, -2
                ).reshape(-1, vis.shape[f_ax], vis.shape[-1])
                weight_m = np.moveaxis(
                    weight[mi].view(np.ndarray), f_ax - 1, -2
                ).reshape(-1, vis.shape[f_ax], vis.shape[-1])
                mask_m = weight_m == 0.0

                # Iterate over remaining axes
                for i in range(vis_m.shape[0]):
                    u, sig, vh = svd_em(vis_m[i], mask_m[i], niter=self.niter)

                    sv_max = max(sig[0], sv_max)

            # Reduce to get the global max.
            global_max = mmodes.comm.allreduce(sv_max, op=MPI.MAX)

            self.log.debug("Global maximum singular value=%.2g", global_max)
            import sys

            sys.stdout.flush()
        else:
            global_max = 0.0

        # Loop over all m's and remove modes below the combined cut
        for mi, m in vis.enumerate(axis=0):

            # Reorder array to have frequency and baseline/pixel at the end
            vis_m = np.moveaxis(
                vis[mi].view(np.ndarray), f_ax - 1, -2
            ).reshape(-1, vis.shape[f_ax], vis.shape[-1])
            weight_m = np.moveaxis(
                weight[mi].view(np.ndarray), f_ax - 1, -2
            ).reshape(-1, vis.shape[f_ax], vis.shape[-1])
            mask_m = weight_m == 0.0

            # Iterate over remaining axes
            for i in range(vis_m.shape[0]):
                u, sig, vh = svd_em(vis_m[i], mask_m[i], niter=self.niter)

                # Zero out singular values below the combined mode cut
                if self.fixed_cut > 0:
                    cut = self.fixed_cut
                else:
                    global_cut = (sig > self.global_threshold * global_max).sum()
                    local_cut = (sig > self.local_threshold * sig[0]).sum()
                    cut = max(global_cut, local_cut)
                sig[:cut] = 0.0

                # Recombine the matrix
                vis_m[i] = np.dot(u, sig[:, np.newaxis] * vh)

            # Reconstruct shape with freq axis second-to-last
            sh = vis.shape
            new_sh = sh[1:f_ax] + sh[f_ax + 1:-1] + (sh[f_ax], sh[-1])
            # Reshape and write back into the mmodes container
            vis[mi] = np.moveaxis(vis_m.reshape(new_sh), -2, f_ax - 1)

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
    A[mask] = np.median(A)

    # Perform cycles of calculating the SVD with the current guess for the
    # missing values, then forming a new estimate of the missing values using a
    # low rank approximation.
    for i in range(niter):

        u, sig, vh = la.svd(A, full_matrices=full_matrices, overwrite_a=False)

        low_rank_A = np.dot(u[:, :rank] * sig[:rank], vh[:rank])
        A[mask] = low_rank_A[mask]

    return u, sig, vh

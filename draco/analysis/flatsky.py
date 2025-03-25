
import numpy as np
from scipy.special import lpn
from caput import config
import healpy as hp
from ducc0 import sht
from cora.util import hputil, coord

from ..core import io, task


class RingmapSHT(task.SingleTask):

    lmax = config.Property(proptype=int, default=512)

    def setup(self, manager):

        self.telescope = io.get_telescope(manager)

    def process(self, rmap):
        # TODO consider supporting m-modes as input

        # handle containers manipulation

        alm = _ringmap2alm(
            rmap.map[:], el=rmap.el[:], lmax=self.lmax,
            obs_lat=np.radians(self.telescope.latitude)
        )

        alm_cont = None

        return alm_cont


def _ringmap2alm(rmap, el, obs_lat, lmax=None):
    """expect a ringmap with dimensions [nmap, nel, nra]"""
    # ring latitude
    theta = np.pi / 2 - (np.arcsin(el) + obs_lat)

    # number of RA samples
    nra = rmap.shape[-1]

    # perform SHT
    res = sht.adjoint_synthesis(
        map=rmap.reshape((rmap.shape[0], -1)),
        theta=theta,
        nphi=np.ones(theta.size, dtype=np.uint64) * nra,
        phi0=np.zeros(theta.size),
        ringstart=np.arange(theta.size, dtype=np.uint64) * nra,
        spin=0,
        lmax=lmax,
        #nthreads=8,
    )

    return res / nra  # normalise m-transform


def _mmode2alm(mmodes, el, obs_lat, lmax=None):
    """expect a hybrid-vis m-modes with dimensions [nmap, nel, nm]"""
    # ring latitude
    theta = np.pi / 2 - (np.arcsin(el) + obs_lat)

    # infer lmax
    if lmax is None:
        lmax = mmodes.shape[-1] - 1

    return sht.leg2alm(
        leg=mmodes[..., :lmax + 1],
        theta=theta,
        spin=0,
        lmax=lmax,
        #nthreads=8,
    )


def _alm2lmax(N):
    lmax = 3 * (np.sqrt(1 + 8 / 9 * (N - 1)) - 1) / 2
    if (int(lmax) != lmax):
        raise ValueError(
            f"There is no dimension that corresponds to an upper triangular matrix of size {N}."
        )
    return int(lmax)


def _cross_ang_ps(a, b, lmax):
    # unpack the alm to matrix form
    lmax_a = _alm2lmax(a.size)
    lmax_b = _alm2lmax(b.size)
    a_2d = hputil.unpack_alm(a, lmax_a)
    b_2d = hputil.unpack_alm(b, lmax_b)

    # normalise by the number of m
    mnorm = 1.0 / (np.arange(lmax + 1) + 1)

    return np.sum(a_2d[:lmax + 1, :lmax + 1] * b_2d[:lmax + 1, :lmax + 1].conj(), axis=1) * mnorm


def _ps2corr(Cl, lmax, theta):
    """Compute the angular correlation function from the power spectrum."""
    if (lmax + 1) > Cl.size:
        raise ValueError(f"lmax of {lmax} too large for Cl with size {Cl.size}.")
    Pl = lpn(lmax, np.cos(theta))[0]
    ell = np.arange(lmax + 1)
    return np.sum(
            ((2 * ell + 1) * Cl[:lmax + 1])[:, np.newaxis] * Pl,
        axis=0
    ) / np.pi / 4

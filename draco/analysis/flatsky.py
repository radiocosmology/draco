
import numpy as np
from ducc0 import sht
from scipy.special import lpn
from caput import config
import healpy as hp
from cora.util import hputil, coord

from ..core import io, task, containers


class RingmapSHT(task.SingleTask):

    lmax = config.Property(proptype=int, default=2048)

    def setup(self, manager):

        self.telescope = io.get_telescope(manager)

    def process(self, rmap):

        rm_view = rmap.map[:].reshape((-1, rmap.map.shape[-2:]))
        n_lm = self.lmax * (self.lmax + 1)
        alm = np.zeros((rm_view.shape[0], n_lm))
        for i in range(alm.shape[0]):
            alm[i] = _ringmap2alm(
                rm_view[i], el=rmap.el[:], lmax=self.lmax,
                obs_lat=np.radians(self.telescope.latitude)
            )

        return containers.SHCoeff(axes_from=rmap, alm=alm, lm=np.arange(alm.shape[-1]))


class ProjectToPatches(task.SingleTask):

    nside = config.Property(proptype=int, default=6)
    width = config.Property(proptype=float, default=10)
    npix = config.Property(proptype=int, default=64)
    lmax = config.Property(proptype=int, default=2048)

    def setup(self, manager, rmap=None):
        telescope = io.get_telescope(manager)
        self.lat = telescope.latitude
        if rmap is None:
            self._el_range = [-1.0, 1.0]
        else:
            self._el_range = rmap.el.min(), rmap.el.max()

    def process(self, alm):
        # get sky area and distribute patches
        dec_range = np.degrees(np.arcsin(self._el_range)) + self.lat
        patch_center = _tile_patches(self.nside, dec_range)

        # generate square grid coordinates
        n = self.npix
        polar_grid = _flat_grid(np.radians(self.width), n, n)
        tx, ty = (np.arange(n) / n - 0.5) * self.width, (np.arange(n) / n - 1.5) * self.width

        # do the projection for every freq/pol
        alm_view = alm.alm[:].reshape((-1,) + alm.alm.shape[-2:])
        rm_patches = np.zeros((alm_view.shape[0], patch_center.shape[0], n, n))
        for i in range(patch_center.shape[0]):
            # rotate the square grid to patch location
            p_rot = np.radians(patch_center[i])
            p_grid = _rotate_patch_grid(polar_grid, p_rot)
            # ensure coordinates are within bounds
            p_grid[:, 1] = np.where(p_grid[:, 1] < 0, p_grid[:, 1] + 2 * np.pi, p_grid[:, 1])
            for j in range(alm_view.shape[0]):
                # project map onto patch
                rm_patches[j, i] = _project_on_patch(alm_view[j], p_grid, lmax=self.lmax).reshape((n, n))

        # return a patches container
        final_shape = alm.alm.shape[:2] + rm_patches.shape[1:]
        patches = containers.TiledPatches(
            axes_from=alm,
            map=rm_patches.reshape(final_shape),
            patch=np.arange(patch_center.shape[0]),
            patch_center=patch_center,
            x=tx,
            y=ty
        )
        return patches


class CorrelatePatches(task.SingleTask):

    do_apod = config.Property(proptype=bool, default=True)
    permute = config.Property(proptype=int, default=None)
    mask_frac = config.Property(proptype=float, default=0.5)

    def setup(self):
        self.rng = np.random.default_rng(seed=self.permute)

    def process(self, a, b):
        if a.map.shape != b.map.shape:
            raise ValueError(
                f"Shapes of map patches don't match: {a.map.shape} != {b.map.shape}"
            )

        n = a.map.shape[-1]  # assumes square patch

        # flag based on fraction masked
        a_flagged = np.sum(a.weight[:] != 0, axis=(-2, -1)) < self.mask_frac * n**2
        b_flagged = np.sum(b.weight[:] != 0, axis=(-2, -1)) < self.mask_frac * n**2
        flagged = a_flagged | b_flagged

        # set apodisation
        if self.do_apod:
            apod = np.hanning(n)[np.newaxis, :] * np.hanning(n)[:, np.newaxis]
        else:
            apod = np.ones((n, n))

        # FFT of patches
        a_ft = np.fft.fft2(a.map[:] * apod)
        b_ft = np.fft.fft2(b.map[:] * apod)

        # u-v samples
        u = 2 * np.pi * np.fft.fftfreq(a_ft.shape[-1], np.abs(a_ft.x[1] - a_ft.x[0]))
        v = 2 * np.pi * np.fft.fftfreq(a_ft.shape[-2], np.abs(a_ft.y[1] - a_ft.y[0]))

        # get the next permutation if configured
        if self.permute is None:
            perm = slice(None)
        else:
            perm = self.rng.permutation(a.shape[-3] - flagged.sum())

        # correlate
        corr = np.mean(a_ft[~flagged][perm] * b_ft[~flagged].conj(), axis=-3)

        return containers.PowerSpectrum3D(axes_from=a, spectrum=corr, u=u, v=v)


def _ringmap2alm(rmap, el, obs_lat, lmax=None, epsilon=1e-3, maxiter=100):
    """expect a ringmap with dimensions [nmap, nel, nra]"""
    # ring latitude
    theta = np.pi / 2 - (np.arcsin(el) + obs_lat)

    # number of RA samples
    nra = rmap.shape[-1]

    # perform SHT
    res = sht.pseudo_analysis(
        map=rmap.reshape((1, -1)),  # 1-component, flattened array
        theta=theta,
        nphi=np.ones(theta.size, dtype=np.uint64) * nra,
        phi0=np.zeros(theta.size),
        ringstart=np.arange(theta.size, dtype=np.uint64) * nra,
        spin=0,
        lmax=lmax,
        maxiter=maxiter,
        epsilon=epsilon,
    )

    print(f"pseudo_analysis finished with status {res[1]} after {res[2]} iterations "
          f"and residual {res[3]}.")

    return res[0]


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


def _tile_patches(nside, dec_range):
    # tile the sky using HEALpix
    print(f"Resolution: {np.degrees(hp.nside2resol(nside))} deg")
    npatch = hp.nside2npix(nside)
    print(f"Num pix: {npatch}")

    # each HEALpix cell will be center of a flat patch
    patch_center = np.array(hp.pix2ang(nside, np.arange(npatch), lonlat=True))

    # only keep those in the given range
    patch_sel = np.where((patch_center[1] > dec_range[0]) & (patch_center[1] < dec_range[1]))[0]

    # return an array as (npatch, theta, phi)
    return np.concatenate(
        (patch_center[1][patch_sel, np.newaxis], patch_center[0][patch_sel, np.newaxis]),
        axis=1
    )


def _flat_grid(patch_width, nx, ny=None):
    """expect patch_width in radians"""
    if ny is None:
        ny = nx

    tx, ty = np.meshgrid(
        (np.arange(nx) - nx // 2) * patch_width / nx,
        (np.arange(ny) - ny // 2) * patch_width / ny
    )

    # return an array as (npix, theta, phi)
    return np.concatenate(
        ((np.pi / 2 - ty).flatten()[:, np.newaxis], tx.flatten()[:, np.newaxis]),
        axis=1,
    )


def _rotate_patch_grid(flat_grid, rot):
    """flat grid is expected in (theta, phi) polar coord"""
    vec_grid = coord.sph_to_cart(flat_grid)
    rmat = np.dot(
        # azimuth rotation
        np.array([
            [np.cos(rot[1]), -np.sin(rot[1]), 0.],
            [np.sin(rot[1]), np.cos(rot[1]), 0.],
            [0., 0., 1.],

        ]),
        # polar rotation first
        np.array([
            [np.cos(rot[0]), 0., -np.sin(rot[0])],
            [0., 1., 0.],
            [np.sin(rot[0]), 0., np.cos(rot[0])],
        ]),
    )
    rot_grid = np.matmul(rmat[np.newaxis, ...], vec_grid[:, :, np.newaxis])[..., 0]
    return coord.cart_to_sph(rot_grid)[:, 1:]


def _project_on_patch(alm, grid, lmax=None, epsilon=1e-9, nthreads=16):
    if lmax is None:
        lmax = hp.Alm.getlmax(alm.size)
    return sht.synthesis_general(
        alm=alm,
        loc=grid,
        spin=0,
        lmax=lmax,
        epsilon=epsilon,
        nthreads=nthreads,
    )

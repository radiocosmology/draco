"""Take a quasar catalog and a 21cm simulated map and generate mock 
catalogs correlated to the 21cm maps and following a selection function
derived from the original catalog.

Pipeline tasks
==============

.. autosummary::
    :toctree:

    SelFuncEstimator
    PdfGenerator
    MockQCatGenerator

Internal functions
==================

.. autosummary::
    :toctree:

    _zlims_to_freq
    _freq_to_z
    _z_to_freq
    _pix_to_radec
    _radec_to_pix

Usage
=====

Generally you would want to use these tasks together. Providing a catalog
path to :class:`SelFuncEstimator`, then feeding the resulting selection
function to :class:`PdfGenerator` and finally passing the resulting
probability distribution function to :class:`MockQCatGenerator` to generate
mock catalogs. Below is an example of yaml file to generate mock catalogs:

>>> spam_config = '''
... pipeline :
...     tasks:
...         -   type:   draco.synthesis.mockcatalog.SelFuncEstimator
...             params: selfunc_params
...             out:    selfunc
... 
...         -   type:     draco.synthesis.mockcatalog.PdfGenerator
...             params:   pdf_params
...             requires: selfunc
...             out:      pdf_map
... 
...         -   type:     draco.synthesis.mockcatalog.MockQCatGenerator
...             params:   mqcat_params
...             requires: pdf_map
...             out:      mockqcat
... 
... selfunc_params:
...     bqcat_path: '/bg01/homescinet/k/krs/jrs65/sdss_quasar_catalog.h5'
...     nside: 16
... 
... pdf_params:
...     qsomaps_path: '/scratch/k/krs/fandino/xcorrSDSS/sim21cm/21cmmap.hdf5'
... 
... mqcat_params:
...     nqsos: 200000
...     ncats: 5
...     save: True
...     output_root: '/scratch/k/krs/fandino/test_mqcat/mqcat'

'''


"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
import healpy as hp

from cora.signal import corr21cm
from cora.util import units
from caput import config
from caput import mpiarray, mpiutil
from ..core import task, containers
from mpi4py import MPI


# Pipeline tasks
# --------------


class SelFuncEstimator(task.SingleTask):
    """Takes a quasar catalog as input and returns an estimate of the
    selection function based on a low rank SVD reconstruction.

    Attributes
    ----------
    bqcat_path : str
        Full path to base quasar catalog.
    nside : int
        NSIDE for catalog maps generated for the SVD.
    n_z : int
        Number of redshift bins for catalog maps generated for the SVD.
    n_modes : int
        Number of modes used in recovering the selection function from
        base catalogue maps SVD.
    z_stt : float
        Starting redshift for catalog maps generated for the SVD.
    z_stp : float
        Stopping redshift for catalog maps generated for the SVD.

    """

    bqcat_path = config.Property(proptype=str)
    density_maps_path = config.Property(proptype=str, default="")

    # These seem to be optimal parameters and should not
    # usually need to be changed from the default values:
    nside = config.Property(proptype=int, default=16)
    n_z = config.Property(proptype=int, default=32)
    n_modes = config.Property(proptype=int, default=7)
    z_stt = config.Property(proptype=float, default=0.8)
    z_stp = config.Property(proptype=float, default=2.5)

    def setup(self):
        """
        """
        # Load base quasar catalog from file:
        self._base_qcat = containers.SpectroscopicCatalog.from_file(self.bqcat_path)

        if self.density_maps_path != "":
            densitymaps = containers.Map.from_file(
                self.density_maps_path, distributed=True
            )
            densityz = _freq_to_z(densitymaps.freq)
            # If density map is given overrride z_stt and z_stp
            idx_stt = np.argmin(densityz["centre"])
            idx_stp = np.argmax(densityz["centre"])
            self.z_stt = densityz["centre"][idx_stt] - 0.5 * densityz["width"][idx_stt]
            self.z_stp = densityz["centre"][idx_stp] + 0.5 * densityz["width"][idx_stp]

    def process(self):
        """Put the base catalog into maps. SVD the maps and recover
        with a small number of modes. This smoothes out the distribution
        quasars and provides an estimate of the selection function used.
        """
        # Number of pixels to use in catalog maps for SVD
        n_pix = hp.pixelfunc.nside2npix(self.nside)

        # Redshift bins edges
        zlims_selfunc = np.linspace(self.z_stt, self.z_stp, self.n_z + 1)
        # Redshift bins centre
        z_selfunc = (zlims_selfunc[:-1] + zlims_selfunc[1:]) * 0.5

        # Have to transform the z axis to a freq axis to return selfunc in
        # containers.Map format:
        freq_selfunc = _zlims_to_freq(z_selfunc, zlims_selfunc)

        # Create maps from original catalog:
        # No point in distributing this in a mpi clever way
        # because I need to SVD it.
        maps = np.zeros((self.n_z, n_pix))
        # Indices of each qso in z axis:
        idxs = (
            np.digitize(self.base_qcat["redshift"]["z"], zlims_selfunc) - 1
        )  # -1 to get indices
        # Map pixel of each qso
        pixls = _radec_to_pix(
            self.base_qcat["position"]["ra"],
            self.base_qcat["position"]["dec"],
            self.nside,
        )
        for jj in range(self.n_z):
            zpixls = pixls[idxs == jj]  # Map pixels of qsos in z bin jj
            for kk in range(n_pix):
                # Number of quasars in z bin jj and pixel kk
                maps[jj, kk] = np.sum(zpixls == kk)

        # SVD the quasar density maps:
        svd = np.linalg.svd(maps, full_matrices=0)

        # Map container to store the selection function:
        self._selfunc = containers.Map(
            nside=self.nside, polarisation=False, freq=freq_selfunc
        )
        # Start as zeroes:
        self._selfunc["map"][:, :, :] = np.zeros(self._selfunc["map"].local_shape)

        # Get axis parameters for distributed map:
        lo = self._selfunc["map"][:, 0, :].local_offset[0]
        ls = self._selfunc["map"][:, 0, :].local_shape[0]
        # Recover the n_modes approximation to the original catalog maps:
        for jj in range(self.n_modes):
            uj = svd[0][:, jj]
            sj = svd[1][jj]
            vj = svd[2][jj, :]
            # Re-constructed mode (distribute in freq/redshift):
            recmode = mpiarray.MPIArray.wrap(
                (uj[:, None] * sj * vj[None, :])[lo : lo + ls], axis=0
            )
            self._selfunc["map"][:, 0, :] = self._selfunc["map"][:, 0, :] + recmode
        # Remove negative entries remaining from SVD recovery:
        self._selfunc["map"][np.where(self._selfunc.map[:] < 0.0)] = 0.0

        self.done = True

        return self.selfunc

    def process_finish(self):
        """
        """
        return None

    @property
    def base_qcat(self):
        return self._base_qcat

    @property
    def selfunc(self):
        return self._selfunc


class PdfGenerator(task.SingleTask):
    """Take a quasar catalog selection function and simulated Quasar
    (biased density) maps and return a PDF map correlated with the 
    density maps and the selection function. This PDF map can be used 
    by the task :class:`MockQCatGenerator` to draw mock catalogs.

    Attributes
    ----------
    qsomaps_path : str
        Full path to simulated QSO maps.
    random_catalog : bool
        Is True generate random catalogs, not correlated with the maps.
        Default is False.

    """

    qsomaps_path = config.Property(proptype=str)
    random_catalog = config.Property(proptype=bool, default=False)

    def setup(self, selfunc):
        """
        """
        self.selfunc = selfunc

        # Load QSO CORA maps from file:
        qsomaps = containers.Map.from_file(self.qsomaps_path, distributed=True)
        if self.random_catalog:
            # To make a random (not correlated) catalog
            qsomaps.map[:] = np.zeros_like(qsomaps.map)

        self.qsomaps = qsomaps  # Setter sets other parameters too

        # For easy access to communicator:
        self.comm_ = self.qsomaps.comm
        self.rank = self.comm_.Get_rank()  # Unused for now

    def process(self):
        """
        """
        # From frequency to redshift:
        z = _freq_to_z(self.qsomaps.freq)
        n_z = len(z)

        # Freq to redshift of selection function:
        z_selfunc = _freq_to_z(self.selfunc.freq)

        # Re-distribute maps in pixels:
        self.qsomaps.redistribute(dist_axis=2)

        # TODO: Change h1maps for something more generic, like density_maps

        rho_m = mpiarray.MPIArray.wrap(self.qsomaps.map[:, 0, :] + 1.0, axis=1)

        # Re-distribute in frequencies before normalizing
        # (which requires summing over pixels)
        # Note: mpiarray.MPIArray.redistribute returns the redistributed array
        # That's different from the behaviour of the containers.
        rho_m = rho_m.redistribute(axis=0)

        # Normalize density to have unit mean in each z-bin:
        rho_m = mpiarray.MPIArray.wrap(
            rho_m / np.mean(rho_m, axis=1)[:, np.newaxis], axis=0
        )

        # Resizing the selection function to match the voxel size of the
        # CORA maps by hand. Result is distributed in axis 0.
        resized_selfunc = self._resize_map(
            self.selfunc.map[:, 0, :], rho_m.global_shape, z, z_selfunc
        )

        # Generate wheights for correct distribution of quasars in redshift:
        z_wheights = np.sum(resized_selfunc, axis=1)
        # Sum wheights in each comm rank. Need array of scalar here.
        z_total_temp = mpiarray.MPIArray.wrap(
            np.array([np.sum(z_wheights, axis=0)]), axis=0
        )
        z_total = np.zeros_like(z_total_temp)

        # Sum accross ranks. All ranks get the same result:
        self.comm_.Allreduce(z_total_temp, z_total)
        # Normalize z_wheights:
        z_wheights = mpiarray.MPIArray.wrap(z_wheights / z_total, axis=0)

        # PDF following selection function and CORA maps:
        # (both rho_m and resized_selfunc are distributed in axis 0)
        pdf = rho_m * resized_selfunc

        # Enforce redshift distribution to follow selection function:
        pdf = mpiarray.MPIArray.wrap(
            pdf / np.sum(pdf, axis=1)[:, np.newaxis] * z_wheights[:, np.newaxis], axis=0
        )

        # Put PDF in a map container:
        pdf_map = containers.Map(
            nside=self._nside, polarisation=False, freq=self.qsomaps.freq
        )

        # I am not sure I need this test every time:
        if pdf_map["map"].local_offset[0] == pdf.local_offset[0]:
            pdf_map["map"][:, 0, :] = pdf
        else:
            raise RuntimeError("Local offsets don't match.")

        self.done = True
        return pdf_map

    def process_finish(self):
        """
        """
        return None

    def _resize_map(self, map, new_shape, z_new, z_old):
        """Re-size map (np.array) to new shape, taking into account
        mpi distribution.
        """
        from ..util import regrid

        # redistribute in axis 1 to re-size axis 0:
        map = mpiarray.MPIArray.wrap(map, axis=0)
        map = map.redistribute(axis=1)

        # Form interpolation matrix:
        interp_m = regrid.lanczos_forward_matrix(z_old["centre"], z_new["centre"])
        # Correct for redshift bin widths:
        interp_m = (
            interp_m / z_old["width"][np.newaxis, :] * z_new["width"][:, np.newaxis]
        )
        # Resize axis 0:
        map = np.dot(interp_m, map)
        map = mpiarray.MPIArray.wrap(np.array(map), axis=1)
        # redistribute in axis 0 to re-size axis 1:
        map = map.redistribute(axis=0)

        # Resize axis 1:
        # new_shape is a global shape, so can only use it in axis 1 here
        resized_map = np.zeros((map.local_shape[0], new_shape[1]))
        n_side = hp.pixelfunc.npix2nside(new_shape[1])  # NSIDE of new shape
        for ii in range(map.local_shape[0]):
            resized_map[ii] = hp.pixelfunc.ud_grade(map[ii, :], nside_out=n_side)

        # Remove negative values. (The Lanczos kernel can make things
        # slightly negative at the edges)
        resized_map = np.where(
            resized_map >= 0.0, resized_map, np.zeros_like(resized_map)
        )

        return mpiarray.MPIArray.wrap(resized_map, axis=0)

    @property
    def qsomaps(self):
        return self._qsomaps

    @qsomaps.setter
    def qsomaps(self, qsomaps):
        """
        Setter for qsomaps
        Also set the attributes:
            self._npix : Number of pixels in CORA QSO maps 
            self._nside : NSIDE of CORA QSO maps

        """
        if isinstance(qsomaps, containers.Map):
            self._qsomaps = qsomaps
            self._npix = len(self._qsomaps.index_map["pixel"])
            self._nside = hp.pixelfunc.npix2nside(self._npix)
        else:
            msg = (
                "qsomaps is not an instance of "
                + "draco.core.containers.Map\n"
                + "Value for _qsomaps not set."
            )
            print(msg)


class MockQCatGenerator(task.SingleTask):
    """Take PDF maps generated by task :class:`PdfGenerator` 
    and use it to draw mock catalogs.

    Attributes
    ----------
    nqsos : int
        Number of quasars to draw in each mock catalog
    ncats : int
        Number of catalogs to generate 
    """

    nqsos = config.Property(proptype=int)
    ncats = config.Property(proptype=int)

    def setup(self, pdf_map):
        """
        """
        self.pdf = pdf_map

        # For easy access to communicator:
        self.comm_ = self.pdf.comm
        self.rank = self.comm_.Get_rank()

        # Easy access to local shapes and offsets
        self.lo = self.pdf.map[:, 0, :].local_offset[0]
        self.ls = self.pdf.map[:, 0, :].local_shape[0]
        self.lo_list = self.comm_.allgather(self.lo)
        self.ls_list = self.comm_.allgather(self.ls)

        # Global shape:
        n_z = self.pdf.map[:, 0, :].global_shape[0]

        # Wheight of each redshift bin in the pdf
        z_wheights = np.sum(self.pdf.map[:, 0, :], axis=1)

        if self.rank == 0:
            # Only rank zero is relevant. All the others are None.
            self.global_z_wheights = np.zeros(n_z)
        else:
            # All processes must have a value for self.global_z_wheights:
            self.global_z_wheights = None

        # Gather z_wheights on rank 0 (necessary to draw a redshift
        # distribution of quasars):
        self.comm_.Gatherv(
            z_wheights,
            [
                self.global_z_wheights,
                tuple(self.ls_list),
                tuple(self.lo_list),
                MPI.DOUBLE,
            ],
            root=0,
        )

        # CDF to draw quasars from:
        self.cdf = np.cumsum(self.pdf.map[:, 0, :], axis=1)
        # Normalize:
        self.cdf = self.cdf / self.cdf[:, -1][:, np.newaxis]

    def process(self):
        """
        """

        if self.rank == 0:
            # Only rank zero is relevant. All the others are None.
            # The number of quasars in each redshift bin follows a multinomial
            # distribution (reshape from (1,nz) to (nz) to make a 1D array):
            global_qso_numbers = np.random.multinomial(
                self.nqsos, self.global_z_wheights
            )
        else:
            # All processes must have a value for qso_numbers:
            global_qso_numbers = None

        qso_numbers = np.zeros(self.ls, dtype=np.int)
        # Need to pass tuples. For some reason lists don't work.
        # qso_numbers has shape (self.ls)
        self.comm_.Scatterv(
            [global_qso_numbers, tuple(self.ls_list), tuple(self.lo_list), MPI.DOUBLE],
            qso_numbers,
        )

        # Generate random numbers to assign voxels.
        # Shape: [self.ls=local # of z-bins][# of qsos in each z-bin]
        rnbs = [np.random.uniform(size=num) for num in qso_numbers]

        # Indices of each random quasar in pdf maps pixels.
        # Shape: [self.ls=local # of z-bins][# of qsos in each z-bin]
        idxs = [np.digitize(rnbs[ii], self.cdf[ii]) for ii in range(len(rnbs))]

        # Generate random nmbrs to randomize position of quasars in each voxel:
        # Random numbers for z-placement range: (-0.5,0.5)
        rz = [np.random.uniform(size=num) - 0.5 for num in qso_numbers]
        # Random numbers for theta-placement range: (-0.5,0.5)
        rtheta = [np.random.uniform(size=num) - 0.5 for num in qso_numbers]
        # Random numbers for phi-placement range: (-0.5,0.5)
        rphi = [np.random.uniform(size=num) - 0.5 for num in qso_numbers]

        # :meth::nside2resol() returns the square root of the pixel area,
        # which is a gross approximation of the pixel size, given the
        # different pixel shapes. I convert to degrees.
        ang_size = hp.pixelfunc.nside2resol(self._nside) * 180.0 / np.pi

        # Global values for redshift bins:
        z = _freq_to_z(self.pdf.freq)

        # Number of quasars in each rank
        nqso_rank = np.sum([len(idxs[ii]) for ii in range(len(idxs))])
        # Local arrays to hold the informations on
        # quasars in the local frequency range
        mock_zs = np.empty(nqso_rank, dtype=np.float64)
        mock_ra = np.empty(nqso_rank, dtype=np.float64)
        mock_dec = np.empty(nqso_rank, dtype=np.float64)
        qso_count = 0
        for ii in range(len(idxs)):  # For each local redshift bin
            for jj in range(len(idxs[ii])):  # For each quasar in in z-bin ii
                decbase, RAbase = _pix_to_radec(idxs[ii][jj], self._nside)
                # global redshift index:
                global_z_index = ii + self.lo
                # Randomly distributed in z bin range:
                z_value = (
                    z["width"][global_z_index] * rz[ii][jj]
                    + z["centre"][global_z_index]
                )
                # Populate local arrays
                mock_zs[qso_count] = z_value
                mock_ra[qso_count] = RAbase + ang_size * rtheta[ii][jj]
                mock_dec[qso_count] = decbase + ang_size * rphi[ii][jj]
                qso_count += 1

        # Arrays to hold the whole quasar set information
        mock_zs_full = np.empty(self.nqsos, dtype=mock_zs.dtype)
        mock_ra_full = np.empty(self.nqsos, dtype=mock_ra.dtype)
        mock_dec_full = np.empty(self.nqsos, dtype=mock_dec.dtype)

        # The counts and displacement arguments of Allgatherv are tuples!
        # Tuple (not list!) of number of QSOs in each rank
        nqso_tuple = tuple(self.comm_.allgather(nqso_rank))
        # Tuple (not list!) of displacements of each rank array in full array
        dspls = tuple(np.insert(arr=np.cumsum(nqso_tuple)[:-1], obj=0, values=0.0))
        # Gather redshifts
        recvbuf = [mock_zs_full, nqso_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_zs, len(mock_zs)]
        self.comm_.Allgatherv(sendbuf, recvbuf)
        # Gather theta
        recvbuf = [mock_ra_full, nqso_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_ra, len(mock_ra)]
        self.comm_.Allgatherv(sendbuf, recvbuf)
        # Gather phi
        recvbuf = [mock_dec_full, nqso_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_dec, len(mock_dec)]
        self.comm_.Allgatherv(sendbuf, recvbuf)

        # Create catalog container
        mock_catalog = containers.SpectroscopicCatalog(
            object_id=np.arange(self.nqsos, dtype=np.uint64)
        )
        mock_catalog["position"][:] = np.empty(
            self.nqsos, dtype=[("ra", mock_ra.dtype), ("dec", mock_dec.dtype)]
        )
        mock_catalog["redshift"][:] = np.empty(
            self.nqsos, dtype=[("z", mock_zs.dtype), ("z_error", mock_zs.dtype)]
        )
        # Assign data to catalog container
        mock_catalog["position"]["ra"][:] = mock_ra_full
        mock_catalog["position"]["dec"][:] = mock_dec_full
        mock_catalog["redshift"]["z"][:] = mock_zs_full
        # There is a provision for z-error. Zero here.
        mock_catalog["redshift"]["z_error"][:] = 0.0

        if self._count == self.ncats - 1:
            self.done = True

        return mock_catalog

    def process_finish(self):
        """
        """
        return None

    @property
    def pdf(self):
        return self._pdf

    @pdf.setter
    def pdf(self, pdf):
        """
        Setter for pdf
        Also set the attributes:
            self._npix : Number of pixels in PDF maps 
            self._nside : NSIDE of PDF maps

        """
        if isinstance(pdf, containers.Map):
            self._pdf = pdf
            self._npix = len(self._pdf.index_map["pixel"])
            self._nside = hp.pixelfunc.npix2nside(self._npix)
        else:
            msg = (
                "pdf is not an instance of "
                + "draco.core.containers.Map\n"
                + "Value for _pdf not set."
            )
            print(msg)


# Internal functions
# ------------------


def _zlims_to_freq(z, zlims):
    freqcentre = units.nu21 / (z + 1)
    freqlims = units.nu21 / (zlims + 1)
    freqwidth = abs(freqlims[:-1] - freqlims[1:])
    return np.array(
        [(freqcentre[ii], freqwidth[ii]) for ii in range(len(z))],
        dtype=[("centre", "<f8"), ("width", "<f8")],
    )


def _freq_to_z(freq):
    fc = freq["centre"]
    fw = freq["width"]
    z = units.nu21 / fc - 1.0

    sgn = np.sign(fc[-1] - fc[0])

    flims = fc - sgn * 0.5 * fw
    flims = np.append(flims, fc[-1] + sgn * 0.5 * fw[-1])
    zlims = units.nu21 / flims - 1.0
    z_width = abs(zlims[:-1] - zlims[1:])

    return np.array(
        [(z[ii], z_width[ii]) for ii in range(len(z))],
        dtype=[("centre", "<f8"), ("width", "<f8")],
    )


def _pix_to_radec(index, nside):
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    return -np.degrees(theta - np.pi / 2.0), np.degrees(phi)


def _radec_to_pix(ra, dec, nside):
    return hp.pixelfunc.ang2pix(nside, np.radians(-dec + 90.0), np.radians(ra))

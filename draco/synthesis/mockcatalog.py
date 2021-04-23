"""Take a source catalog and a (possibly biased) matter simulated map
and generate mock catalogs correlated to the matter maps and following
a selection function derived from the original catalog.

Pipeline tasks
==============

.. autosummary::
    :toctree:

    SelFuncEstimator
    PdfGenerator
    MockCatGenerator

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
probability distribution function to :class:`MockCatGenerator` to generate
mock catalogs. Below is an example of yaml file to generate mock catalogs:

>>> spam_config = '''
... pipeline :
...     tasks:
...         -   type:   draco.synthesis.mockcatalog.SelFuncEstimatorFromParams
...             params: selfunc_params
...             out:    selfunc
...
...         -   type:     draco.synthesis.mockcatalog.PdfGenerator
...             params:   pdf_params
...             requires: selfunc
...             out:      pdf_map
...
...         -   type:     draco.synthesis.mockcatalog.MockCatGenerator
...             params:   mqcat_params
...             requires: pdf_map
...             out:      mockcat
...
... selfunc_params:
...     bcat_path: '/bg01/homescinet/k/krs/jrs65/sdss_quasar_catalog.h5'
...     nside: 16
...
... pdf_params:
...     source_maps_path: '/scratch/k/krs/fandino/xcorrSDSS/sim21cm/21cmmap.hdf5'
...
... mqcat_params:
...     nsources: 200000
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
from cora.util import units, cosmology
from caput import config
from caput import mpiarray, mpiutil
from ..core import task, containers
from mpi4py import MPI


# Pipeline tasks
# --------------


# class SelFuncEstimator(SelFuncEstimatorFromParams):
#     """Estimate selection function from Catalog passed into the setup routine.
#     """
#
#     def setup(self, cat):
#         """Add the container to the internal namespace.
#
#         Parameters
#         ----------
#         cont : containers.SpectroscopicCatalog
#         """
#         self._base_qcat = cat


class SelFuncEstimatorFromParams(task.SingleTask):
    """Takes a source catalog as input and returns an estimate of the
    selection function based on a low rank SVD reconstruction.

    Attributes
    ----------
    bcat_path : str, optional
        Full path to base source catalog. If unspecified, a trivial selection
        function (all ones) is assumed. Default: None
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

    bcat_path = config.Property(proptype=str, default=None)

    # These seem to be optimal parameters and should not
    # usually need to be changed from the default values:
    nside = config.Property(proptype=int, default=16)
    n_z = config.Property(proptype=int, default=32)
    n_modes = config.Property(proptype=int, default=7)
    z_stt = config.Property(proptype=float, default=0.8)
    z_stp = config.Property(proptype=float, default=2.5)

    def setup(self):
        """Load container from file.
        """
        # Load base source catalog from file, if specified:
        self._base_qcat = None
        if self.bcat_path is not None:
            self._base_qcat = containers.SpectroscopicCatalog.from_file(self.bcat_path)

    def process(self):
        """Put the base catalog into maps. SVD the maps and recover
        with a small number of modes. This smoothes out the distribution
        sources and provides an estimate of the selection function used.
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

        # Map container to store the selection function:
        self._selfunc = containers.Map(
            nside=self.nside, polarisation=False, freq=freq_selfunc
        )

        # If no input catalog was specified, assume trivial selection function
        if self.base_qcat is None:
            self._selfunc["map"][:, :, :] = np.ones(self._selfunc["map"].local_shape)

        else:
            # Start as zeroes:
            self._selfunc["map"][:, :, :] = np.zeros(self._selfunc["map"].local_shape)

            # Create maps from original catalog:
            # No point in distributing this in a mpi clever way
            # because I need to SVD it.
            maps = np.zeros((self.n_z, n_pix))
            # Indices of each source in z axis:
            idxs = (
                np.digitize(self.base_qcat["redshift"]["z"], zlims_selfunc) - 1
            )  # -1 to get indices
            # Map pixel of each source
            pixls = _radec_to_pix(
                self.base_qcat["position"]["ra"],
                self.base_qcat["position"]["dec"],
                self.nside,
            )
            for jj in range(self.n_z):
                zpixls = pixls[idxs == jj]  # Map pixels of sources in z bin jj
                for kk in range(n_pix):
                    # Number of sources in z bin jj and pixel kk
                    maps[jj, kk] = np.sum(zpixls == kk)

            # SVD the source density maps:
            svd = np.linalg.svd(maps, full_matrices=0)

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


def _cat_to_maps(cat, nside, n_z, z_stt, z_stp):

    # Number of pixels to use in catalog maps for SVD
    n_pix = hp.pixelfunc.nside2npix(nside)

    # Redshift bins edges
    zlims_selfunc = np.linspace(z_stt, z_stp, n_z + 1)
    # Redshift bins centre
    z_selfunc = (zlims_selfunc[:-1] + zlims_selfunc[1:]) * 0.5

    # Have to transform the z axis to a freq axis to return selfunc in
    # containers.Map format:
    freq_selfunc = _zlims_to_freq(z_selfunc, zlims_selfunc)

    # Create maps from original catalog:
    # No point in distributing this in a mpi clever way
    # because I need to SVD it.
    maps = np.zeros((n_z, n_pix))
    # Indices of each source in z axis:
    idxs = (
        np.digitize(cat["redshift"]["z"], zlims_selfunc) - 1
    )  # -1 to get indices
    # Map pixel of each source
    pixls = _radec_to_pix(
        cat["position"]["ra"],
        cat["position"]["dec"],
        nside,
    )
    for jj in range(n_z):
        zpixls = pixls[idxs == jj]  # Map pixels of sources in z bin jj
        for kk in range(n_pix):
            # Number of sources in z bin jj and pixel kk
            maps[jj, kk] = np.sum(zpixls == kk)

    return maps


class SelFuncEstimator(SelFuncEstimatorFromParams):
    """Estimate selection function from Catalog passed into the setup routine.
    """

    def setup(self, cat):
        """Add the container to the internal namespace.

        Parameters
        ----------
        cont : containers.SpectroscopicCatalog
        """
        self._base_qcat = cat


class PdfGenerator(task.SingleTask):
    """Take a source catalog selection function and simulated source
    (biased density) maps and return a PDF map correlated with the
    density maps and the selection function. This PDF map can be used
    by the task :class:`MockCatGenerator` to draw mock catalogs.

    Attributes
    ----------
    source_maps_path : str
        Full path to simulated source maps (biased matter density fluctuations).
    random_catalog : bool
        Is True generate random catalogs, not correlated with the maps.
        Default is False.

    """

    source_maps_path = config.Property(proptype=str)
    random_catalog = config.Property(proptype=bool, default=False)
    no_selfunc = config.Property(proptype=bool, default=False)
    use_voxel_volumes = config.Property(proptype=bool, default=False)

    def setup(self, selfunc):
        """
        """
        self.selfunc = selfunc

        # Load source maps from file:
        source_maps = containers.Map.from_file(self.source_maps_path, distributed=True)
        if self.random_catalog:
            # To make a random (not correlated) catalog
            source_maps.map[:] = np.zeros_like(source_maps.map)

        self.source_maps = source_maps  # Setter sets other parameters too

        # For easy access to communicator:
        self.comm_ = self.source_maps.comm
        self.rank = self.comm_.Get_rank()  # Unused for now

    def process(self):
        """
        """
        # From frequency to redshift:
        z = _freq_to_z(self.source_maps.index_map["freq"])
        n_z = len(z)

        # Freq to redshift of selection function:
        z_selfunc = _freq_to_z(self.selfunc.index_map["freq"])

        # Re-distribute maps in pixels:
        self.source_maps.redistribute(dist_axis=2)

        # TODO: Change h1maps for something more generic, like density_maps

        rho_m = mpiarray.MPIArray.wrap(self.source_maps.map[:, 0, :] + 1.0, axis=1)

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
        resized_selfunc = _resize_map(
            self.selfunc.map[:, 0, :], rho_m.global_shape, z, z_selfunc
        )
        # Generate wheights for correct distribution of sources in redshift:
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
        if self.no_selfunc:
            self.log.debug("Using trivial selection function to generate PDF!")
            pdf = rho_m
            if not self.use_voxel_volumes:
                # Set each frequency channel to have equal total probability
                z_wheights = 1/n_z * np.ones_like(z_wheights)
            else:
                # Set total probability for each frequency channel based
                # on voxel volume for that channel.
                # Healpix maps have equal-angular-area pixels, so the voxel
                # area is proportional to \chi^2 * (\chi_max - \chi_min),
                # where we use \chi_centre for the first \chi (which incorporates
                # the z-dependence of transverse area), and the second factor
                # is the voxel size along the z direction.

                cosmo = cosmology.Cosmology()
                z_wheights_global = np.zeros(
                    len(self.source_maps.index_map["freq"]),
                    dtype=np.float64
                )

                # First, we compute the normalization for each channel
                # globally
                for fi, freq in enumerate(self.source_maps.index_map["freq"]):
                    z_min = units.nu21 / (freq[0] + 0.5 * freq[1]) - 1
                    z_max = units.nu21 / (freq[0] - 0.5 * freq[1]) - 1
                    z_mean = units.nu21 / freq[0] - 1

                    z_wheights_global[fi] = (
                        cosmo.comoving_distance(z_mean)**2
                        * (
                            cosmo.comoving_distance(z_max)
                            - cosmo.comoving_distance(z_min)
                        )
                    )

                z_wheights_global /= z_wheights_global.sum()

                # Select local section of weights
                z_wheights = z_wheights_global[
                    z_wheights.local_offset[0] :
                    z_wheights.local_offset[0] + z_wheights.local_shape[0]
                ]
                print('Rank %d: z_wheights are' % mpiutil.rank, z_wheights)

        else:
            pdf = rho_m * resized_selfunc

        # Enforce redshift distribution to follow selection function:
        pdf = mpiarray.MPIArray.wrap(
            pdf / np.sum(pdf, axis=1)[:, np.newaxis] * z_wheights[:, np.newaxis], axis=0
        )

        # Put PDF in a map container:
        pdf_map = containers.Map(
            nside=self._nside, polarisation=False, freq=self.source_maps.index_map["freq"]
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

    @property
    def source_maps(self):
        return self._source_maps

    @source_maps.setter
    def source_maps(self, source_maps):
        """
        Setter for source_maps
        Also set the attributes:
            self._npix : Number of pixels in source maps
            self._nside : NSIDE of source maps

        """
        if isinstance(source_maps, containers.Map):
            self._source_maps = source_maps
            self._npix = len(self._source_maps.index_map["pixel"])
            self._nside = hp.pixelfunc.npix2nside(self._npix)
        else:
            msg = (
                "source_maps is not an instance of "
                + "draco.core.containers.Map\n"
                + "Value for _source_maps not set."
            )
            print(msg)




def _resize_map(map, new_shape, z_new, z_old):
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



class MockCatGenerator(task.SingleTask):
    """Take PDF maps generated by task :class:`PdfGenerator`
    and use it to draw mock catalogs.

    Attributes
    ----------
    nsources : int
        Number of sources to draw in each mock catalog
    ncats : int
        Number of catalogs to generate
    """

    nsources = config.Property(proptype=int)
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
        # distribution of sources):
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

        # CDF to draw sources from:
        self.cdf = np.cumsum(self.pdf.map[:, 0, :], axis=1)
        # Normalize:
        self.cdf = self.cdf / self.cdf[:, -1][:, np.newaxis]

    def process(self):
        """
        """

        if self.rank == 0:
            # Only rank zero is relevant. All the others are None.
            # The number of sources in each redshift bin follows a multinomial
            # distribution (reshape from (1,nz) to (nz) to make a 1D array):
            global_source_numbers = np.random.multinomial(
                self.nsources, self.global_z_wheights
            )
        else:
            # All processes must have a value for source_numbers:
            global_source_numbers = None

        source_numbers = np.zeros(self.ls, dtype=np.int)
        # Need to pass tuples. For some reason lists don't work.
        # source_numbers has shape (self.ls)
        self.comm_.Scatterv(
            [global_source_numbers, tuple(self.ls_list), tuple(self.lo_list), MPI.DOUBLE],
            source_numbers,
        )

        # Generate random numbers to assign voxels.
        # Shape: [self.ls=local # of z-bins][# of sources in each z-bin]
        rnbs = [np.random.uniform(size=num) for num in source_numbers]

        # Indices of each random source in pdf maps pixels.
        # Shape: [self.ls=local # of z-bins][# of sources in each z-bin]
        idxs = [np.digitize(rnbs[ii], self.cdf[ii]) for ii in range(len(rnbs))]

        # Generate random nmbrs to randomize position of sources in each voxel:
        # Random numbers for z-placement range: (-0.5,0.5)
        rz = [np.random.uniform(size=num) - 0.5 for num in source_numbers]
        # Random numbers for theta-placement range: (-0.5,0.5)
        rtheta = [np.random.uniform(size=num) - 0.5 for num in source_numbers]
        # Random numbers for phi-placement range: (-0.5,0.5)
        rphi = [np.random.uniform(size=num) - 0.5 for num in source_numbers]

        # :meth::nside2resol() returns the square root of the pixel area,
        # which is a gross approximation of the pixel size, given the
        # different pixel shapes. I convert to degrees.
        ang_size = hp.pixelfunc.nside2resol(self._nside) * 180.0 / np.pi

        # Global values for redshift bins:
        z = _freq_to_z(self.pdf.index_map["freq"][:])

        # Number of sources in each rank
        nsource_rank = np.sum([len(idxs[ii]) for ii in range(len(idxs))])
        # Local arrays to hold the informations on
        # sources in the local frequency range
        mock_zs = np.empty(nsource_rank, dtype=np.float64)
        mock_ra = np.empty(nsource_rank, dtype=np.float64)
        mock_dec = np.empty(nsource_rank, dtype=np.float64)
        source_count = 0
        for ii in range(len(idxs)):  # For each local redshift bin
            for jj in range(len(idxs[ii])):  # For each source in in z-bin ii
                decbase, RAbase = _pix_to_radec(idxs[ii][jj], self._nside)
                # global redshift index:
                global_z_index = ii + self.lo
                # Randomly distributed in z bin range:
                z_value = (
                    z["width"][global_z_index] * rz[ii][jj]
                    + z["centre"][global_z_index]
                )
                # Populate local arrays
                mock_zs[source_count] = z_value
                mock_ra[source_count] = RAbase + ang_size * rtheta[ii][jj]
                mock_dec[source_count] = decbase + ang_size * rphi[ii][jj]
                source_count += 1

        # Arrays to hold the whole source set information
        mock_zs_full = np.empty(self.nsources, dtype=mock_zs.dtype)
        mock_ra_full = np.empty(self.nsources, dtype=mock_ra.dtype)
        mock_dec_full = np.empty(self.nsources, dtype=mock_dec.dtype)

        # The counts and displacement arguments of Allgatherv are tuples!
        # Tuple (not list!) of number of sources in each rank
        nsource_tuple = tuple(self.comm_.allgather(nsource_rank))
        # Tuple (not list!) of displacements of each rank array in full array
        dspls = tuple(np.insert(arr=np.cumsum(nsource_tuple)[:-1], obj=0, values=0.0))
        # Gather redshifts
        recvbuf = [mock_zs_full, nsource_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_zs, len(mock_zs)]
        self.comm_.Allgatherv(sendbuf, recvbuf)
        # Gather theta
        recvbuf = [mock_ra_full, nsource_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_ra, len(mock_ra)]
        self.comm_.Allgatherv(sendbuf, recvbuf)
        # Gather phi
        recvbuf = [mock_dec_full, nsource_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_dec, len(mock_dec)]
        self.comm_.Allgatherv(sendbuf, recvbuf)

        # Create catalog container
        mock_catalog = containers.SpectroscopicCatalog(
            object_id=np.arange(self.nsources, dtype=np.uint64)
        )
        mock_catalog["position"][:] = np.empty(
            self.nsources, dtype=[("ra", mock_ra.dtype), ("dec", mock_dec.dtype)]
        )
        mock_catalog["redshift"][:] = np.empty(
            self.nsources, dtype=[("z", mock_zs.dtype), ("z_error", mock_zs.dtype)]
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



class MapPixLocGenerator(task.SingleTask):
    """Generate a list of sky positions corresponding to the pixel
    centers of an input Healpix map.

    Attributes
    ----------
    freq_idx : int
        Index of frequency channel to assign to all "sources".
    """

    freq_idx = config.Property(proptype=int)

    def setup(self, in_map):
        """
        """
        self.map = in_map

        # For easy access to communicator:
        self.comm_ = self.map.comm
        self.rank = self.comm_.Get_rank()

        # # Easy access to local shapes and offsets
        # self.lo = self.map[:, 0, :].local_offset[0]
        # self.ls = self.map[:, 0, :].local_shape[0]
        # self.lo_list = self.comm_.allgather(self.lo)
        # self.ls_list = self.comm_.allgather(self.ls)

        # Global shape of frequency axis
        n_z = self.map.map[:, 0, :].global_shape[0]

        if not isinstance(self.map, containers.Map):
            raise RuntimeError("Input map is not actually a map!")
        else:
            self.npix = len(self.map.index_map["pixel"])
            self.nside = hp.pixelfunc.npix2nside(self.npix)

        self.z_arr = _freq_to_z(self.map.index_map["freq"])
        self.z = self.z_arr[self.freq_idx]["centre"]


    def process(self):
        """
        """

        local_pix_numbers = mpiutil.partition_list_mpi(np.arange(self.npix))
        npix_rank = len(local_pix_numbers)

        pix_dec, pix_ra = _pix_to_radec(local_pix_numbers, self.nside)

        # Arrays to hold the whole source set information
        ra_full = np.empty(self.npix, dtype=pix_ra.dtype)
        dec_full = np.empty(self.npix, dtype=pix_dec.dtype)

        # The counts and displacement arguments of Allgatherv are tuples!
        # Tuple (not list!) of number of pixels in each rank
        npix_tuple = tuple(self.comm_.allgather(npix_rank))
        # Tuple (not list!) of displacements of each rank array in full array
        dspls = tuple(np.insert(arr=np.cumsum(npix_tuple)[:-1], obj=0, values=0.0))
        # Gather theta
        recvbuf = [ra_full, npix_tuple, dspls, MPI.DOUBLE]
        sendbuf = [pix_ra, len(pix_ra)]
        self.comm_.Allgatherv(sendbuf, recvbuf)
        # Gather phi
        recvbuf = [dec_full, npix_tuple, dspls, MPI.DOUBLE]
        sendbuf = [pix_dec, len(pix_dec)]
        self.comm_.Allgatherv(sendbuf, recvbuf)

        # Create catalog container
        mock_catalog = containers.SpectroscopicCatalog(
            object_id=np.arange(self.npix, dtype=np.uint64)
        )
        mock_catalog["position"][:] = np.empty(
            self.npix, dtype=[("ra", pix_ra.dtype), ("dec", pix_dec.dtype)]
        )
        mock_catalog["redshift"][:] = np.empty(
            self.npix, dtype=[("z", pix_ra.dtype), ("z_error", pix_ra.dtype)]
        )
        # Assign data to catalog container
        mock_catalog["position"]["ra"][:] = ra_full
        mock_catalog["position"]["dec"][:] = dec_full
        mock_catalog["redshift"]["z"][:] = (
            self.z * np.ones(self.npix, dtype=pix_ra.dtype)
        )
        # There is a provision for z-error. Zero here.
        mock_catalog["redshift"]["z_error"][:] = 0.0

        self.done = True
        return mock_catalog

    def process_finish(self):
        """
        """
        return None


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

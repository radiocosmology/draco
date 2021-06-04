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
from cora.util import units
from caput import config
from caput import mpiarray, mpiutil
from ..core import task, containers
from mpi4py import MPI


# Pipeline tasks
# --------------

class SelFuncEstimator(task.SingleTask):
    """Takes a source catalog as input and returns an estimate of the
    selection function based on a low rank SVD reconstruction.

    The defaults for nside, n_z, and n_modes have been empirically determined
    to produce reasonable results for the selection function when z_min = 0.8,
    z_max = 2.5.

    Redshifts are binned into n_z equispaced bins with min and max
    bin edges set by z_min and z_max.

    Attributes
    ----------
    nside : int
        Healpix Nside for catalog maps generated for the SVD.
        Default: 16.
    n_z : int
        Number of redshift bins for catalog maps generated for the SVD.
        Default: 32.
    z_min : float
        Lower edge of minimum redshift bin for catalog maps generated for the SVD.
        Default: 0.8.
    z_max : float
        Upper edge of maximum redshift for catalog maps generated for the SVD.
        Default: 2.5.
    n_modes : int
        Number of SVD modes used in recovering the selection function from
        the catalog maps.
        Default: 7.
    """

    bcat_path = config.Property(proptype=str, default=None)

    # These seem to be optimal parameters for eBOSS quasars, and
    # usually should not need to be changed from the default values:
    nside = config.Property(proptype=int, default=16)
    n_z = config.Property(proptype=int, default=32)
    z_min = config.Property(proptype=float, default=0.8)
    z_max = config.Property(proptype=float, default=2.5)
    n_modes = config.Property(proptype=int, default=7)

    def process(self, cat):
        """Estimate selection function from SVD of catalog map.

        After binning the positions in the catalog into redshift bins
        and healpix pixels, we SVD the n_z x n_pixel map and reconstruct
        the catalog with a small number of modes. Doing this at low angular
        resolution smoothes out the distribution of sources and provides
        and estimate of the selection function.

        Parameters
        ----------
        data : :class:`containers.SpectroscopicCatalog`
            Input catalog.

        Returns
        -------
        selfunc : :class:`containers.Map`
            The visibility dataset with new weights.

        """

        # Compute redshift bin edges and centers
        zlims_selfunc = np.linspace(self.z_min, self.z_max, self.n_z + 1)
        z_selfunc = (zlims_selfunc[:-1] + zlims_selfunc[1:]) * 0.5

        # Transform redshift bin edges to frequency bin edges
        freq_selfunc = _zlims_to_freq(z_selfunc, zlims_selfunc)

        # Create Map container to store the selection function
        selfunc = containers.Map(
            nside=self.nside, polarisation=False, freq=freq_selfunc
        )

        # Initialize selection function to zero
        selfunc["map"][:] = 0 #np.zeros(selfunc["map"].local_shape)

        # Create maps from original catalog (on each MPI rank separately)
        maps = _cat_to_maps(cat, self.nside, zlims_selfunc)

        # SVD the n_z x n_pixel map of source counts
        svd = np.linalg.svd(maps, full_matrices=0)

        # Get axis parameters for distributed map:
        lo = selfunc["map"][:, 0, :].local_offset[0]
        ls = selfunc["map"][:, 0, :].local_shape[0]

        # Accumulate the modes we wish to keep in the Map container
        for mode_i in range(self.n_modes):
            uj = svd[0][:, mode_i]
            sj = svd[1][mode_i]
            vj = svd[2][mode_i, :]

            # Wrap reconstructed selfunc mode into MPIArray, so that
            # we can add to distributed map dataset
            recmode = mpiarray.MPIArray.wrap(
                (uj[:, None] * sj * vj[None, :])[lo : lo + ls], axis=0
            )
            selfunc["map"][:, 0, :] += recmode

        # Remove negative entries remaining from SVD recovery:
        selfunc["map"][np.where(selfunc.map[:] < 0.0)] = 0.0

        return selfunc



class ResizeSelFuncMap(task.SingleTask):
    """Take a selection function map and simulated source
    (biased density) map and return a selection function map with the
    same resolution and frequency sampling as the source map.
    """

    def process(self, selfunc, source_map):
        """Resize selection function map.

        Parameters
        ----------
        selfunc : :class:`containers.Map`
            Input selection function.
        source_map : :class:`containers.Map`
            Map whose frequency and angular redshift resolution the
            output selection function map will be matched to. This will
            typically be the same map passed to the `PdfGenerator` task.

        Returns
        -------
        new_selfunc : class:`containers.Map`
            Resized selection function.
        """
        from ..util import regrid

        # Convert frequency axes to redshifts
        z_selfunc = _freq_to_z(selfunc.index_map["freq"])
        z_source = _freq_to_z(source_map.index_map["freq"])
        n_z_source = len(z_source)

        # Make container for resized selection function map
        new_selfunc = containers.Map(
            polarisation=False,
            axes_from=source_map,
            attrs_from=source_map
        )

        # Form matrix to interpolate frequency/z axis
        interp_m = regrid.lanczos_forward_matrix(
            z_selfunc["centre"], z_source["centre"]
        )
        # Correct for redshift bin widths:
        interp_m *= (
            z_source["width"][:, np.newaxis] / z_selfunc["width"][np.newaxis, :]
        )

        # Redistribute selfunc along pixel axis, so we can resize
        # the frequency axis
        selfunc.redistribute("pixel")

        # Interpolate input selection function onto new redshift bins,
        # and wrap in MPIArray distributed along pixel axis
        selfunc_map_newz = mpiarray.MPIArray.wrap(
            np.dot(interp_m, selfunc.map[:, 0, :]), axis=1
        )

        # Redistribute along frequency axis
        selfunc_map_newz = selfunc_map_newz.redistribute(axis=0)

        # Determine desired output healpix Nside parameter
        nside = hp.npix2nside(len(new_selfunc.index_map["pixel"]))

        # Get local section of container for output selection function
        new_selfunc_map_local = new_selfunc.map[:]

        # For each frequency in local section, up/downgrade healpix maps
        # of selection function to desired resolution, and set negative
        # pixel values (which the Lanczos interpolation can create) to zero
        for fi in range(selfunc_map_newz.local_shape[0]):
            new_selfunc_map_local[:][fi, 0] = hp.ud_grade(selfunc_map_newz[fi], nside)
            new_selfunc_map_local[:][fi, 0][new_selfunc_map_local[:][fi, 0][:] < 0] = 0

        return new_selfunc


class PdfGeneratorBase(task.SingleTask):
    """Base class for PDF generator (non-functional).

    Take a source catalog selection function and simulated source
    (biased density) map and return a PDF map constructed from the
    product of the two, appropriately normalized. This PDF map can be used
    by the task :class:`MockCatGenerator` to draw mock catalogs.

    Derived classes must implement process().
    """

    def make_pdf_map(self, source_map, z_weights, selfunc=None):
        """Make PDF map from source map, redshift weights, and selection function.

        Parameters
        ----------
        source_map : :class:`containers.Map`
            Overdensity map to base PDF on.
        z_weights : `MPIArray`
            Relative weight of each redshift/frequency bin in PDF.
        selfunc : :class:`containers.Map`, optional
            Selection function for objects drawn from PDF. If not specified,
            a uniform selection function is assumed.

        Returns
        -------
        pdf_map : :class:`containers.Map`
            Output PDF map.
        """

        # Assuming source map is overdensity, add 1 to form rho/rho_mean
        rho = mpiarray.MPIArray.wrap(source_map.map[:, 0, :] + 1.0, axis=0)

        # Normalize density to have unit mean in each z-bin:
        rho = mpiarray.MPIArray.wrap(
            rho / np.mean(rho, axis=1)[:, np.newaxis], axis=0
        )

        if selfunc is not None:
            # Get local section of selection function
            selfunc_local = selfunc.map[:, 0, :]

            # Multiply selection function into density
            pdf = mpiarray.MPIArray.wrap(rho * selfunc_local, axis=0)

        else:
            pdf = mpiarray.MPIArray.wrap(rho, axis=0)

        # Normalize by redshift weights
        pdf = mpiarray.MPIArray.wrap(
            pdf / np.sum(pdf, axis=1)[:, np.newaxis] * z_weights[:, np.newaxis], axis=0
        )

        # Make container for PDF
        nside = hp.npix2nside(len(source_map.index_map["pixel"]))
        pdf_map = containers.Map(
            nside=nside, polarisation=False, freq=source_map.index_map["freq"]
        )

        # Put computed PDF into local section of container
        pdf_map_local = pdf_map.map[:]
        pdf_map_local[:, 0, :] = pdf

        return pdf_map


    def process(self):
        raise NotImplementedError(
            f"{self.__class__} must define a process method."
        )


class PdfGeneratorUncorrelated(PdfGeneratorBase):
    """Generate uniform PDF for making uncorrelated mocks.
    """

    def process(self, source_map):
        """Make PDF map with uniform z weights and delta_g=0.

        Parameters
        ----------
        source_map : :class:`containers.Map`
            Overdensity map that determines z and angular resolution
            of output PDF map.

        Returns
        -------
        pdf_map : :class:`containers.Map`
            Output PDF map.
        """

        # Get local section of source map, and set to zero
        source_map_local = source_map.map[:, 0, :]
        source_map_local[:] = 0

        # Get local and global shape of frequency axis
        ls = source_map.map.local_shape[0]
        gs = source_map.map.global_shape[0]

        # Set each frequency channel to have equal total probability
        z_weights = mpiarray.MPIArray.wrap(
            1 / gs * np.ones(ls), axis=0
        )

        # Create PDF map
        pdf_map = self.make_pdf_map(source_map, z_weights)

        return pdf_map


class PdfGeneratorWithSelfunc(PdfGeneratorBase):
    """Generate PDF that incorporates a selection function.
    """

    def process(self, source_map, selfunc):
        """Make PDF map that incorporates the selection function.

        Parameters
        ----------
        source_map : :class:`containers.Map`
            Overdensity map that determines z and angular resolution
            of output PDF map.
        selfunc : :class:`containers.Map`
            Selection function map. Must have same z and angular resolution
            as source_map. Typically taken from `ResizeSelFuncMap`.

        Returns
        -------
        pdf_map : :class:`containers.Map`
            Output PDF map.
        """

        # Get MPI comm
        comm_ = source_map.comm

        # Get local section of selection function
        selfunc_local = selfunc.map[:, 0, :]

        # Generate weights for distribution of sources in redshift:
        # first, sum over selfunc pixel values at each z (z_weights),
        # then sum these over all z per rank (z_weights_local_sum)
        # and combine into sum across all ranks (z_weights_sum).
        # TODO: there must be a cleaner way to get z_weights_sum
        # that uses built-in MPIArray functionality...
        z_weights = np.sum(selfunc_local, axis=1)
        z_weights_local_sum = mpiarray.MPIArray.wrap(
            np.array([np.sum(z_weights, axis=0)]), axis=0
        )
        z_weights_sum = np.zeros_like(z_weights_local_sum)
        comm_.Allreduce(z_weights_local_sum, z_weights_sum)

        # Normalize z_weights by grand total
        z_weights = mpiarray.MPIArray.wrap(z_weights / z_weights_sum, axis=0)

        # Create PDF map
        pdf_map = self.make_pdf_map(source_map, z_weights, selfunc)

        return pdf_map


class PdfGeneratorNoSelfunc(PdfGeneratorBase):
    """Generate PDF that assumes a trivial selection function.

    Attributes
    ----------
    use_voxel_volumes : bool
        If true, set redshift weights based on relative comoving volumes
        of voxels corresponding to each frequency channel. Default: False.
    """

    use_voxel_volumes = config.Property(proptype=bool, default=False)

    def process(self, source_map):
        """Make PDF map that assumes a trivial selection function.

        Parameters
        ----------
        source_map : :class:`containers.Map`
            Overdensity map that determines z and angular resolution
            of output PDF map.

        Returns
        -------
        pdf_map : :class:`containers.Map`
            Output PDF map.
        """

        # Get local offset and shape of frequency axis, and global shape
        lo = source_map.map.local_offset[0]
        ls = source_map.map.local_shape[0]
        gs = source_map.map.global_shape[0]

        if not self.use_voxel_volumes:
            # Set each frequency channel to have equal total probability
            z_weights = mpiarray.MPIArray.wrap(
                1 / gs * np.ones(ls), axis=0
            )

        else:
            # Set total probability for each frequency channel based
            # on voxel volume for that channel.
            # Healpix maps have equal-angular-area pixels, so the voxel
            # area is proportional to \chi^2 * (\chi_max - \chi_min),
            # where we use \chi_centre for the first \chi (which incorporates
            # the z-dependence of transverse area), and the second factor
            # is the voxel size along the z direction.
            from cora.util import cosmology

            cosmo = cosmology.Cosmology()
            z_weights_global = np.zeros(gs, dtype=np.float64)

            # First, we compute the normalization for each channel
            # globally
            for fi, freq in enumerate(source_map.index_map["freq"]):
                z_min = units.nu21 / (freq[0] + 0.5 * freq[1]) - 1
                z_max = units.nu21 / (freq[0] - 0.5 * freq[1]) - 1
                z_mean = units.nu21 / freq[0] - 1

                z_weights_global[fi] = (
                    cosmo.comoving_distance(z_mean)**2
                    * (
                        cosmo.comoving_distance(z_max)
                        - cosmo.comoving_distance(z_min)
                    )
                )

            z_weights_global /= z_weights_global.sum()

            # Select local section of weights
            z_weights = mpiarray.MPIArray.wrap(
                z_weights_global[lo : lo + ls], axis=0
            )

        # Create PDF map
        pdf_map = self.make_pdf_map(source_map, z_weights)

        return pdf_map


class MockCatGenerator(task.SingleTask):
    """Take PDF maps generated by task :class:`PdfGenerator`
    and use it to draw mock catalogs.

    Attributes
    ----------
    nsources : int
        Number of sources to draw in each mock catalog
    ncats : int
        Number of catalogs to generate
    sigma_z : float, optional
        Standard deviation of Gaussian redshift errors (default: None)
    sigma_z_over_1plusz : float, optional
        Standard deviation of Gaussian redshift errors will be set to
        this parameter times (1+z). Only one of this and `sigma_z` can
        be specified. Default: None
    z_at_channel_centers : bool, optional
        Place each object at a redshift corresponding to the center of
        its frequency channel (True), or randomly distribute each object's
        redshift within its channel (False): Default: False
    """

    nsources = config.Property(proptype=int)
    ncats = config.Property(proptype=int)

    sigma_z = config.Property(proptype=float, default=None)
    sigma_z_over_1plusz = config.Property(proptype=float, default=None)

    z_at_channel_centers = config.Property(proptype=bool, default=False)

    def setup(self, pdf_map):
        """
        """
        self.pdf = pdf_map

        # For easy access to communicator:
        self.comm_ = self.pdf.comm
        self.rank = self.comm_.Get_rank()

        # Check that only one z error spec has been specified
        if (self.sigma_z is not None) and (self.sigma_z_over_1plusz is not None):
            raise config.CaputConfigError(
                "Only one of sigma_z and sigma_z_over_1plusz can be specified!"
            )

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

        # Generate random numbers to randomize position of sources in each voxel:
        # Random numbers for z-placement range: (-0.5,0.5)
        if not self.z_at_channel_centers:
            rz = [np.random.uniform(size=num) - 0.5 for num in source_numbers]
        # Generate random numbers for z errors, as standard normals
        # to be multiplied by appropriate standard deviation later
        if (self.sigma_z is not None) or (self.sigma_z_over_1plusz is not None):
            rzerr = [np.random.normal(size=num) for num in source_numbers]
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
        mock_zerrs = np.empty(nsource_rank, dtype=np.float64)
        mock_ra = np.empty(nsource_rank, dtype=np.float64)
        mock_dec = np.empty(nsource_rank, dtype=np.float64)
        source_count = 0
        for ii in range(len(idxs)):  # For each local redshift bin
            for jj in range(len(idxs[ii])):  # For each source in in z-bin ii
                decbase, RAbase = _pix_to_radec(idxs[ii][jj], self._nside)
                # global redshift index:
                global_z_index = ii + self.lo
                # Randomly distributed in z bin range:
                z_value = z["centre"][global_z_index]
                if not self.z_at_channel_centers:
                    z_value += z["width"][global_z_index] * rz[ii][jj]
                # If desired, add Gaussian z errors:
                if self.sigma_z is not None:
                    z_value += rzerr[ii][jj] * self.sigma_z
                    mock_zerrs[source_count] = rzerr[ii][jj] * self.sigma_z
                elif self.sigma_z_over_1plusz is not None:
                    z_value += (
                        rzerr[ii][jj] * self.sigma_z_over_1plusz * (1+z_value)
                    )
                    mock_zerrs[source_count] = (
                        rzerr[ii][jj] * self.sigma_z_over_1plusz * (1+z_value)
                    )
                else:
                    mock_zerrs[source_count] = 0
                # Populate local arrays
                mock_zs[source_count] = z_value
                mock_ra[source_count] = RAbase + ang_size * rtheta[ii][jj]
                mock_dec[source_count] = decbase + ang_size * rphi[ii][jj]
                source_count += 1

        # Arrays to hold the whole source set information
        mock_zs_full = np.empty(self.nsources, dtype=mock_zs.dtype)
        mock_zerrs_full = np.empty(self.nsources, dtype=mock_zs.dtype)
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
        # Gather redshift errors
        recvbuf = [mock_zerrs_full, nsource_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_zerrs, len(mock_zs)]
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
        # There is a provision for z-error
        mock_catalog["redshift"]["z_error"][:] = mock_zerrs_full

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

def _cat_to_maps(cat, nside, zlims_selfunc):
    """Grid a catalog of sky and z positions onto healpix maps.

    Parameters
    ----------
    cat : containers.SpectroscopicCatalog
        Input catalog.
    nside : int
        Healpix Nside parameter for output maps.
    zlims_selfunc : np.ndarray
        Edges of target redshift bins.

    Returns
    -------
    maps : np.ndarray
        Output healpix maps, packed as [n_z, n_pix].
    """

    # Number of pixels to use in catalog maps for SVD
    n_pix = hp.nside2npix(nside)

    # Number of redshift bins
    n_z = len(zlims_selfunc) - 1

    # Create maps from original catalog (on each MPI rank separately)
    maps = np.zeros((n_z, n_pix))
    # Compute indices of each source along z axis
    idxs = (
        np.digitize(cat["redshift"]["z"], zlims_selfunc) - 1
    )  # -1 to get indices
    # Map pixel of each source
    pixels = _radec_to_pix(
        cat["position"]["ra"], cat["position"]["dec"], nside
    )

    for zi in range(n_z):
        # Get map pixels containing sources in redshift bin zi
        zpixels = pixels[idxs == zi]
        # For each pixel in map, set pixel value to number of sources
        # within that pixel
        for pi in range(n_pix):
            maps[zi, pi] = np.sum(zpixels == pi)

    return maps

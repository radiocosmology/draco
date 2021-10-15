"""Tasks for making mock catalogs.

See Usage section for usage.

Pipeline tasks
==============

.. autosummary::
    :toctree:

    SelectionFunctionEstimator
    ResizeSelectionFunctionMap
    PdfGeneratorBase
    PdfGeneratorUncorrelated
    PdfGeneratorNoSelectionFunction
    PdfGeneratorWithSelectionFunction
    MockCatalogGenerator
    AddGaussianZErrorsToCatalog
    AddEBOSSZErrorsToCatalog
    MapPixelLocationGenerator

Usage
=====

Generally you would want to use these tasks together. A catalog is fed to
:class:`SelectionFunctionEstimator`, which generates a selection function map from a
low-rank SVD approximation to the positions in the catalog.
:class:`ResizeSelectionFunctionMap` resizes this to match the resolution of a simulated
map of galaxy overdensity delta_g. The resized selection function and delta_g
map are then fed to :class:`PdfGeneratorWithSelectionFunction`, which makes a PDF map
from which simulated sources are drawn in :class:`MockCatalogGenerator`. The PDF can also
be generated without a selection function, or assuming a uniform distribution
of sources.

:class:`MapPixelLocationGenerator` is a specialized task that creates a catalog whose
"sources" are located at Healpix pixel centers for a given angular resolution.

Below is an example workflow:

>>> mock_config = '''
... pipeline :
...     tasks:
...         - type: draco.core.io.LoadFilesFromParams
...           out: cat_for_selfunc
...           params:
...               files:
...                   - "/path/to/data/catalog.h5"
...
...         - type:   draco.synthesis.mockcatalog.SelectionFunctionEstimator
...           in: cat_for_selfunc
...           out: selfunc
...           params:
...               save: False
...
...         - type: draco.core.io.LoadMaps
...           out: source_map
...           params:
...               maps:
...                   files:
...                       - "/path/to/delta_g/map.h5"
...
...          - type: draco.synthesis.mockcatalog.ResizeSelectionFunctionMap
...            in: [selfunc, source_map]
...            out: resized_selfunc
...            params:
...                smooth: True
...                save: True
...                output_name: /path/to/saved/resized_selfunc.h5
...
...          - type: draco.synthesis.mockcatalog.PdfGeneratorWithSelectionFunction
...            in: [source_map, resized_selfunc]
...            out: pdf_map
...            params:
...                save: False
...
...          - type: draco.synthesis.mockcatalog.MockCatalogGenerator
...            requires: pdf_map
...            out: mock_cat
...            params:
...                nsource: 100000
...                ncat: 1
...                save: True
...                output_root: mock_
...
"""

# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
import healpy as hp
import scipy.stats

from cora.signal import corr21cm
from cora.util import units
from caput import config
from caput import mpiarray, mpiutil
from ..core import task, containers
from ..util import random, tools
from mpi4py import MPI

# Constants
C = units.c


# Pipeline tasks
# --------------


class SelectionFunctionEstimator(task.SingleTask):
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
    tracer : str, optional
        Set an optional tracer attribute that can be used to identify the type of
        catalog later in the pipeline.
    """

    bcat_path = config.Property(proptype=str, default=None)

    # These seem to be optimal parameters for eBOSS quasars, and
    # usually should not need to be changed from the default values:
    nside = config.Property(proptype=int, default=16)
    n_z = config.Property(proptype=int, default=32)
    z_min = config.Property(proptype=float, default=0.8)
    z_max = config.Property(proptype=float, default=2.5)
    n_modes = config.Property(proptype=int, default=7)

    tracer = config.Property(proptype=str, default=None)

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
            nside=self.nside, polarisation=False, freq=freq_selfunc, attrs_from=cat
        )

        # Initialize selection function to zero
        selfunc["map"][:] = 0  # np.zeros(selfunc["map"].local_shape)

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

        # Set a tracer attribute
        if self.tracer is not None:
            selfunc.attrs["tracer"] = self.tracer

        return selfunc


class ResizeSelectionFunctionMap(task.SingleTask):
    """Take a selection function map and simulated source
    (biased density) map and return a selection function map with the
    same resolution and frequency sampling as the source map.

    Attributes
    ----------
    smooth : bool
        Smooth the resized selection function on the scale of the original
        pixel area. This helps to erase the imprint of the original pixelization
        on the resized map, particularly at the edges of the selection function.
    """

    smooth = config.Property(proptype=bool, default=False)

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
            polarisation=False, axes_from=source_map, attrs_from=source_map
        )

        # Form matrix to interpolate frequency/z axis
        interp_m = regrid.lanczos_forward_matrix(
            z_selfunc["centre"], z_source["centre"]
        )
        # Correct for redshift bin widths:
        interp_m *= z_source["width"][:, np.newaxis] / z_selfunc["width"][np.newaxis, :]

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
        nside = new_selfunc.nside

        # Get local section of container for output selection function
        new_selfunc_map_local = new_selfunc.map[:]

        # For each frequency in local section, up/downgrade healpix maps
        # of selection function to desired resolution, and set negative
        # pixel values (which the Lanczos interpolation can create) to zero
        for fi in range(selfunc_map_newz.local_shape[0]):
            new_selfunc_map_local[:][fi, 0] = hp.ud_grade(selfunc_map_newz[fi], nside)

            # If desired, convolve the resized selection function with a
            # Gaussian with FWHM equal to the sqrt of the original pixel area.
            # This smoothes out the edges of the map, which will otherwise retain
            # the shape of the original pixelization.
            if self.smooth:
                old_nside = selfunc.nside
                smoothing_fwhm = hp.nside2resol(old_nside)
                new_selfunc_map_local[:][fi, 0] = hp.smoothing(
                    new_selfunc_map_local[:][fi, 0], fwhm=smoothing_fwhm, verbose=False
                )

            new_selfunc_map_local[:][fi, 0][new_selfunc_map_local[:][fi, 0][:] < 0] = 0

        return new_selfunc


class PdfGeneratorBase(task.SingleTask):
    """Base class for PDF generator (non-functional).

    Take a source catalog selection function and simulated source
    (biased density) map and return a PDF map constructed from the
    product of the two, appropriately normalized. This PDF map can be used
    by the task :class:`MockCatalogGenerator` to draw mock catalogs.

    Derived classes must implement process().

    Attributes
    ----------
    tracer : str, optional
        Set an optional tracer attribute that can be used to identify the type of
        catalog later in the pipeline.
    """

    tracer = config.Property(proptype=str, default=None)

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

        if (rho < 0).any():
            self.log.error("Found negative entries in source map.")

        # Normalize density to have unit mean in each z-bin:
        rho = mpiarray.MPIArray.wrap(rho / np.mean(rho, axis=1)[:, np.newaxis], axis=0)

        if selfunc is not None:
            # Get local section of selection function
            selfunc_local = selfunc.map[:, 0, :]

            if (selfunc_local < 0).any():
                self.log.error("Found negative entries in selection function.")

            # Multiply selection function into density
            pdf = mpiarray.MPIArray.wrap(rho * selfunc_local, axis=0)

        else:
            pdf = mpiarray.MPIArray.wrap(rho, axis=0)

        # Normalize by redshift weights
        pdf = mpiarray.MPIArray.wrap(
            pdf
            * tools.invert_no_zero(np.sum(pdf, axis=1))[:, np.newaxis]
            * z_weights[:, np.newaxis],
            axis=0,
        )

        # Make container for PDF
        pdf_map = containers.Map(
            nside=source_map.nside,
            polarisation=False,
            freq=source_map.index_map["freq"],
            attrs_from=selfunc,
        )

        # Put computed PDF into local section of container
        pdf_map_local = pdf_map.map[:]
        pdf_map_local[:, 0, :] = pdf

        # Set a tracer attribute
        if self.tracer is not None:
            pdf_map.attrs["tracer"] = self.tracer

        return pdf_map

    def process(self):
        raise NotImplementedError(f"{self.__class__} must define a process method.")


class PdfGeneratorUncorrelated(PdfGeneratorBase):
    """Generate uniform PDF for making uncorrelated mocks."""

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
        z_weights = mpiarray.MPIArray.wrap(1 / gs * np.ones(ls), axis=0)

        # Create PDF map
        pdf_map = self.make_pdf_map(source_map, z_weights)

        return pdf_map


class PdfGeneratorWithSelectionFunction(PdfGeneratorBase):
    """Generate PDF that incorporates a selection function."""

    def process(self, source_map, selfunc):
        """Make PDF map that incorporates the selection function.

        Parameters
        ----------
        source_map : :class:`containers.Map`
            Overdensity map that determines z and angular resolution
            of output PDF map.
        selfunc : :class:`containers.Map`
            Selection function map. Must have same z and angular resolution
            as source_map. Typically taken from `ResizeSelectionFunctionMap`.

        Returns
        -------
        pdf_map : :class:`containers.Map`
            Output PDF map.
        """

        # Get local section of selection function
        selfunc_local = selfunc.map[:, 0, :]

        # Generate weights for distribution of sources in redshift:
        # first, sum over selfunc pixel values at each z (z_weights),
        # then sum these over all z (z_weights_sum).
        z_weights = selfunc_local.sum(axis=1)
        z_weights_sum = self.comm.allreduce(z_weights.sum())

        # Normalize z_weights by grand total
        z_weights = mpiarray.MPIArray.wrap(z_weights / z_weights_sum, axis=0)

        # Create PDF map
        pdf_map = self.make_pdf_map(source_map, z_weights, selfunc)

        return pdf_map


class PdfGeneratorNoSelectionFunction(PdfGeneratorBase):
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
            z_weights = mpiarray.MPIArray.wrap(1 / gs * np.ones(ls), axis=0)

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

                z_weights_global[fi] = cosmo.comoving_distance(z_mean) ** 2 * (
                    cosmo.comoving_distance(z_max) - cosmo.comoving_distance(z_min)
                )

            z_weights_global /= z_weights_global.sum()

            # Select local section of weights
            z_weights = mpiarray.MPIArray.wrap(z_weights_global[lo : lo + ls], axis=0)

        # Create PDF map
        pdf_map = self.make_pdf_map(source_map, z_weights)

        return pdf_map


class MockCatalogGenerator(task.SingleTask, random.RandomTask):
    """Take PDF maps generated by task :class:`PdfGenerator`
    and use it to draw mock catalogs.

    Attributes
    ----------
    nsource : int
        Number of sources to draw in each mock catalog.
    ncat : int
        Number of catalogs to generate.
    z_at_channel_centers : bool, optional
        Place each source at a redshift corresponding to the center of
        its frequency channel (True), or randomly distribute each source's
        redshift within its channel (False). Default: False.
    srcs_at_pixel_centers : bool, optional
        Place each source precisely at Healpix pixel center (True), or
        randomly distribute each source within pixel (False).
        Default: False.
    """

    nsource = config.Property(proptype=int)
    ncat = config.Property(proptype=int)

    z_at_channel_centers = config.Property(proptype=bool, default=False)
    srcs_at_pixel_centers = config.Property(proptype=bool, default=False)

    def setup(self, pdf_map):
        """Pre-load information from PDF.

        Parameters
        ----------
        pdf_map : :class:`containers.Map`
            PDF from which to draw positions of sources.
        """

        # Get PDF map container and corresponding healpix Nside
        self.pdf = pdf_map
        self.nside = self.pdf.nside

        # Get MPI rank
        self.rank = self.comm.Get_rank()

        # Get local shapes and offsets of frequency axis
        self.lo = self.pdf.map[:, 0, :].local_offset[0]
        self.ls = self.pdf.map[:, 0, :].local_shape[0]
        self.lo_list = self.comm.allgather(self.lo)
        self.ls_list = self.comm.allgather(self.ls)

        # Global shape of frequency axis
        n_z = self.pdf.map[:, 0, :].global_shape[0]

        # Weight of each redshift bin in the PDF, as sum over all
        # PDF map pixels at that redshift
        z_weights = np.sum(self.pdf.map[:, 0, :], axis=1)

        # Initialize array to hold global z_weights
        if self.rank == 0:
            # Only rank zero is relevant
            self.global_z_weights = np.zeros(n_z)
        else:
            # All processes must have a value for self.global_z_weights
            self.global_z_weights = None

        # Gather z_weights on rank 0 (necessary to draw a redshift
        # distribution of sources):
        self.comm.Gatherv(
            z_weights,
            [
                self.global_z_weights,
                tuple(self.ls_list),
                tuple(self.lo_list),
                MPI.DOUBLE,
            ],
            root=0,
        )

        # CDF to draw sources from, as cumulative sum over pixel values
        # at each redshift
        self.cdf = np.cumsum(self.pdf.map[:, 0, :], axis=1)
        # Normalize CDF by final entry
        self.cdf = self.cdf / self.cdf[:, -1][:, np.newaxis]

    def process(self):
        """Make a mock catalog based on input PDF.

        Returns
        ----------
        mock_catalog : :class:`containers.SpectroscopicCatalog`
            Simulated catalog.
        """

        if self.rank == 0:
            # Only rank zero is relevant.
            # The number of sources in each redshift bin follows a multinomial
            # distribution (reshape from (1,nz) to (nz) to make a 1D array):
            global_source_numbers = self.rng.multinomial(
                self.nsource, self.global_z_weights
            )
        else:
            # All processes must have a value for source_numbers:
            global_source_numbers = None

        # Send number of sources per redshift to local sections on each rank.
        # Need to pass tuples. For some reason lists don't work.
        # source_numbers has shape (self.ls)
        source_numbers = np.zeros(self.ls, dtype=np.int)
        self.comm.Scatterv(
            [
                global_source_numbers,
                tuple(self.ls_list),
                tuple(self.lo_list),
                MPI.DOUBLE,
            ],
            source_numbers,
        )

        # Compute the square root of the angular pixel area,
        # as a gross approximation of the pixel size.
        ang_size = np.rad2deg(hp.nside2resol(self.nside))

        # Redshifts corresponding to frequencies at bin centers
        z_global = _freq_to_z(self.pdf.index_map["freq"][:])

        # Get total number of sources on this rank, and make arrays to hold
        # information for all sources
        nsource_rank = source_numbers.sum()
        mock_zs = np.empty(nsource_rank, dtype=np.float64)
        mock_ra = np.empty(nsource_rank, dtype=np.float64)
        mock_dec = np.empty(nsource_rank, dtype=np.float64)

        # Loop over local redshift bins
        source_offset = 0
        for zi, nsource_bin in enumerate(source_numbers):
            # Draw a uniform random number in [0,1] for each source.
            # This will determine which angular pixel the source is assigned to.
            rnbs = self.rng.uniform(size=nsource_bin)

            # For each source, determine index of pixel the source falls into
            pix_idxs = np.digitize(rnbs, self.cdf[zi])

            # If desired, generate random numbers to randomize position of sources
            # within z bin. These are uniform random numbers in [-0.5, 0.5], which
            # will determine the source's relative displacement from the bin's
            # mean redshift.
            if not self.z_at_channel_centers:
                rz = self.rng.uniform(size=nsource_bin) - 0.5

            # If desired, generate random numbers to randomize position of sources
            # in each healpix pixel. These are uniform random numbers in [-0.5, 0.5],
            # which will determine the source's relative displacement from the pixel's
            # central RA and dec.
            if not self.srcs_at_pixel_centers:
                rtheta = self.rng.uniform(size=nsource_bin) - 0.5
                rphi = self.rng.uniform(size=nsource_bin) - 0.5

            # Get global index of z bin, and make array of z values of sources,
            # set to central z of bin
            global_z_index = zi + self.lo
            z_value = z_global["centre"][global_z_index] * np.ones(nsource_bin)

            # Get dec, RA of center of pixel containing each source
            decbase, RAbase = _pix_to_radec(pix_idxs, self.nside)
            # If desired, add random angular offsets from pixel centers
            if not self.srcs_at_pixel_centers:
                decbase += ang_size * rtheta
                RAbase += ang_size * rphi

            # If desired, add random offset within z bin to z of each source
            if not self.z_at_channel_centers:
                z_value += z_global["width"][global_z_index] * rz

            # Populate local arrays of source redshift, RA, dec
            mock_zs[source_offset : source_offset + nsource_bin] = z_value
            mock_ra[source_offset : source_offset + nsource_bin] = RAbase
            mock_dec[source_offset : source_offset + nsource_bin] = decbase

            # Increment source_offset to start of next block of sources
            # in mock_... arrays
            source_offset += nsource_bin

        # Define arrays to hold full source catalog
        mock_zs_full = np.empty(self.nsource, dtype=mock_zs.dtype)
        mock_ra_full = np.empty(self.nsource, dtype=mock_ra.dtype)
        mock_dec_full = np.empty(self.nsource, dtype=mock_dec.dtype)

        # Tuple (not list!) of number of sources in each rank
        # Note: the counts and displacement arguments of Allgatherv are tuples!
        nsource_tuple = tuple(self.comm.allgather(nsource_rank))
        # Tuple (not list!) of displacements of each rank array in full array
        dspls = tuple(np.insert(arr=np.cumsum(nsource_tuple)[:-1], obj=0, values=0.0))
        # Gather redshifts
        recvbuf = [mock_zs_full, nsource_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_zs, len(mock_zs)]
        self.comm.Allgatherv(sendbuf, recvbuf)
        # Gather theta
        recvbuf = [mock_dec_full, nsource_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_dec, len(mock_dec)]
        self.comm.Allgatherv(sendbuf, recvbuf)
        # Gather phi
        recvbuf = [mock_ra_full, nsource_tuple, dspls, MPI.DOUBLE]
        sendbuf = [mock_ra, len(mock_ra)]
        self.comm.Allgatherv(sendbuf, recvbuf)

        # Create catalog container
        mock_catalog = containers.SpectroscopicCatalog(
            object_id=np.arange(self.nsource, dtype=np.uint64),
            attrs_from=self.pdf,
        )

        # Create position and redshift datasets
        mock_catalog["position"][:] = np.empty(
            self.nsource, dtype=[("ra", mock_ra.dtype), ("dec", mock_dec.dtype)]
        )
        mock_catalog["redshift"][:] = np.empty(
            self.nsource, dtype=[("z", mock_zs.dtype), ("z_error", mock_zs.dtype)]
        )
        # Assign data to catalog container
        mock_catalog["position"]["ra"][:] = mock_ra_full
        mock_catalog["position"]["dec"][:] = mock_dec_full
        mock_catalog["redshift"]["z"][:] = mock_zs_full
        mock_catalog["redshift"]["z_error"][:] = 0

        # If we've created the requested number of mocks, prepare to exit
        if self._count == self.ncat - 1:
            self.done = True

        return mock_catalog


class AddGaussianZErrorsToCatalog(task.SingleTask, random.RandomTask):
    """Add random Gaussian redshift errors to redshifts in a catalog.

    The standard deviation of the errors is determined either
    by sigma_z or sigma_z / (1+z), or by the `z_error` field of the
    catalog.

    Note that the errors are added to the catalog in place, such that
    if the original catalog is subsequently used in a pipeline, it will
    have the errors included.

    Attributes
    ----------
    use_catalog_z_errors : bool
        Set standard deviation of Gaussian error based on `z_error` value
        for each source in catalog. If True, overrides `sigma`.
        Default: False.
    sigma : float
        Standard deviation corresponding to choice in `sigma_type`.
    sigma_type : string
        Interpretation of `sigma`:
            'sigma_z' - Standard deviation of Gaussian for z errors.
            'sigma_z_over_1plusz' - Standard deviation divided by (1+z).
    """

    use_catalog_z_errors = config.Property(proptype=bool, default=False)
    sigma = config.Property(proptype=float)
    sigma_type = config.enum(["sigma_z", "sigma_z_over_1plusz"])

    def process(self, cat):
        """Generate random redshift errors and add to redshifts in catalog.

        Parameters
        ----------
        cat : :class:`containers.SpectroscopicCatalog`
            Input catalog.

        Returns
        ----------
        cat_out : :class:`containers.SpectroscopicCatalog`
            Catalog with redshift errors added.
        """

        # Get redshifts from catalog
        cat_z = cat["redshift"]["z"][:]
        cat_z_err = cat["redshift"]["z_error"][:]

        # Generate standard normal z errors
        z_err = self.rng.normal(size=cat_z.shape[0])
        # Multiply by appropriate sigma
        if self.use_catalog_z_errors:
            if not np.any(cat_z_err):
                self.log.error(
                    "Warning: no existing z_error information in catalog, so no z errors will be added"
                )
            z_err *= cat_z_err
        elif self.sigma_type == "sigma_z":
            z_err *= self.sigma
        else:  # self.sigma_type == "sigma_z_over_1plusz"
            z_err *= self.sigma * (1 + cat_z)

        # Add errors to catalog redshifts
        cat_z += z_err

        # TODO: store information about error distribution in z_error field

        return cat


class AddEBOSSZErrorsToCatalog(task.SingleTask, random.RandomTask):
    """Add eBOSS-type random redshift errors to redshifts in a catalog.

    See the docstrings for _{qso,elg,lrg}_velocity_error() for descriptions
    of each redshift error distribution.

    Note also that the errors are added to the catalog in place, such that
    if the original catalog is subsequently used in a pipeline, it will
    have the errors included.

    Attributes
    ----------
    tracer : {"ELG"|"LRG"|"QSO"|"QSOalt"}
        Generate redshift errors corresponding to this eBOSS sample.
        If not specified, task will attempt to detect the tracer type from
        the catalog's `tracer` attribute or its tag. Default: None
    """

    tracer = config.enum(["QSO", "ELG", "LRG", "QSOalt"], default=None)

    def process(self, cat):
        """Generate random redshift errors and add to redshifts in catalog.

        Parameters
        ----------
        cat : :class:`containers.SpectroscopicCatalog`
            Input catalog.

        Returns
        ----------
        cat_out : :class:`containers.SpectroscopicCatalog`
            Catalog with redshift errors added.
        """

        tracer = self.tracer

        # If tracer not specified in config, check to see whether it's stored
        # in the catalog's 'tracer' attribute or in its tag
        if tracer is None:
            if "tracer" in cat.attrs:
                tracer = cat.attrs["tracer"].upper()

                if tracer not in _velocity_error_function_lookup:
                    raise ValueError(
                        f"Tracer explicitly set to '{tracer}' in catalog, but value not supported."
                    )
            else:
                for key in _velocity_error_function_lookup.keys():
                    if key in cat.attrs["tag"].upper():
                        tracer = key
                        break

                if tracer is None:
                    raise ValueError(
                        "Must specify eBOSS tracer in config property, "
                        "catalog 'tracer' attribute, or catalog 'tag' attribute."
                    )

        self.log.info(f"Applying {tracer} redshift errors.")

        # Get redshifts from catalog
        cat_z = cat["redshift"]["z"][:]
        cat_z_err = cat["redshift"]["z_error"][:]

        # Generate redshift errors for the chosen tracer
        z_err = self._generate_z_errors(cat_z, tracer)

        # Add errors to catalog redshifts
        cat_z += z_err

        # TODO: store information about error distribution in z_error field

        return cat

    def _generate_z_errors(self, z, tracer):
        """Generate redshift errors using a tracer-specific velocity error distribution.

        See e.g. Eq. (A1) from https://arxiv.org/abs/1012.2912 for the
        relationship between redshift errors and peculiar velocity errors.

        Parameters
        ----------
        z: np.ndarray[nsource,]
            Source redshifts.
        tracer : {"ELG"|"LRG"|"QSO"}
            Name of the tracer.

        Returns
        -------
        dz: np.ndarray[nsource,]
            Perturbations to source redshifts based on random velocity errors.
        """

        if tracer not in _velocity_error_function_lookup:
            raise ValueError(
                f"Do not recognize {tracer}.  Must define a method "
                "for drawing random velocity errors for this tracer."
            )

        err_func = _velocity_error_function_lookup[tracer]

        dv = err_func(z, self.rng)

        dz = (1.0 + z) * dv / (C * 1e-3)

        return dz

    @staticmethod
    def qso_velocity_error(z, rng):
        """Draw random velocity errors for quasars.

        This is taken from Lyke et al. 2020 (https://arxiv.org/abs/2007.09001).
        Section 4.6 and Appendix A are the relevant parts. Figure 4 shows the
        distribution of redshift errors. It is well modelled by the sum of
        two Gaussians with standard deviations 150 and 1000 km/s.
        Roughly 1/6 of the quasars belong to the wider Gaussian.

        Parameters
        ----------
        z : np.ndarray
            True redshift for the object.
        rng : numpy.random.Generator
            Numpy RNG to use for generating random numbers.

        Returns
        -------
        dv: np.ndarray[nsample,]
            Velocity errors in km / s.
        """

        QSO_SIG1 = 150.0
        QSO_SIG2 = 1000.0
        QSO_F = 4.478

        nsample = len(z)

        dv1 = rng.normal(scale=QSO_SIG1, size=nsample)
        dv2 = rng.normal(scale=QSO_SIG2, size=nsample)

        u = rng.uniform(size=nsample)
        flag = u >= (1.0 / (1.0 + QSO_F))

        dv = np.where(flag, dv1, dv2)

        return dv

    @staticmethod
    def qsoalt_velocity_error(z, rng):
        """Draw random velocity errors for quasars using a redshift dependent model.

        This is based on the Lyke et al. model use in `qso_velocity_error` but fixing an
        issue with the fraction of quasars in the wide distribution at all redshifts, and
        reducing the errors at low redshift to account for the behaviour seen in Figure
        9 on Lyke et al.

        Parameters
        ----------
        z : np.ndarray
            True redshift for the object.
        rng : numpy.random.Generator
            Numpy RNG to use for generating random numbers.

        Returns
        -------
        dv: np.ndarray[nsample,]
            Velocity errors in km / s.
        """
        QSO_SIG1_highz = 150.0
        QSO_SIG1_lowz = 90.0
        QSO_SIG2 = 1000.0

        QSO_F_highz = 35.0
        QSO_ztrans = 1.0
        QSO_zwidth = 0.05

        def smooth_step_function(z, zt, zw, fl, fh):
            return (1 + np.tanh((z - zt) / zw)) * (fh - fl) / 2 + fl

        def invfz(z):
            return smooth_step_function(z, QSO_ztrans, QSO_zwidth, 0, 1 / QSO_F_highz)

        def sig1z(z):
            return smooth_step_function(
                z, QSO_ztrans, QSO_zwidth, QSO_SIG1_lowz, QSO_SIG1_highz
            )

        nsample = len(z)

        # A random variable to decide which Gaussian to draw the error from
        invf = invfz(z)
        u = rng.uniform(size=nz)
        flag = u >= (invf / (1.0 + invf))

        dv1 = rng.standard_normal(nsample) * sig1z(z)
        dv2 = rng.standard_normal(nsample) * QSO_SIG2

        dv = np.where(flag, dv1, dv2)

        return dv

    @staticmethod
    def lrg_velocity_error(z, rng):
        """Draw random velocity errors for luminous red galaxies.

        This is taken from Ross et al. 2020 (https://arxiv.org/abs/2007.09000).  Figure
        2 shows the distribution of redshift differences for repeated observations of
        the same object; this is well fit by a Gaussian with width 92 km/s. They state
        that this corresponds to a Gaussian distribution for the single-measurement
        redshift errors, with width 65.6 km/s.  There is a bit of a tail that is not
        being captured in their Gaussian fit, and is hence not simulated in this
        routine.

        Parameters
        ----------
        z : np.ndarray
            True redshift for the object.
        rng : numpy.random.Generator
            Numpy RNG to use for generating random numbers.

        Returns
        -------
        dv: np.ndarray[nsample,]
            Velocity errors in km / s.
        """

        LRG_SIG = 65.6

        dv = rng.normal(scale=LRG_SIG, size=len(z))

        return dv

    @staticmethod
    def elg_velocity_error(z, rng):
        """Draw random velocity errors for emission line galaxies.

        This is taken from Raichoor et al. 2020 (https://arxiv.org/abs/2007.09007).
        They do not plot the error distribution, but Section 2.3 provides three
        percentiles:
            "Additionally, we can assess with repeats that 99.5, 95, and 50
            percent of our redshift estimates have a precision better than
            300 km s−1, 100 km s−1, and 20 km s−1, respectively."
        These percentiles do not follow a Gaussian, but are reasonably well fit
        by a Tukey lambda distribution if the scale and shape parameters
        are allowed to float.

        Parameters
        ----------
        z : np.ndarray
            True redshift for the object.
        rng : numpy.random.Generator
            Numpy RNG to use for generating random numbers.

        Returns
        -------
        dv: np.ndarray[nsample,]
            Velocity errors in km / s.
        """

        ELG_SIG = 11.877
        ELG_LAMBDA = -0.4028

        dist = scipy.stats.tukeylambda
        dist.random_state = rng
        dv = dist.rvs(ELG_LAMBDA, scale=ELG_SIG, size=len(z))

        return dv


_velocity_error_function_lookup = {
    "QSO": AddEBOSSZErrorsToCatalog.qso_velocity_error,
    "QSOalt": AddEBOSSZErrorsToCatalog.qsoalt_velocity_error,
    "ELG": AddEBOSSZErrorsToCatalog.elg_velocity_error,
    "LRG": AddEBOSSZErrorsToCatalog.lrg_velocity_error,
}


class MapPixelLocationGenerator(task.SingleTask):
    """Generate a 'catalog' of Healpix pixel centers.

    This is useful if you want to stack on each Healpix pixel for
    a given Healpix resolution (determined by an input map).
    This task outputs a SpectroscopicCatalog
    that can then be fed to the usual beamforming task.

    All "sources" are assigned to the same frequency channel, for simplicity.

    Attributes
    ----------
    freq_idx : int
        Index of frequency channel to assign to all "sources".
    """

    freq_idx = config.Property(proptype=int)

    def setup(self, in_map):
        """Pre-load information from input map."""
        self.map_ = in_map

        # Get MPI rank
        self.rank = self.comm.Get_rank()

        # Global shape of frequency axis
        n_z = self.map_.map[:, 0, :].global_shape[0]

        # Get desired N_pix and Nside
        self.npix = len(self.map_.index_map["pixel"])
        self.nside = self.map_.nside

        # Get redshift to assign to all "sources"
        self.z_arr = _freq_to_z(self.map_.index_map["freq"])
        self.z = self.z_arr[self.freq_idx]["centre"]

    def process(self):
        """Make a catalog of pixel positions.

        Returns
        ----------
        mock_catalog : :class:`containers.SpectroscopicCatalog`
            Output catalog.
        """

        # Get local section of Healpix pixel indices
        local_pix_indices = mpiutil.partition_list_mpi(np.arange(self.npix))
        npix_rank = len(local_pix_indices)

        # Convert pixel indices to (dec,RA)
        pix_dec, pix_ra = _pix_to_radec(local_pix_indices, self.nside)

        # Make arrays to hold the whole source set information
        ra_full = np.empty(self.npix, dtype=pix_ra.dtype)
        dec_full = np.empty(self.npix, dtype=pix_dec.dtype)

        # Tuple (not list!) of number of pixels in each rank
        # The counts and displacement arguments of Allgatherv are tuples!
        npix_tuple = tuple(self.comm.allgather(npix_rank))
        # Tuple (not list!) of displacements of each rank array in full array
        dspls = tuple(np.insert(arr=np.cumsum(npix_tuple)[:-1], obj=0, values=0.0))
        # Gather theta
        recvbuf = [ra_full, npix_tuple, dspls, MPI.DOUBLE]
        sendbuf = [pix_ra, len(pix_ra)]
        self.comm.Allgatherv(sendbuf, recvbuf)
        # Gather phi
        recvbuf = [dec_full, npix_tuple, dspls, MPI.DOUBLE]
        sendbuf = [pix_dec, len(pix_dec)]
        self.comm.Allgatherv(sendbuf, recvbuf)

        # Create catalog container
        mock_catalog = containers.SpectroscopicCatalog(
            object_id=np.arange(self.npix, dtype=np.uint64)
        )

        # Create position and redshift datasets
        mock_catalog["position"][:] = np.empty(
            self.npix, dtype=[("ra", pix_ra.dtype), ("dec", pix_dec.dtype)]
        )
        mock_catalog["redshift"][:] = np.empty(
            self.npix, dtype=[("z", pix_ra.dtype), ("z_error", pix_ra.dtype)]
        )
        # Assign data to catalog container
        mock_catalog["position"]["ra"][:] = ra_full
        mock_catalog["position"]["dec"][:] = dec_full
        mock_catalog["redshift"]["z"][:] = self.z * np.ones(
            self.npix, dtype=pix_ra.dtype
        )
        mock_catalog["redshift"]["z_error"][:] = 0.0

        self.done = True
        return mock_catalog


# Internal functions
# ------------------


def _zlims_to_freq(z, zlims):
    """Convert redshift bins to frequency.

    Parameters
    ----------
    z : np.array
        Redshift bin centers.
    zlims : np.array
        Redshift bin edges.

    Returns
    -------
    freqs : np.ndarray
        Array of tuples of frequency bin centers and widths.
    """
    freqcentre = units.nu21 / (z + 1)
    freqlims = units.nu21 / (zlims + 1)
    freqwidth = abs(freqlims[:-1] - freqlims[1:])
    return np.array(
        [(freqcentre[ii], freqwidth[ii]) for ii in range(len(z))],
        dtype=[("centre", "<f8"), ("width", "<f8")],
    )


def _freq_to_z(freq):
    """Convert frequency bins to redshift.

    Parameters
    ----------
    freq : np.array
        Array of tuples of frequency bin centers and widths.

    Returns
    -------
    freq : np.ndarray
        Array of tuples of z bin centers and widths
    """
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
    """Convert healpix pixel indices to (dec, RA).

    Parameters
    ----------
    index : np.array
        Array of healpix pixel indices.
    nside : int
        Healpix nside corresponding to pixel indices.

    Returns
    -------
    dec, RA : np.ndarray
        Output dec and ra coordinates, in degrees.
    """
    theta, phi = hp.pix2ang(nside, index)
    return -np.degrees(theta - np.pi / 2.0), np.degrees(phi)


def _radec_to_pix(ra, dec, nside):
    """Convert (RA, dec) to nearest healpix pixels.

    Parameters
    ----------
    ra, dec : np.array
        Input RA and dec coordinates, in degrees.
    nside : int
        Healpix nside corresponding to input coordinates.

    Returns
    -------
    index : np.array
        Array of healpix pixel indices.
    """
    return hp.ang2pix(nside, np.radians(-dec + 90.0), np.radians(ra))


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
    idxs = np.digitize(cat["redshift"]["z"], zlims_selfunc) - 1  # -1 to get indices
    # Map pixel of each source
    pixels = _radec_to_pix(cat["position"]["ra"], cat["position"]["dec"], nside)

    for zi in range(n_z):
        # Get map pixels containing sources in redshift bin zi
        zpixels = pixels[idxs == zi]
        # For each pixel in map, set pixel value to number of sources
        # within that pixel
        for pi in range(n_pix):
            maps[zi, pi] = np.sum(zpixels == pi)

    return maps

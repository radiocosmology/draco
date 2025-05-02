"""Map making tasks (:mod:`~draco.analysis.ringmapmaker`).

.. currentmodule:: draco.analysis.ringmapmaker

Tools for beamforming data for arrays with a cartesian layout.

Tasks
=====

.. autosummary::
    :toctree: generated/

    MakeVisGrid
    BeamformNS
    BeamformEW
    RingMapMaker
    TikhonovRingMapMakerAnalytical
    TikhonovRingMapMakerExternal
    WienerRingMapMakerAnalytical
    WienerRingMapMakerExternal
    RADependentWeights
    ReconstructVisWeight
"""

import numpy as np
import scipy.constants
from caput import config, task
from mpi4py import MPI
from numpy.lib.recfunctions import structured_to_unstructured

from ..core import containers, io
from ..util import tools
from ..util.exception import ConfigError
from . import transform


class MakeVisGrid(task.SingleTask):
    """Arrange the visibilities onto a 2D grid.

    This will fill out the visibilities in the half plane `x >= 0` where x is the EW
    baseline separation.

    Attributes
    ----------
    centered : bool
        If True, place the zero NS separation at the center of the y-axis with
        the baselines given in ascending order. Otherwise the zero separation
        is at position zero, and the baselines are in FFT order.  Default is False.
    save_redundancy : bool
        If True, computes and stores the redundancy of each visibility.
        Default is True.
    """

    centered = config.Property(proptype=bool, default=False)
    save_redundancy = config.Property(proptype=bool, default=True)

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
            Telescope object to use
        """
        self.telescope = io.get_telescope(tel)

    def process(self, sstream):
        """Computes the ringmap.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        rm : containers.RingMap
        """
        # Convert prodstack into a type that can be easily compared against
        # `uniquepairs`
        ps_sstream = structured_to_unstructured(sstream.prodstack, dtype=np.int16)
        ps_tel = structured_to_unstructured(self.telescope.prodstack, dtype=np.int16)

        if not np.array_equal(ps_sstream, ps_tel):
            raise ValueError(
                "Products in sstream do not match those in the beam transfers."
            )

        # Calculation the set of polarisations in the data, and which polarisation
        # index every entry corresponds to
        polpair = self.telescope.polarisation[self.telescope.uniquepairs].view("U2")
        pol, pind = np.unique(polpair, return_inverse=True)

        if len(pol) != 4:
            raise RuntimeError(f"Expected to find four polarisations. Got {pol}")

        # Find the mapping from a polarisation index to it's complement with the feeds
        # reversed
        pconjmap = np.unique([pj + pi for pi, pj in pol], return_inverse=True)[1]

        # Determine the layout of the visibilities on the grid. This isn't trivial
        # because of potential rotation of the telescope with respect to NS
        xind, yind, min_xsep, min_ysep = find_grid_indices(self.telescope.baselines)

        # Define several variables describing the baseline configuration.
        nx = np.abs(xind).max() + 1
        max_yind = np.abs(yind).max()
        ny = 2 * max_yind + 1
        vis_pos_x = np.arange(nx) * min_xsep

        if self.centered:
            vis_pos_y = np.arange(-max_yind, max_yind + 1) * min_ysep
            ns_offset = max_yind
        else:
            vis_pos_y = np.fft.fftfreq(ny, d=(1.0 / (ny * min_ysep)))
            ns_offset = 0

        # Extract the right ascension to initialise the new container with (or
        # calculate from timestamp)
        if "ra" in sstream.index_map:
            ra = sstream.ra
        elif "lsd" in sstream.attrs:
            ra = 360 * (self.telescope.unix_to_lsd(sstream.time) - sstream.attrs["lsd"])
        else:
            ra = self.telescope.lsa(sstream.time)

        # Create container for output
        grid = containers.VisGridStream(
            pol=pol,
            ew=vis_pos_x,
            ns=vis_pos_y,
            ra=ra,
            axes_from=sstream,
            attrs_from=sstream,
        )

        # Calculate the redundancy
        if self.save_redundancy:
            redundancy = tools.calculate_redundancy(
                sstream.input_flags[:],
                sstream.index_map["prod"][:],
                sstream.reverse_map["stack"]["stack"][:],
                sstream.vis.shape[1],
            )

            grid.add_dataset("redundancy")

        # Redistribute over frequency
        sstream.redistribute("freq")
        grid.redistribute("freq")

        # De-reference distributed arrays outside loop to save repeated MPI calls
        ssv = sstream.vis[:].local_array
        ssw = sstream.weight[:].local_array

        gsv = grid.vis[:].local_array
        gsv[:] = 0.0

        gsw = grid.weight[:].local_array
        gsw[:] = 0.0

        if self.save_redundancy:
            gsr = grid.redundancy[:]
            gsr[:] = 0

        # Unpack visibilities into new array
        for vis_ind, (p_ind, x_ind, y_ind) in enumerate(zip(pind, xind, yind)):
            # Different behavior for intracylinder and intercylinder baselines.
            gsv[p_ind, :, x_ind, ns_offset + y_ind, :] = ssv[:, vis_ind]
            gsw[p_ind, :, x_ind, ns_offset + y_ind, :] = ssw[:, vis_ind]
            if self.save_redundancy:
                gsr[p_ind, x_ind, ns_offset + y_ind, :] = redundancy[vis_ind]

            if x_ind == 0:
                pc_ind = pconjmap[p_ind]
                gsv[pc_ind, :, x_ind, ns_offset - y_ind, :] = ssv[:, vis_ind].conj()
                gsw[pc_ind, :, x_ind, ns_offset - y_ind, :] = ssw[:, vis_ind]
                if self.save_redundancy:
                    gsr[pc_ind, x_ind, ns_offset - y_ind, :] = redundancy[vis_ind]

        return grid


class BeamformNS(task.SingleTask):
    """A simple and quick map-maker that forms a series of beams on the meridian.

    This is designed to run on data after it has been collapsed down to
    non-redundant baselines only.

    Attributes
    ----------
    npix : int
        Number of map pixels in the declination dimension.  Default is 512.
    span : float
        Span of map in the declination dimension. Value of 1.0 generates a map
        that spans from horizon-to-horizon.  Default is 1.0.
    weight : string
        How to weight the non-redundant baselines.  Options include:
            'natural' - each baseline weighted by its redundancy (default)
            'inverse_variance' - each baseline weighted by the weight attribute
            'uniform' - each baseline given equal weight
        And any window function supported by drao.util.tools.window_generalised,
        such as 'hann', 'hanning', 'hamming', 'blackman', 'nuttall',
        'blackman_nuttall', 'blackman_harris' 'triangular', and 'tukey-0.X'.
    scaled : bool
        Scale the window to match the lowest frequency. This should make the
        beams more frequency independent.  Not supported for 'inverse_variance'
        and 'natural' weight.
    include_auto: bool
        Include autocorrelations in the calculation.  Default is False.
    save_dirty_beam : bool
        If True, computes and stores the dirty beam.  Default is False.
    precision : int
        Floating point precision to use when applying the beamforming
        matrix. Using 32-bit precision results in a 2-3x reduction in
        runtime, at the cost of potential numerical errors. Default is 64.
    """

    npix = config.Property(proptype=int, default=512)
    span = config.Property(proptype=float, default=1.0)
    weight = config.Property(proptype=str, default="natural")
    scaled = config.Property(proptype=bool, default=False)
    include_auto = config.Property(proptype=bool, default=False)
    save_dirty_beam = config.Property(proptype=bool, default=False)
    precision = config.enum([32, 64], default=64)

    def process(self, gstream):
        """Computes the ringmap.

        Parameters
        ----------
        gstream : VisGridStream
            The input stream.

        Returns
        -------
        bf : HybridVisStream
        """
        # Redistribute over frequency
        gstream.redistribute("freq")

        gsv = gstream.vis[:].local_array
        gsw = gstream.weight[:].local_array
        if self.weight == "natural":
            if "redundancy" not in gstream.datasets:
                raise ConfigError(
                    "Must set save_redundancy = True for task "
                    "MakeVisGrid in order to use a natural weight scheme."
                )
            gsr = gstream.redundancy[:]

        # Construct phase array
        el = self.span * np.linspace(-1.0, 1.0, self.npix)

        # Create empty ring map
        hv = containers.HybridVisStream(el=el, axes_from=gstream, attrs_from=gstream)
        if self.save_dirty_beam:
            hv.add_dataset("dirty_beam")
        hv.redistribute("freq")

        # Dereference datasets
        hvv = hv.vis[:].local_array
        hvw = hv.weight[:].local_array
        if self.save_dirty_beam:
            hvb = hv.dirty_beam[:].local_array

        nspos = gstream.index_map["ns"][:]
        freq = gstream.freq

        # Get the largest baseline present across all nodes while accounting for masking
        baselines_present = np.any(gsw > 0, axis=(0, 1, 2, 4))
        nsmax_local = (
            np.abs(nspos[baselines_present]).max()
            if baselines_present.sum() > 0
            else 0.0
        )
        nsmax = self.comm.allreduce(nsmax_local, op=MPI.MAX)
        self.log.info(f"Maximum NS baseline is {nsmax:.2f}m")

        # Record how the beamforming was done to enable easy
        # reconstruction of synthesized beam
        hv.attrs["beamform_ns_weight"] = self.weight
        hv.attrs["beamform_ns_scaled"] = self.scaled
        hv.attrs["beamform_ns_include_auto"] = self.include_auto
        hv.attrs["beamform_ns_freqmin"] = freq.min()
        hv.attrs["beamform_ns_nsmax"] = nsmax

        # choose the precision which will be used in `matmul`
        complex_dtype = np.dtype(f"complex{2*self.precision:.0f}")
        real_dtype = np.dtype(f"float{self.precision:.0f}")

        # precompute a phase array
        phase = (2.0 * np.pi * nspos[np.newaxis] * el[:, np.newaxis]).astype(
            complex_dtype
        )

        # Loop over local frequencies and fill ring map
        for lfi, fi in gstream.vis[:].enumerate(1):
            # Get the current frequency and wavelength
            fr = freq[fi]
            iwv = (fr * 1e6) / scipy.constants.c

            vpos = nspos * iwv

            if self.scaled:
                iwvmin = (freq.min() * 1e6) / scipy.constants.c
                vmax = nsmax * iwvmin
            else:
                vmax = nsmax * iwv

            if self.weight == "inverse_variance":
                gw = gsw[:, lfi].copy()
            elif self.weight == "natural":
                gw = gsr.astype(np.float32)
            else:
                x = 0.5 * (vpos / vmax + 1)
                ns_weight = tools.window_generalised(x, window=self.weight).astype(
                    real_dtype
                )
                gw = (gsw[:, lfi] > 0) * ns_weight[
                    np.newaxis, np.newaxis, :, np.newaxis
                ]

            # Ensure we skip entries which are flagged out entirely
            gw *= gsw[:, lfi] > 0

            # Remove auto-correlations
            if not self.include_auto:
                gw[..., 0, 0, :] = 0.0

            # Normalize by sum of weights
            norm = np.sum(gw, axis=-2)
            gw *= tools.invert_no_zero(norm)[..., np.newaxis, :]

            # Create array that will be used for the inverse
            # discrete Fourier transform in y-direction
            F = np.exp(-1.0j * phase * iwv)

            # Calculate the hybrid visibilities
            gv = gsv[:, lfi]
            np.matmul(F, gv * gw, out=hvv[:, lfi])

            # Calculate the dirty beam
            if self.save_dirty_beam:
                hvb[:, lfi] = np.matmul(F, gw * np.ones_like(gv)).real

            # Estimate the weights assuming that the errors are all uncorrelated
            t = np.sum(tools.invert_no_zero(gsw[:, lfi]) * gw**2, axis=-2)
            hvw[:, lfi] = tools.invert_no_zero(t)

        return hv


class BeamformEW(task.SingleTask):
    """Final beam forming in the EW direction.

    This is designed to run on data after the NS beam forming.

    Attributes
    ----------
    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.  Default is False.
    single_beam: bool
        Only calculate the map for the central beam. Default is False.
    weight_ew : string
        How to weight the EW baselines? One of:
            'natural' - weight by the redundancy of the EW baselines.
            'uniform' - give each EW baseline uniform weight.
    flag_ew : list
        List of boolean values with length equal to the number of EW baselines.
        If provided, this specifies what baselines are included in the calculation
        with True/False denoting include/exclude.
    """

    exclude_intracyl = config.Property(proptype=bool, default=False)
    single_beam = config.Property(proptype=bool, default=False)
    weight_ew = config.enum(["natural", "uniform"], default="natural")
    flag_ew = config.Property(proptype=np.array)

    def process(self, hstream):
        """Computes the ringmap.

        Parameters
        ----------
        hstream : HybridVisStream
            The input stream.

        Returns
        -------
        rm : RingMap
        """
        # Redistribute over frequency
        hstream.redistribute("freq")

        # Create empty ring map
        n_ew = len(hstream.index_map["ew"])
        nbeam = 1 if self.single_beam else 2 * n_ew - 1

        # Determine the weighting coefficient in the EW direction
        if self.weight_ew == "uniform":
            weight_ew = np.ones(n_ew)
        else:  # self.weight_ew == "natural"
            weight_ew = n_ew - np.arange(n_ew)

        # Exclude the in cylinder baselines if requested (EW = 0)
        if self.exclude_intracyl:
            weight_ew[0] = 0.0

        # Apply a flag if provided
        if self.flag_ew is not None and self.flag_ew.size == n_ew:
            weight_ew *= self.flag_ew.astype(bool).astype(weight_ew.dtype)

        # Factor to include negative elements in sum single beam sum
        if self.single_beam:
            weight_ew[1:] *= 2

        # Normalise the weights
        weight_ew = weight_ew / weight_ew.sum()

        # Reshape ew weights so that they will broadcast against
        # the vis and weight datasets
        weight_ew2 = weight_ew[:, np.newaxis] ** 2
        weight_ew = weight_ew[:, np.newaxis, np.newaxis]

        # Derive the new polarisation index map and rotation matrix
        pol, P = self._get_pol(hstream.index_map["pol"])
        P2 = np.abs(P) ** 2

        # Determine if we need to process the dirty beam
        save_dirty_beam = "dirty_beam" in hstream.datasets

        # Create ring map, copying over axes/attrs and add the optional datasets
        rm = containers.RingMap(
            beam=nbeam, pol=pol, axes_from=hstream, attrs_from=hstream
        )
        rm.add_dataset("rms")

        # Make sure ring map is distributed over frequency
        rm.redistribute("freq")

        # Dereference datasets
        hvv = hstream.vis[:].local_array
        hvw = hstream.weight[:].local_array

        if save_dirty_beam:
            # Only add this dataset if the input container has
            # a `dirty_beam` dataset
            rm.add_dataset("dirty_beam")
            rmb = rm.dirty_beam[:].local_array

            hvb = hstream.dirty_beam[:].local_array

        rmm = rm.map[:].local_array
        rmw = rm.weight[:].local_array
        rmr = rm.rms[:].local_array

        # Loop over local frequencies and fill ring map
        for lfi, fi in hstream.vis[:].enumerate(axis=1):
            # Rotate the polarisations
            v = np.tensordot(P, hvv[:, lfi], axes=(1, 0))

            # Apply the EW weighting
            v *= weight_ew

            # Perform  inverse fast fourier transform in x-direction
            if self.single_beam:
                # Only need the 0th term of the irfft, equivalent to summing in
                # then EW direction
                beamformed_data = np.sum(v.real, axis=1)[:, np.newaxis]
            else:
                beamformed_data = np.fft.irfft(v, nbeam, axis=1) * nbeam

            # Save to container (shifting to the final axis ordering)
            rmm[:, :, lfi] = beamformed_data.transpose(1, 0, 3, 2)

            # Propagate variance in the visibilities to the ringmap.
            # Factor of 1/2 because we are taking the real component.
            var = np.tensordot(P2, tools.invert_no_zero(hvw[:, lfi]), axes=(1, 0))
            rm_var = 0.5 * np.sum(weight_ew2 * var, axis=1)

            rmw[:, lfi] = tools.invert_no_zero(rm_var[..., np.newaxis])
            rmr[:, lfi] = rm_var**0.5

            # Repeat all the same operations for the dirty beam if available.
            if save_dirty_beam:
                b = np.tensordot(P, hvb[:, lfi], axes=(1, 0))
                b *= weight_ew[np.newaxis, :, np.newaxis, np.newaxis]

                if self.single_beam:
                    dirty_beam = np.sum(b.real, axis=1)[:, np.newaxis]
                else:
                    dirty_beam = np.fft.irfft(b, nbeam, axis=1) * nbeam

                rmb[:, :, lfi] = dirty_beam.transpose(1, 0, 3, 2)

        return rm

    @staticmethod
    def _get_pol(pols):
        """Derive the output polarizations based on the input index map."""
        # Require both cross-pol terms to exist if at least one exists
        if ("XY" in pols) or ("YX" in pols):
            if ("XY" in pols) ^ ("YX" in pols):
                raise ValueError(
                    "If cross-pols exist, both XY and YX must be present. "
                    f"Got {pols}."
                )
            dpol = ["reXY", "imXY"]
        else:
            dpol = []

        if "XX" in pols:
            # XX is first
            dpol = ["XX", *dpol]

        if "YY" in pols:
            # YY is last
            dpol.append("YY")

        # This matrix takes the linear combinations of polarisations
        # required to rotate from XY, YX basis into reXY and imXY
        P = np.eye(len(dpol), dtype=np.complex64)

        # add the cross-pol terms if they exist
        if "reXY" in dpol:
            i = dpol.index("reXY")
            P[i, i : i + 2] = [0.5, 0.5]
            P[i + 1, i : i + 2] = [-0.5j, 0.5j]

        return np.array(dpol, dtype="U4"), P


class RingMapMaker(task.group_tasks(MakeVisGrid, BeamformNS, BeamformEW)):
    """Make a ringmap from the data."""


class DeconvolveHybridMBase(task.SingleTask):
    """Base class for deconvolving ringmapmakers (non-functional).

    Attributes
    ----------
    exclude_intracyl : bool
        Exclude intracylinder baselines from the calculation.
    skip_deconvolution : bool
        Do not attempt to deconvolve the instrument transfer function.
    reference_declination : float, optional
        Declination at which to set the flux normalization if `skip_deconvolution` is True.
        A source transiting at this declination will have a peak value equal to its flux.
        If `None` (default), the zenith is used.
    save_dirty_beam : bool
        Create a `dirty_beam` dataset in the output container that contains
        the synthesized beam in the EW direction at each declination.
    window_type : {"none"|"uniform"|"hann"|"hanning"|"hamming"|"blackman"|
                   "nuttall"|"blackman_nuttall"|"blackman_harris"}
        Apply this type of window to the deconvolved m-mode transform to shape
        the synthesized beam in the EW direction.  Note that if this parameter
        is not provided or is set to "none", then a window will not be applied.
    window_size : float
        Determines the width of the window.  If window_size = 1.0,
        then the window will span all m's where we expect to have
        sensitivity based on the EW baseline distances.  Values smaller
        or larger than 1.0 will shrink or extend the window by the
        corresponding fractional amount.  Only relevant if the window_type
        parameter is provided.
    window_scaled : bool
        Use the same window for all frequencies in an attempt to produce
        a frequency independent synthesized beam in the EW direction.
        Only relevant if the window_type parameter is provided.
    """

    exclude_intracyl = config.Property(proptype=bool, default=False)
    skip_deconvolution = config.Property(proptype=bool, default=False)
    reference_declination = config.Property(proptype=float, default=None)
    save_dirty_beam = config.Property(proptype=bool, default=False)

    window_type = config.enum(
        [
            "none",
            "uniform",
            "hann",
            "hanning",
            "hamming",
            "blackman",
            "nuttall",
            "blackman_nuttall",
            "blackman_harris",
        ],
        default="none",
    )
    window_size = config.Property(proptype=float, default=1.0)
    window_scaled = config.Property(proptype=bool, default=False)

    def setup(self, manager: io.TelescopeConvertible = None):
        """Set the telescope instance if a manager object is given.

        The telescope instance is only needed if window_type is not "none"
        or if reference_declination is not None.

        Parameters
        ----------
        manager : manager.ProductManager, optional
            The telescope/manager used to extract the latitude and
            convert sin(za) to declination.
        """
        if manager is not None:
            self.telescope = io.get_telescope(manager)
        elif self.window_type != "none":
            raise RuntimeError("Must provide manager object if applying window.")
        else:
            self.telescope = None

    def process(
        self,
        hybrid_vis_m: containers.HybridVisMModes,
        hybrid_beam_m: containers.HybridVisMModes,
    ) -> containers.RingMap:
        """Generate a deconvolved ringmap using an input beam model.

        Parameters
        ----------
        hybrid_vis_m : containers.HybridVisMModes
            M-mode transform of hybrid beamformed visibilities.

        hybrid_beam_m : containers.HybridVisMModes
            M-mode transform of the beam after converting into
            hybrid beamformed visibilities.

        Returns
        -------
        ringmap
            The deconvolved ring map.
        """
        # Validate that the visibilites and beams match
        if not np.array_equal(hybrid_vis_m.freq, hybrid_beam_m.freq):
            raise ValueError("Frequencies do not match for beam and visibilities.")

        if not np.array_equal(
            hybrid_vis_m.index_map["el"], hybrid_beam_m.index_map["el"]
        ):
            raise ValueError("Elevations do not match for beam and visibilities.")

        if not np.array_equal(
            hybrid_vis_m.index_map["ew"], hybrid_beam_m.index_map["ew"]
        ):
            raise ValueError("EW baselines do not match for beam and visibilities.")

        if not np.array_equal(
            hybrid_vis_m.index_map["pol"], hybrid_beam_m.index_map["pol"]
        ):
            raise ValueError("Polarisations do not match for beam and visibilities.")

        if hybrid_vis_m.mmax > hybrid_beam_m.mmax:
            raise ValueError("Beam model must have higher m-max than the visibilities")

        # Distribute over frequency
        hybrid_vis_m.redistribute("freq")
        hybrid_beam_m.redistribute("freq")

        # Determine local frequencies
        nfreq = hybrid_vis_m.vis.local_shape[3]
        fstart = hybrid_vis_m.vis.local_offset[3]
        fstop = fstart + nfreq

        local_freq = hybrid_vis_m.freq[fstart:fstop]

        # Number of RA samples in the final output
        m = hybrid_vis_m.index_map["m"]
        mmax = hybrid_vis_m.mmax

        nra = 2 * mmax + int(hybrid_vis_m.oddra)

        # Create ring map, copying over axes/attrs
        rm = containers.RingMap(
            beam=1,
            ra=nra,
            axes_from=hybrid_vis_m,
            attrs_from=hybrid_vis_m,
            distributed=hybrid_vis_m.distributed,
            comm=hybrid_vis_m.comm,
        )
        if self.save_dirty_beam:
            rm.add_dataset("dirty_beam")

        rm.weight[:] = 0.0
        rm.redistribute("freq")

        # Add attributes describing the EW weighting scheme
        rm.attrs["exclude_intracyl"] = self.exclude_intracyl
        if hasattr(self, "weight_ew"):
            rm.attrs["weight_ew"] = self.weight_ew

        # Determine the window that will be applied to the m-mode transform
        if self.window_type != "none":
            window = self._get_window(hybrid_vis_m)

            # Expand window so it can broadcast against an array with a pol axis
            window = window[:, :, np.newaxis, :]
        else:
            window = np.ones(nfreq, dtype=np.float32)

        # Determine index closest to the reference declination
        if self.skip_deconvolution:
            el = rm.index_map["el"]
            if self.reference_declination is None:
                iref = np.argmin(np.abs(el))
                self.log.info("Normalizing the map to zenith.")
            else:
                dec = np.degrees(np.arcsin(el)) + self.telescope.latitude
                iref = np.argmin(np.abs(dec - self.reference_declination))
                self.log.info(f"Normalizing the map to Decl. = {dec[iref]:0.2f} deg.")

        # Dereference datasets
        hv = hybrid_vis_m.vis[:].view(np.ndarray)
        hw = hybrid_vis_m.weight[:].view(np.ndarray)

        # Dereference the beams, and trim to the set of m's present in the input data
        bv = hybrid_beam_m.vis[:].view(np.ndarray)[: (mmax + 1)]

        rmm = rm.map[:]
        rmw = rm.weight[:]
        if self.save_dirty_beam:
            rmb = rm.dirty_beam[:]

        # Loop over frequencies
        for lfi, freq in enumerate(local_freq):
            find = (slice(None),) * 3 + (lfi,)

            hvf = hv[find]
            bvf = bv[find]

            winf = window[lfi]

            # Make copy here because we will modify weights if exclude_intracyl is set
            inv_var = hw[find][..., np.newaxis].copy()

            # Get the EW weights using method defined by subclass
            weight = self._get_weight(inv_var) * (inv_var > 0.0)

            # Calculate the normalization
            sum_weight = (weight * np.abs(bvf) ** 2).sum(axis=(1, -2))

            # Get the regularisation term, exact prescription is defined by subclass
            if not self.skip_deconvolution:
                epsilon = self._get_regularisation(freq, m)
                C_inv = epsilon + sum_weight
            else:
                C_inv = 1.0

            # Solve for the sky m-modes
            map_m = (
                winf
                * (bvf.conj() * weight * hvf).sum(axis=(1, -2))
                * tools.invert_no_zero(C_inv)
            )

            # Calculate the dirty beam m-modes
            dirty_beam_m = winf * sum_weight * tools.invert_no_zero(C_inv)

            # Calculate the point source normalization (dirty beam at transit)
            norm = tools.invert_no_zero(dirty_beam_m.mean(axis=0))[:, np.newaxis, :]
            if self.skip_deconvolution:
                norm = norm[:, :, iref, np.newaxis]

            # Fill in the ringmap
            rmm[0, :, lfi] = (
                np.fft.irfft(map_m.transpose(1, 2, 0), axis=-1, n=nra).transpose(
                    0, 2, 1
                )
                * norm
            )

            # Fill in the dirty beam
            if self.save_dirty_beam:
                rmb[0, :, lfi] = (
                    np.fft.irfft(
                        dirty_beam_m.transpose(1, 2, 0), axis=-1, n=nra
                    ).transpose(0, 2, 1)
                    * norm
                )

            # Calculate the expected map noise by propagating the uncertainty on the m's
            # We use an unusual order of operations here to prevent floating point
            # overflow, which can occur as the north-south beam drops to zero at large
            # zenith angles.  This results in an otherwise unnecessary sqrt and several
            # multiplications.
            var = tools.invert_no_zero(inv_var)
            sigma = np.sqrt(np.sum((weight * np.abs(bvf)) ** 2 * var, axis=(1, -2)))

            sum_var_map_m = (
                0.5
                * np.sum(
                    (
                        sigma
                        * winf
                        * norm[np.newaxis, :, 0]
                        * tools.invert_no_zero((mmax + 1) * C_inv)
                    )
                    ** 2,
                    axis=0,
                )[:, np.newaxis, :]
            )

            rmw[:, lfi] = tools.invert_no_zero(sum_var_map_m)

        return rm

    def _get_window(self, hybrid_vis_m):
        """Return the window to be applied to the m-mode transform.

        Parameters
        ----------
        hybrid_vis_m : containers.HybridVisMModes
            The m-mode transform of the hybrid visiblities.
            Must be distributed over the frequency axis.

        Returns
        -------
        window : np.ndarray[nfreq, nm, nel]
            The window to be applied to the deconvolved m-mode transform.
            This will influence the shape of the synthesized beam in
            the EW direction.
        """
        msg = "independent" if self.window_scaled else "dependent"
        self.log.info(
            f"Applying a frequency {msg} {self.window_type} window "
            f"with a relative width of {self.window_size}."
        )

        # Extract the axes that we will need from the input container
        freq = hybrid_vis_m.freq
        m = hybrid_vis_m.index_map["m"]
        ew = hybrid_vis_m.index_map["ew"]
        el = hybrid_vis_m.index_map["el"]

        # If the window is frequency dependent, then we will only
        # need the frequencies local to this node.
        nlocal = hybrid_vis_m.vis.local_shape[3]

        if not self.window_scaled:
            fstart = hybrid_vis_m.vis.local_offset[3]
            freq = freq[fstart : fstart + nlocal]

        # Determine the minimum and maximum m for each frequency and declination
        dec = np.arcsin(el[np.newaxis, :]) + np.radians(self.telescope.latitude)
        lmbda = scipy.constants.c / (freq[:, np.newaxis] * 1e6)

        isort = np.argsort(np.abs(ew))
        ews = np.abs(ew)[isort]

        max_ew = ews[-1] + 0.5 * (ews[-1] - ews[-2])

        if self.exclude_intracyl:
            # If we are excluding intra-cylinder baselines, then the window should
            # smoothly transition to zero as m decreases towards the minimum m
            # measured by the shortest inter-cylinder baseline.
            min_ew = 0.5 * ews[ews > 0.0][0]
        else:
            min_ew = -max_ew

        center = 0.5 * (min_ew + max_ew)
        width = self.window_size * (max_ew - min_ew)

        ew_to_m = 2.0 * np.pi * np.abs(np.cos(dec)) / lmbda
        min_m = ew_to_m * (center - 0.5 * width)
        max_m = ew_to_m * (center + 0.5 * width)

        # If frequency indepenent, then we need to determine a single
        # minimum and maximum m that is valid for all frequencies.
        if self.window_scaled:
            min_m = np.max(min_m, axis=0, keepdims=True)
            max_m = np.min(max_m, axis=0, keepdims=True)

        # Loop over frequencies and elevations and calculate the window
        nfreq, nel = min_m.shape
        window = np.zeros((nfreq, m.size, nel), dtype=np.float32)

        for ff in range(nfreq):
            for ee in range(nel):
                mmin = min_m[ff, ee]
                mmax = max_m[ff, ee]

                in_range = np.flatnonzero((m >= mmin) & (m <= mmax))

                if in_range.size > 0:
                    x = (m[in_range] - mmin) / (mmax - mmin)

                    window[ff, in_range, ee] = tools.window_generalised(
                        x, window=self.window_type
                    )

        # If frequency independent, then repeat the same window
        # for every local frequency.
        if self.window_scaled:
            window = np.repeat(window, nlocal, axis=0)

        return window

    def _get_weight(self, inv_var):
        """Return the weight to be used when averaging over EW baselines.

        Any subclass must define this method in order to be a functional
        deconvolving ringmap maker.

        Parameters
        ----------
        inv_var : np.ndarray[nm, nmsign, npol, new, nel]
            The inverse variance of the noise in the m-mode transform
            of the hybrid visibilities.

        Returns
        -------
        weight :  np.ndarray[nm, nmsign, npol, new, nel] (or can broadcast against)
            The weight given to each EW baseline.
        """
        raise NotImplementedError(f"{self.__class__} must define a _get_weight method.")

    def _get_regularisation(self, freq, m):
        """Return the parameter used to regularize the deconvolution operation.

        Any subclass must define this method in order to be a functional
        deconvolving ringmap maker.

        Parameters
        ----------
        freq : float
            The frequency in MHz.
        m : np.ndarray[nm,]
            The m-modes.

        Returns
        -------
        epsilon : np.ndarray[nm, npol, nel] (or can broadcast against)
            The regularisation parameter that appears in the denominator of
            the deconvolution equation.
        """
        raise NotImplementedError(
            f"{self.__class__} must define a _get_regularisation method."
        )


class DeconvolveAnalyticalBeam(DeconvolveHybridMBase):
    """Base class for deconvolving the driftscan model of the beam (non-functional)."""

    telescope = None

    def setup(self, telescope: io.TelescopeConvertible):
        """Set the telescope object.

        Parameters
        ----------
        telescope
            The telescope object to use.
        """
        self.telescope = io.get_telescope(telescope)

    def process(
        self,
        hybrid_vis_m: containers.HybridVisMModes,
    ) -> containers.RingMap:
        """Generate a deconvolved ringmap using an analytic beam model.

        Parameters
        ----------
        hybrid_vis_m : containers.HybridVisMModes
            M-mode transform of hybrid beamformed visibilities.

        Returns
        -------
        ringmap
            The deconvolved ring map.
        """
        # Prepare the external beam m-modes and save to class attribute
        hybrid_beam_m = self._get_beam_mmodes(hybrid_vis_m)

        return super().process(hybrid_vis_m, hybrid_beam_m)

    def _get_beam_mmodes(
        self, hybrid_vis_m: containers.HybridVisMModes
    ) -> containers.HybridVisMModes:
        # NOTE: Coefficients taken from Mateus's fits, but adjust to fix the definition
        # of sigma, and be the widths for the "voltage" beam
        def sig_chime_X(freq, dec):
            """EW voltage beam widths (in sigma) for CHIME X pol."""
            return 14.87857614 / freq / np.cos(dec)

        def sig_chime_Y(freq, dec):
            """EW voltage beam widths (in sigma) for CHIME Y pol."""
            return 9.95746878 / freq / np.cos(dec)

        beam_width = {"X": sig_chime_X, "Y": sig_chime_Y}

        def A(phi, sigma):
            """A Gaussian like function on the circle."""
            return np.exp(-((2 * np.tan(phi / 2)) ** 2) / (2 * sigma**2))

        def B(phi, u, sigma):
            """Azimuthal beam transfer function."""
            return np.exp(2.0j * np.pi * u * np.sin(phi)) * A(phi, sigma)

        # Determine the RA axis from the maximum m-mode in the hybrid visibilities
        mmax = hybrid_vis_m.mmax
        nra = 2 * mmax + int(hybrid_vis_m.oddra)

        dec = np.arcsin(hybrid_vis_m.index_map["el"]) + np.radians(
            self.telescope.latitude
        )
        pol = hybrid_vis_m.index_map["pol"]

        # Determine RA axis for beam
        ra = np.linspace(0.0, 360.0, nra, endpoint=False)
        phi_arr = np.radians(ra)[np.newaxis, np.newaxis, np.newaxis, :]

        hybrid_beam_m = containers.empty_like(hybrid_vis_m)

        # Loop over all local frequencies and calculate the beam m-modes
        for lfi, fi in hybrid_vis_m.vis[:].enumerate(axis=3):
            freq = hybrid_vis_m.freq[fi]

            # Calculate the baseline distance in wavelengths
            wv = scipy.constants.c * 1e-6 / freq
            u = hybrid_vis_m.index_map["ew"] / wv

            # Calculate the projected baseline distance
            u_dec = u[:, np.newaxis] * np.cos(dec)[np.newaxis, :]
            u_arr = u_dec[np.newaxis, :, :, np.newaxis]

            # Construct an array containing the width of the beam for
            # each polarisation and declination
            sig = np.zeros((pol.size, dec.size), dtype=dec.dtype)
            for pi, (pa, pb) in enumerate(pol):
                # Get the effective beamwidth for the polarisation combination
                sig_a = beam_width[pa](freq, dec)
                sig_b = beam_width[pb](freq, dec)
                sig[pi] = sig_a * sig_b / (sig_a**2 + sig_b**2) ** 0.5

            sig_arr = sig[:, np.newaxis, :, np.newaxis]

            # Calculate the effective beam transfer function
            B_arr = B(phi_arr, u_arr, sig_arr)

            hybrid_beam_m.vis[:, :, :, fi] = transform._make_marray(
                B_arr.conj(), mmax=mmax
            )

        return hybrid_beam_m


class TikhonovRingMapMaker(DeconvolveHybridMBase):
    """Class for making maps using a Tikhonov regularisation scheme.

    Attributes
    ----------
    weight_ew : string
        How to weight the EW baselines? One of:
            'natural' - weight by the redundancy of the EW baselines.
            'uniform' - give each EW baseline uniform weight.
            'inverse_variance' - weight by the expected inverse noise variance
                                 saved to the `weight` dataset.

    inv_SN:
        Regularisation parameter.
    """

    weight_ew = config.enum(
        ["natural", "uniform", "inverse_variance"], default="natural"
    )
    inv_SN = config.Property(proptype=float, default=1e-6)

    def _get_weight(self, inv_var):
        if self.weight_ew == "inverse_variance":
            weight_ew = inv_var

        else:
            n_ew = inv_var.shape[-2]

            if self.weight_ew == "uniform":
                weight_ew = np.ones(n_ew)
            else:  # self.weight_ew == "natural"
                weight_ew = n_ew - np.arange(n_ew)

            expand = [None] * inv_var.ndim
            expand[-2] = slice(None)
            weight_ew = weight_ew[tuple(expand)]

        if self.exclude_intracyl:
            weight_ew[..., 0, :] = 0.0

        return weight_ew * tools.invert_no_zero(
            np.sum(weight_ew, axis=-2, keepdims=True)
        )

    def _get_regularisation(self, *args):
        return self.inv_SN


class WienerRingMapMaker(DeconvolveHybridMBase):
    r"""Class for map making using a Wiener regularisation scheme.

    Compared to TikhonovRingMapMaker, this task has a frequency and m-mode
    dependent regularisation parameter given by the ratio of the noise spectrum
    to the expected signal spectrum. The noise spectrum is obtained from the
    `weight` dataset and the signal spectrum is obtained from a power-law model
    for the extragalactic point source and diffuse galactic synchrotron emission
    whose parameters can be changed by the user.

    .. math::
        |V_{gal}| = a_{gal} \left(\frac{\nu}{\nu_{0}\right)^{\alpha_{gal}} m^{\beta_{gal}}
        |V_{psrc}| = a_{psrc} \left(\frac{\nu}{\nu_{0}\right)^{\alpha_{psrc}}
        S = |V_{gal}|^2 + |V_{psrc}|^2

    Attributes
    ----------
    gal_amp : float
        Prior for the amplitude of the m-mode transform of the
        diffuse galactic synchrotron emission.
    gal_alpha : float
        Prior for the power-law exponent describing the frequency dependence
        of the m-mode transform of the diffuse galactic synchrotron emission.
    gal_beta : float
        Prior for the power-law exponent describing the m dependence of
        of the m-mode transform of the diffuse galactic synchrotron emission.
    psrc_amp : float
        Prior for the amplitude of the m-mode transform of the
        extra-galactic point source emission.
    psrc_alpha : float
        Prior for the power-law exponent describing the frequency dependence
        of the m-mode transform of the extra-galactic point source emission.
    """

    gal_amp = config.Property(proptype=float, default=1.41)
    gal_alpha = config.Property(proptype=float, default=-1.75)
    gal_beta = config.Property(proptype=float, default=-0.75)

    psrc_amp = config.Property(proptype=float, default=0.045)
    psrc_alpha = config.Property(proptype=float, default=-1.0)

    pivot_freq = 600.0
    weight_ew = "inverse_variance"

    def _get_regularisation(self, freq, m, *args):
        gal = (
            self.gal_amp
            * (freq / self.pivot_freq) ** self.gal_alpha
            * np.where(m > 0.0, m, 1.0) ** self.gal_beta
        )
        psrc = self.psrc_amp * (freq / self.pivot_freq) ** self.psrc_alpha

        spectrum = gal**2 + psrc**2

        # Expand the array so that it can be broadcast against an
        # array of shape (nm, npol, nel)
        return tools.invert_no_zero(spectrum[:, np.newaxis, np.newaxis])

    def _get_weight(self, inv_var):
        weight_ew = inv_var
        if self.exclude_intracyl:
            weight_ew[..., 0, :] = 0.0

        return weight_ew


class TikhonovRingMapMakerAnalytical(DeconvolveAnalyticalBeam, TikhonovRingMapMaker):
    """Make a ringmap using Tikhonov deconvolution of an analytical beam model."""


class WienerRingMapMakerAnalytical(DeconvolveAnalyticalBeam, WienerRingMapMaker):
    """Make a ringmap using Wiener deconvolution of an analytical beam model."""


# Aliases to support old names
TikhonovRingMapMakerExternal = TikhonovRingMapMaker
WienerRingMapMakerExternal = WienerRingMapMaker


class RADependentWeights(task.SingleTask):
    """Re-establish an RA dependence to the `weight` dataset of a deconvolved ringmap.

    This RA dependence was lost in the round-trip m-mode transform.
    """

    def process(
        self, hybrid_vis: containers.HybridVisStream, ringmap: containers.RingMap
    ) -> containers.RingMap:
        """Scale the ringmap weights by the RA dependence of the hybrid visibility weights.

        Parameters
        ----------
        hybrid_vis : containers.HybridVisStream
            Hybrid beamformed visibilities before deconvolution.
            Used to measure the RA dependence of the `weight` dataset.
        ringmap : containers.RingMap
            RingMap after deconvolution.

        Returns
        -------
        ringmap : containers.RingMap
            The input ringmap container with the `weight` dataset scaled
            by an RA dependent factor determined from hybrid_vis.
        """
        # Determine how the EW baselines were averaged in the ringmap maker
        exclude_intracyl = ringmap.attrs.get("exclude_intracyl", None)
        weight_scheme = ringmap.attrs.get("weight_ew", None)

        if (exclude_intracyl is None) or (weight_scheme is None):
            msg = (
                "The ring map maker must save `weight_ew` and `exclude_intracyl` "
                "config parameters to the container attributes in order to "
                "reconstruct the RA dependence of the noise."
            )
            raise RuntimeError(msg)

        # Ensure containers are distributed over the same axis
        hybrid_vis.redistribute("freq")
        ringmap.redistribute("freq")

        # Extract the variance of the hybrid visibilities from the weight dataset
        var = tools.invert_no_zero(hybrid_vis.weight[:].local_array)

        # Calculate the time averaged variance
        var_time_avg = np.mean(var, axis=-1, keepdims=True)

        # Determine the weights that where used to average over the EW baselines
        if weight_scheme == "inverse_variance":
            weight_ew = tools.invert_no_zero(var_time_avg)

        else:
            n_ew = var.shape[-2]

            if weight_scheme == "uniform":
                weight_ew = np.ones(n_ew)
            else:  # weight_scheme == "natural"
                weight_ew = n_ew - np.arange(n_ew)

            expand = [None] * var.ndim
            expand[-2] = slice(None)
            weight_ew = weight_ew[tuple(expand)]

        if exclude_intracyl:
            weight_ew[..., 0, :] = 0.0

        # Use the baseline averaged variance divided by the baseline averaged,
        # time averaged variance as an approximatation for the RA dependence of
        # the noise variance in the deconvolved ringmap.  Note that this is
        # inverted in the equation below since we want to scale the weights.
        ra_dependence = np.sum(
            weight_ew**2 * var_time_avg, axis=-2
        ) * tools.invert_no_zero(np.sum(weight_ew**2 * var, axis=-2))

        # Scale the ringmap weights by the RA dependence
        ringmap.weight[:].local_array[:] *= ra_dependence[..., np.newaxis]

        return ringmap


class ReconstructVisWeight(transform.TelescopeStreamMixIn, task.SingleTask):
    """Compute visibility weights that match hybrid beamformed visibility weights.

    This is useful for generating noise realizations of hybrid beamformed
    visibilities that maintain proper correlation along the el axis.  It uses
    the setup method defined in TelescopeStreamMixIn to create the appropriate
    prod, stack, and reverse_stack axis from a telescope instance.
    """

    def process(self, hv):
        """Create a sidereal stream with weights that will reproduce the input weights.

        Parameters
        ----------
        hv : containers.HybridVisStream
            Hybrid beamformed visibilities with the desired weights.

        Returns
        -------
        ss : containers.SiderealStream
            A sidereal stream with visibilities set to zero and
            with weights proportional to the baseline redundancy
            and normalized to reproduce the input weights after
            beamforming in the north-south direction.
        """
        # Extract parameters needed to calculate the window
        # function in the y direction
        weight = hv.attrs["beamform_ns_weight"]
        if weight == "inverse_variance":
            raise ValueError("Weight scheme inverse_variance not supported.")

        include_auto = hv.attrs["beamform_ns_include_auto"]
        scaled = hv.attrs["beamform_ns_scaled"]
        freqmin = hv.attrs["beamform_ns_freqmin"]
        nsmax = hv.attrs["beamform_ns_nsmax"]

        wvmin = scipy.constants.c * 1e-6 / freqmin

        # Calculation the set of polarisations in the data, and which polarisation
        # index every entry corresponds to
        polpair = self.telescope.polarisation[self.telescope.uniquepairs].view("U2")
        polpair, pind = np.unique(polpair, return_inverse=True)

        # Determine the layout of the visibilities on the grid.
        xind, yind, min_xsep, min_ysep = find_grid_indices(self.telescope.baselines)

        baseline_flag = np.abs(yind * min_ysep) <= (nsmax + 0.5 * min_ysep)

        # Determine north-south grid
        ny = 2 * np.abs(yind).max() + 1
        nspos = np.fft.fftfreq(ny, d=(1.0 / (ny * min_ysep)))

        # Check that the input container has the same east-west grid
        vis_pos_x = np.arange(np.max(np.abs(xind)) + 1) * min_xsep

        ewpos = hv.index_map["ew"]
        nx = ewpos.size

        if not np.array_equal(vis_pos_x, ewpos):
            raise RuntimeError("Downselected ew axis not currently supported.")

        # Select the polarisations present in hvis
        pol = hv.index_map["pol"]
        npol = pol.size

        pol_lookup = {key: ind for ind, key in enumerate(hv.pol)}

        pol_remap = np.array([pol_lookup.get(pstr, -1) for pstr in polpair[pind]])
        pol_flag = pol_remap >= 0

        # Downselect the baselines to process
        flag = pol_flag & baseline_flag

        xind = xind[flag]
        yind = yind[flag]
        pind = pol_remap[flag]

        pconjmap = np.unique([pj + pi for pi, pj in pol], return_inverse=True)[1]

        # Create output container
        ss = containers.SiderealStream(
            axes_from=hv,
            attrs_from=hv,
            input=self.telescope.input_index,
            prod=self.bt_prod,
            stack=self.bt_stack,
            reverse_map_stack=self.bt_rev,
            distributed=hv.distributed,
            comm=hv.comm,
        )

        hv.redistribute("freq")
        ss.redistribute("freq")

        # Assume that the noise variance scales with
        # the redundancy of the baseline.
        input_flags = np.all(self.telescope.feedmask, axis=-1, keepdims=True)
        nbaseline = tools.calculate_redundancy(
            input_flags,
            ss.index_map["prod"],
            ss.reverse_map["stack"]["stack"][:],
            ss.vis.shape[1],
        )[:, 0]

        nbaseline_valid = nbaseline[flag]

        # Place the redundancy onto a regular grid
        nbaseline_grid = np.zeros((npol, nx, ny), dtype=float)
        nbaseline_grid[pind, xind, yind] = nbaseline_valid

        intra = np.flatnonzero(xind == 0)
        nbaseline_grid[pconjmap[pind[intra]], 0, -yind[intra]] = nbaseline_valid[intra]

        # Determine the wavelengths
        wavelength = scipy.constants.c * 1e-6 / ss.freq[ss.weight[:].local_bounds]

        # Initialize datasets in output container
        ss.vis[:] = 0.0

        wss = ss.weight[:].local_array

        # Assume that the weight in the sidereal stream
        # scales with redundancy
        wss[:] = np.where(flag, nbaseline, 0.0)[np.newaxis, :, np.newaxis]

        # Dereference datasets in input container
        whv = hv.weight[:].local_array

        # If natural weighting was used, then we can perform most of the
        # calculation now, since it won't change with frequency
        if weight == "natural":

            window = nbaseline_grid.copy()

            # Remove auto-correlations
            if not include_auto:
                window[:, 0, 0] = 0.0

            # Normalize by sum of weights
            norm = np.sum(window, axis=-1, keepdims=True)
            window *= tools.invert_no_zero(norm)

            # Conversion between weight in hybrid beamformed visibilities
            # to weight in visibilities
            noise_factor = np.sum(
                window**2 * tools.invert_no_zero(nbaseline_grid), axis=-1
            )

        # Loop over local frequencies
        for ff, wv in enumerate(wavelength):

            # If natural weighting was not used, then caclulate the
            # window function based on the frequency.
            if weight != "natural":

                vpos = nspos / wv
                vmax = nsmax / wvmin if scaled else nsmax / wv

                x = 0.5 * (vpos / vmax + 1)
                window = np.ones((npol, nx, ny), dtype=float)
                window *= tools.window_generalised(x, window=weight)

                # Remove auto-correlations
                if not include_auto:
                    window[:, 0, 0] = 0.0

                # Normalize by sum of weights
                norm = np.sum(window, axis=-1, keepdims=True)
                window *= tools.invert_no_zero(norm)

                # Conversion between weight in hybrid beamformed visibilities
                # to weight in visibilities
                noise_factor = np.sum(
                    window**2 * tools.invert_no_zero(nbaseline_grid), axis=-1
                )

            w0 = noise_factor[:, :, np.newaxis] * whv[:, ff, :, :]

            wss[ff][flag] *= w0[pind, xind, :]

        return ss


def find_basis(baselines):
    """Find the basis unit vectors of the grid of baselines.

    Parameters
    ----------
    baselines : np.ndarray[nbase, 2]
        X and Y displacements of the baselines.

    Returns
    -------
    xhat, yhat : np.ndarray[2]
        Unit vectors pointing in the mostly X and mostly Y directions of the grid.
    """
    # Find the shortest baseline, this should give one of the axes
    bl = np.sum(baselines**2, axis=1)
    bl[bl == 0] = 1e30
    ind = np.argmin(bl)

    # Determine the basis vectors and label them xhat and yhat
    e1 = baselines[ind]
    e2 = np.array([e1[1], -e1[0]])

    xh, yh = (e1, e2) if abs(e1[0]) > abs(e2[0]) else (e2, e1)

    xh = xh / np.dot(xh, xh) ** 0.5 * np.sign(xh[0])
    yh = yh / np.dot(yh, yh) ** 0.5 * np.sign(yh[1])

    return xh, yh


def find_grid_indices(baselines):
    """Find the indices of each baseline in the grid, and the spacing.

    Parameters
    ----------
    baselines : np.ndarray[nbase, 2]
        X and Y displacements of the baselines.

    Returns
    -------
    xind, yind : np.ndarray[nbase]
        Indices of the baselines in the grid.
    dx, dy : float
        Spacing of the grid in each direction.
    """

    def _get_inds(s):
        s_abs = np.abs(s)
        d = s_abs[s_abs > 1e-4].min()
        return np.rint(s / d).astype(np.int64), d

    xh, yh = find_basis(baselines)

    xind, dx = _get_inds(np.dot(baselines, xh))
    yind, dy = _get_inds(np.dot(baselines, yh))

    return xind, yind, dx, dy

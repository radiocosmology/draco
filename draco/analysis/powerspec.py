"""Power spectrum estimation from ringmap."""

import numpy as np
from astropy.cosmology import Planck15
from caput import config, mpiarray
from cora.util import units

from draco.analysis.delay import flatten_axes
from draco.core import containers, io, task
from draco.util import tools

f21 = units.nu21  # 21cm line frequency in MHz
C = units.c  # Speed of light in m/s
# We are setting cosmology to Planck2015 result
# one can use Planck2018 too
cosmo = Planck15

# Lat/Lon of CHIME
_LAT_LON = {
    "chime": [49.3207125, -119.623670],
}


class TransformJyPerBeamToKelvin(task.SingleTask):
    """Transform the ringmap in unit Jy/beam to Kelvin unit.

    This estimates the PSF solid angle (in sr unit) using the maximum baseline
    in the telescope class and use Rayleigh-Jeans factor to convert the
    ringmap in Jy/beam unit to Kelvin unit.

    Attributes
    ----------
    convert_Jy_per_beam_to_K : bool
        If True, apply a Jansky per beam to Kelvin conversion factor. If False apply a Kelvin to
        Jansky per beam conversion.
    ncyl : int
      number of cylinders to include in the maximum baseline estimate
      Note that, this should be equal to the numbers used to make the actual map.
      Default is 3, i.e, the map is made with all the east-west baselines.
      ncyl = 0 means that map is made with intracylinder baselines only.
    """

    ncyl = config.Property(proptype=int, default=3)

    def setup(self, telescope):
        """Set the telescope needed to obtain baselines.

        Parameters
        ----------
        telescope : TransitTelescope
            The telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

        # Get the maximum baseline from the telescope class
        self.bl_max = self._get_max_baseline()

    def process(self, rm):
        """Apply the Jy per beam to Kelvin conversion to the ringmap.

        Parameters
        ----------
        rm : containers.RingMap
            Data for which to estimate the power spectrum.

        Returns
        -------
        delay spectrum : containers.DelayTransform
        """
        rm.redistribute("freq")

        if not isinstance(rm, containers.RingMap):
            raise ValueError(
                f"Input container must be instance of RingMap (received {rm.__class__})"
            )

        # Get the local frequencies in the ringmap
        data_axes = list(rm.map.attrs["axis"])
        ax_freq = data_axes.index("freq")
        sfreq = rm.map.local_offset[ax_freq]
        efreq = sfreq + rm.map.local_shape[ax_freq]
        local_freq = rm.freq[sfreq:efreq]

        # Estimate the conversion factor
        factor = jy_per_beam_to_kelvin(local_freq, self.bl_max)

        # Genearate an output container
        out_map = containers.empty_like(rm)
        out_map.map[:] = 0.0
        out_map.weight[:] = 0.0

        # store the map after applying the conversion factor
        out_map.map[:] = (
            rm.map[:].local_array
            * factor[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        )
        out_map.weight[:] = (
            rm.weight[:].local_array
            * factor[np.newaxis, :, np.newaxis, np.newaxis] ** 2
        )

        return out_map

    def _get_max_baseline(self):
        from draco.analysis.ringmapmaker import find_grid_indices

        prod = self.telescope.prodstack
        baselines = (
            self.telescope.feedpositions[prod["input_a"], :]
            - self.telescope.feedpositions[prod["input_b"], :]
        )
        xind, yind, dx, dy = find_grid_indices(baselines)
        baslines = baselines[xind <= self.ncyl]
        bl = np.sqrt(np.sum(baslines**2, axis=-1))
        return bl.max()


class DelayTransformMapFFT(task.SingleTask):
    """Transform the ringmap from frequency to delay domain.

    This transforms the ringmap from frequency to delay domain by doing simple FFT.
    This will only work when the map is foreground filtered and close to noise.
    Otherwise  FFT along frequency axis will leak foreground power to high delays
    due to the presence of RFI masked missing data along frequency.

    The delay spectrum  output is indexed by a `baseline` axis. This
    axis is the composite axis of all the axes in the container except the frequency
    axis and the ra axis. These constituent axes are included in the index map,
    and their order is given by the `baseline_axes` attribute.

    Attributes
    ----------
    apply_pixel_mask : bool, optional
        If true, apply the pixel mask to the data, which is stored
        in the weight dataset. Default: False.
    apply_window : bool, optional
        Whether to apply apodisation to frequency axis. Default: True.
    window : window available in :func:`draco.util.tools.window_generalised()`, optional
        Apodisation to perform on frequency axis. Default: 'nuttall'.
    """

    apply_pixel_mask = config.Property(proptype=bool, default=False)
    apply_window = config.Property(proptype=bool, default=True)
    window = config.enum(
        [
            "uniform",
            "hann",
            "hanning",
            "hamming",
            "blackman",
            "nuttall",
            "blackman_nuttall",
            "blackman_harris",
        ],
        default="nuttall",
    )

    def process(self, rm):
        """Estimate the delay spectrum of the ringmap.

        Parameters
        ----------
        rm : containers.RingMap
            Data for which to estimate the power spectrum.

        Returns
        -------
        delay spectrum : containers.DelayTransform
        """
        rm.redistribute("freq")

        if not isinstance(rm, containers.RingMap):
            raise ValueError(
                f"Input container must be instance of RingMap (received {rm.__class__})"
            )

        freq_spacing = np.abs(np.diff(rm.freq[:])).mean()
        ndelay = len(rm.freq)

        # Compute delays in micro-sec
        self.delays = np.fft.fftshift(np.fft.fftfreq(ndelay, d=freq_spacing))

        # Get relevant views of data and weights, and create output container.
        # Find the relevant axis positions
        data_view, bl_axes = flatten_axes(rm.map, ["ra", "freq"])
        weight_view, _ = flatten_axes(rm.weight, ["ra", "freq"], match_dset=rm.map)

        # baseline axis
        bl = np.prod([len(rm.index_map[ax]) for ax in bl_axes])

        # Initialise the spectrum container
        delay_spectrum = containers.DelayTransform(
            baseline=bl,
            sample=rm.index_map["ra"],
            delay=self.delays,
            attrs_from=rm,
        )

        delay_spectrum.redistribute("baseline")
        delay_spectrum.spectrum[:] = 0.0

        # Copy the index maps for all the flattened axes into the output container, and
        # write out their order into an attribute so we can reconstruct this easily
        # when loading in the spectrum
        for ax in bl_axes:
            delay_spectrum.create_index_map(ax, rm.index_map[ax])
        delay_spectrum.attrs["baseline_axes"] = bl_axes

        # Save the frequency axis of the input data as an attribute in the output
        # container
        delay_spectrum.attrs["freq"] = rm.freq

        # Do delay transform
        for lbi, bi in delay_spectrum.spectrum[:].enumerate(axis=0):
            self.log.debug(
                f"Estimating the delay spectrum of each baseline {bi}/{bl} by FFT "
            )

            # Get the local selections
            data = data_view.local_array[lbi]
            weight = weight_view.local_array[lbi]

            # Apply the pixel mask stored in weight to the data
            # This is needed before taking FFT for spatial pixel mask
            if self.apply_pixel_mask:
                pixel_mask = np.where(weight > 0.0, 1.0, weight)
                data *= pixel_mask

            # Do FFT along freq
            if self.apply_window:
                fsel = np.arange(ndelay)
                x = fsel / ndelay
                w = tools.window_generalised(x, window=self.window)

                # Estimate the equivalent noise bandwidth for the tapering window
                # and store it as an attribute
                NEB_freq = noise_equivalent_bandwidth(w)
                delay_spectrum.attrs["window_los"] = self.window
                delay_spectrum.attrs["effective_bandwidth"] = NEB_freq

                # Now apply the tapering function to the data and
                # take iFFT
                w = w[np.newaxis, :]
                yspec = np.fft.fftshift(np.fft.ifft(data * w, axis=-1), axes=-1)

            else:
                yspec = np.fft.fftshift(np.fft.ifft(data, axis=-1), axes=-1)
                delay_spectrum.attrs["window_los"] = "None"
                delay_spectrum.attrs["effective_bandwidth"] = 1.0

            delay_spectrum.spectrum[bi] = yspec

        return delay_spectrum


class SpatialTransformDelayMap(task.SingleTask):
    """Spatial transform the delay map from (RA,DEC) to (u,v) domain.

    This transforms the delay data cube to spatial (u,v) domain
    by taking a 2D FFT along (RA,DEC) axes.

    Attributes
    ----------
    apply_spatial_window : bool, optional
        Whether to apply apodisation to RA and Dec axis before taking
        spatial FFT. Default: True.
    spatial_window : window available in :func:`draco.util.tools.window_generalised()`, optional
        Apodisation to perform along spatial axes. Default: 'Tukey-0.5'.
        Here "tukey-0.5" means 0.5 is the fraction of the full window that
        will be tapered.
    ew_min : float
     Minimum east-west baseline in meter.
    ew_max : float
      Maximum east-west baseline in meter.
    ns_bl : float
      basline along north-south direction to include.
    """

    apply_spatial_window = config.Property(proptype=bool, default=True)
    spatial_window = config.enum(
        [
            "uniform",
            "hann",
            "hanning",
            "hamming",
            "blackman",
            "nuttall",
            "blackman_nuttall",
            "blackman_harris",
            "tukey-0.5",
        ],
        default="tukey-0.5",
    )
    ew_min = config.Property(proptype=float, default=9.0)
    ew_max = config.Property(proptype=float, default=50.0)
    ns_bl = config.Property(proptype=float, default=60.0)

    def process(self, ds):
        """Estimate the spatial transform of the delay map.

        Parameters
        ----------
        ds : containers.DelayTransform
          The delay map, whose spatial transform will be estimated.

        Returns
        -------
        spatial cube : containers.SpatialDelayCube
           The data cube in (delay,u,v) domain.
        """
        if not isinstance(ds, containers.DelayTransform):
            raise ValueError(
                f"Input container must be instance of DelayTransform (received {ds.__class__})"
            )

        ds.redistribute("delay")

        # Extract required data axes
        delay = ds.index_map["delay"]  # micro-sec
        el = ds.index_map["el"]
        pol = ds.index_map["pol"]
        ra = ds.index_map["sample"]  # deg
        dec = sza2dec(el)  # deg
        freq = ds.attrs["freq"]  # MHz
        wl = C / (freq * 1e6)  # wavelength in meter

        # Unpack the baseline axis of the delay spectrum
        # and reshape it as (pol,delay,ra,el)
        axes = list(ds.attrs["baseline_axes"])
        shp = tuple([ds.index_map[ax].size for ax in axes])
        data_view = ds.spectrum[:].local_array.reshape(*shp, ra.size, -1)[0, :]
        data_view = np.swapaxes(data_view, 1, 3)
        data_view = mpiarray.MPIArray.wrap(data_view, axis=1, comm=ds.comm)
        # redistribute over delay axis
        data_view = data_view.redistribute(axis=1)

        # Estimate the Fourier modes
        nu_c = freq[int(freq.size / 2.0)]  # central freq of the band
        redshift = f21 / nu_c - 1  # redshift at the center of the band
        kx, ky, u, v, k_parallel = get_fourier_modes(ra, dec, delay * 1e-6, redshift)

        # Estimate the uv-mask
        uv_mask = spatial_mask(
            kx, ky, self.ew_min, self.ew_max, self.ns_bl, wl.min(), wl.max(), redshift
        )

        # Estimate the volume of the cube
        vol_cube = vol_normalization(ra, dec, freq, redshift)

        # Initialise the spatial transformed data cube container
        vis_cube = containers.SpatialDelayCube(
            pol=pol, delay=delay, kx=kx, ky=ky, attrs_from=ds
        )
        vis_cube.redistribute("delay")
        vis_cube.data_tau_uv[:] = 0.0

        # Save the central freq of the band and the
        # corresponding redshift and the cube volume in the attrs
        vis_cube.attrs["freq_center"] = nu_c
        vis_cube.attrs["redshift"] = redshift
        vis_cube.attrs["volume"] = vol_cube

        # Create index map for (u,v) coordinates and k_parallel
        fourier_axes = ["u", "v", "k_parallel", "uv_mask"]
        for ff in fourier_axes:
            vis_cube.create_index_map(ff, locals()[ff])

        # Save the spatial window as an attribute
        if self.apply_spatial_window:
            vis_cube.attrs["window_spatial"] = self.spatial_window
        else:
            vis_cube.attrs["window_spatial"] = "None"

        # Spatial transform to Fourier domain
        # loop over pol
        for pp, psr in enumerate(pol):
            self.log.debug(f"Estimating spatial FFT for pol: {psr}")
            # loop over delays
            for lde, de in vis_cube.data_tau_uv[:].enumerate(axis=1):
                slc = (pp, lde, slice(None), slice(None))
                data = np.ascontiguousarray(data_view.local_array[slc])
                data_uv, NEB_ra, NEB_dec = image_to_uv(
                    data,
                    ra=ra,
                    dec=dec,
                    window=self.spatial_window if self.apply_spatial_window else None,
                )
                vis_cube.data_tau_uv[pp, de] = data_uv

        # save the equivalent noise bandwidth for the
        # spatial tapering window as an attribute
        vis_cube.attrs["effective_ra"] = NEB_ra
        vis_cube.attrs["effective_dec"] = NEB_dec

        return vis_cube


class CrossPowerSpectrum3D(task.SingleTask):
    """Estimate the 3D cross power spectrum of two data cubes .

    This estimates the 3D cross power spectrum of two data cubes by taking
    real part of correlation between two complex data cubes and normalize that
    by the volume of the data cube in Mpc^3. The unit of the output power spectrum
    is K^2Mpc^3.

    """

    def process(self, data_1, data_2):
        """Estimate the cross power spectrum of two data cubes.

        Parameters
        ----------
        data_1 : containers.SpatialDelayCube
          The 1st data cube in fourier domain.
        data_2 : containers.SpatialDelayCube
          The 2nd data cube in fourier domain.

        Returns
        -------
        cross_ps : containers.Powerspec3D
           The 3D cross power spectum.
        """
        # Validate the shapes of two data cubes match
        if data_1.data_tau_uv.shape != data_2.data_tau_uv.shape:
            raise ValueError(
                f"Size of data_1 ({data_1.shape}) must match "
                f"data_2 ({data_2.shape})"
            )

        # Validate the types are the same
        if type(data_1) is not type(data_2):
            raise TypeError(
                f"type(data_1) (={type(data_1)}) must match "
                f"type(data_2) (={type(data_2)})"
            )

        data_1.redistribute("delay")
        data_2.redistribute("delay")

        # Extract required data axes
        delay = data_1.index_map["delay"]  # micro-sec
        kx = data_1.index_map["kx"]  # Mpc^-1
        ky = data_1.index_map["ky"]  # Mpc^-1
        pol = data_1.index_map["pol"]

        # Dereference the required datasets and
        vis_cube_1 = data_1.data_tau_uv[:].local_array
        vis_cube_2 = data_2.data_tau_uv[:].local_array

        # Compute power spectrum normalization factor
        # this is the survey volume, corrected  for the
        # tapering window function used for FFT
        volume_cube = data_1.attrs["volume"]
        NEB_freq = data_1.attrs["effective_bandwidth"]
        NEB_ra = data_1.attrs["effective_ra"]
        NEB_dec = data_1.attrs["effective_dec"]
        NEB = 1 / (NEB_freq * NEB_ra * NEB_dec)
        ps_norm = volume_cube * NEB

        # Initialise the 3D power spectrum container
        ps_cube = containers.Powerspec3D(
            pol=np.array(pol), delay=delay, kx=kx, ky=ky, attrs_from=data_1
        )
        ps_cube.redistribute("delay")
        ps_cube.ps3D[:] = 0.0

        # Save the power spectrum normalization factor in the attrs
        ps_cube.attrs["ps_norm"] = ps_norm

        # Create index map for (u,v) coordinates,k_parallel axis
        # and uv-mask
        fourier_axes = ["u", "v", "k_parallel", "uv_mask"]
        for ax in fourier_axes:
            ps_cube.create_index_map(ax, data_1.index_map[ax])

        # Estimate the cross power spectrum
        for pp, psr in enumerate(pol):
            self.log.debug(f"Estimating power spectrum for pol: {psr}")
            pol_id = list(pol).index(psr)

            for lde, de in ps_cube.ps3D[:].enumerate(axis=1):
                slc = (pol_id, lde, slice(None), slice(None))
                cube_1 = np.ascontiguousarray(vis_cube_1[slc])
                cube_2 = np.ascontiguousarray(vis_cube_2[slc])
                ps_cube.ps3D[pp, de] = get_3D_ps(
                    cube_1, cube_2, vol_norm_factor=ps_norm
                )

        return ps_cube


class AutoPowerSpectrum3D(CrossPowerSpectrum3D):
    """Estimate the 3D auto power spectrum of a data cube."""

    def process(self, data):
        """Estimate auto power spectrum.

        Parameters
        ----------
        data : containers.SpatialDelayCube
           The  data cube in fourier domain.

        Returns
        -------
        auto_ps : containers.Powerspec3D
           The 3D auto power spectum.
        """
        return super().process(data, data)


class CylindricalPowerSpectrum2D(task.SingleTask):
    """Estimate the cylindrically averaged 2D power spectrum.

    This estimates the cylindrically averaged 2D power spectrum from a
    3D power spectrum cube.

    Attributes
    ----------
    bl_min : float
       The minimum baseline length in meter to include in power spectrum binning. Default: 14.0m
    bl_max : float
       The minimum baseline length in meter to include in power spectrum binning. Default: 60.0m
    Nbins_2D : int
       The number of bins in 2D cylindrical binning. Default: 35
    logbins_2D : bool, optional
        If True, use logarithmic binning in cylindrical averaging. Default: False
    delay_cut : float
        Throw away the delay modes below this cutoff during spherical averaging, unit sec.
        This is same for both polarization. Default: 300.0e-9
    """

    bl_min = config.Property(proptype=float, default=14.0)
    bl_max = config.Property(proptype=float, default=60.0)
    Nbins_2D = config.Property(proptype=int, default=35)
    logbins_2D = config.Property(proptype=bool, default=False)
    delay_cut = config.Property(proptype=float, default=300.0e-9)

    def setup(self, weight=None):
        """Set the weight to use as the inverse variance weight.

        Parameters
        ----------
        weight : `containers.Powerspec3D`
            weight power spectrum estimated from many
            noise simulation to be used as inverse variance weight.
            Note this is variance, we need to take inverse of this
            to have inverse variance weight.
        """
        if weight is not None:
            self.weight = weight
        else:
            self.weight = None

    def process(self, ps):
        """Estimate the cylindrically averaged power spectrum.

        Parameters
        ----------
        ps : containers.Powerspec3D
          The 3D power spectrum cube.

        Returns
        -------
        cross_ps : containers.Powerspec2D
           The 2D power spectum.
        """
        if not isinstance(ps, containers.Powerspec3D):
            raise ValueError(
                f"Input container must be instance of Powerspec3D (received {ps.__class__})"
            )

        ps.redistribute("delay")
        
        # Extract required data axes
        pol = ps.index_map["pol"]
        kpar = ps.index_map["k_parallel"]
        u = ps.index_map["u"]
        v = ps.index_map["v"]
        uv_mask = ps.index_map["uv_mask"]
        redshift = ps.attrs["redshift"]
        nu_c = ps.attrs["freq_center"]
        wl = C / (nu_c * 1e6)  # m

        # find out the kperp bins
        u_min_lambda = self.bl_min / wl
        u_max_lambda = self.bl_max / wl
        kperp_min = u_to_kperp(u_min_lambda, redshift)
        kperp_max = u_to_kperp(u_max_lambda, redshift)

        if self.logbins_2D:
            kperp = np.logspace(np.log10(kperp_min), np.log10(kperp_max), self.Nbins_2D)
        else:
            kperp = np.linspace(kperp_min, kperp_max, self.Nbins_2D)

        # Take the center of the kperp bins
        kperp_cent = 0.5 * (kperp[1:] + kperp[:-1])

        # Dereference the required datasets
        ps_3D = ps.ps3D[:].local_array

        if self.weight is None:
            weight = np.ones_like(ps_3D)
        else:
            # input weight is variance and we are taking the
            # inverse to have a inverse variance weight
            self.weight.redistribute("delay")
            weight = tools.invert_no_zero(self.weight.ps3D[:].local_array)

        # Define the 2D power spectrum container
        pspec_2D = containers.Powerspec2D(
            pol=pol, kpar=kpar, kperp=kperp_cent, attrs_from=ps
        )
        pspec_2D.redistribute("kpar")
        pspec_2D.ps2D[:] = 0.0

        # save the delay cut value in attrs
        pspec_2D.attrs["delay_cut"] = self.delay_cut

        # loop over polarization
        for pp, psr in enumerate(pol):
            self.log.debug(f"Estimating 2D power spectrum for pol: {psr}")

            # loop over k_parallel axis
            for lde, de in pspec_2D.ps2D[:].enumerate(axis=1):
                slc = (pp, lde, slice(None), slice(None))
                data = np.ascontiguousarray(ps_3D[slc])
                W = np.ascontiguousarray(weight[slc])

                # keep the non-zero visibilities between minimum and maximum baseline
                # for each delay channel and take the flatten array; i.e, the output is (nvis),
                # where nvis is number of vis between two limit

                ps3D_flat, uu, vv = reshape_data_cube(
                    data, u, v, u_min_lambda, u_max_lambda
                )

                # do the same for uv-spatial mask and weight
                # and apply the mask before passing to 2D ps estimate
                mask_flat, _, _ = reshape_data_cube(
                    uv_mask, u, v, u_min_lambda, u_max_lambda
                )

                weight_flat, _, _ = reshape_data_cube(
                    W, u, v, u_min_lambda, u_max_lambda
                )

                # Cylindrical 2D power spectrum
                pspec_2D.ps2D[pp, de], pspec_2D.ps2D_weight[pp, de] = get_2d_ps(
                    ps3D_flat[mask_flat],
                    weight=weight_flat[mask_flat]
                    ** 2,  # square the weight for inverse variance weighted avg of power spectrum
                    kperp_bins=kperp,
                    uu=uu[mask_flat],
                    vv=vv[mask_flat],
                    redshift=redshift,
                )

        # Generate the signal window mask corresponding to the delay cut
        # Note that, we are not applying this mask to the 2D power spectrum
        # Instead, we save this as a dataset in the container, which can
        # be applied during plotting
        pspec_2D.redistribute("pol")
        pspec_2D.signal_mask[:] = True

        if self.delay_cut > 0.0:
            kpar_lim = delays_to_kpara(self.delay_cut, redshift)
            ibins = np.where((kpar > -kpar_lim) & (kpar < kpar_lim))
            for ii, jj in enumerate(pol):
                pspec_2D.signal_mask[ii, ibins, :] = False

        return pspec_2D


class SphericalPowerSpectrum2Dto1D(task.SingleTask):
    """Estimate the spherically averaged 1D power spectrum.

    This estimates the spherically averaged 1D power spectrum from a
    2D cylindrical power spectrum.

    Attributes
    ----------
    Nbins_3D : int
       The number of bins in 3D spherical binning. Default: 9
    logbins_3D : bool, optional
        If True, use logarithmic binning in cylindrical averaging. Default: False

    """

    Nbins_3D = config.Property(proptype=int, default=9)
    logbins_3D = config.Property(proptype=bool, default=True)

    def process(self, ps2D):
        """Estimate the spherically averaged power spectrum.

        Parameters
        ----------
        ps2D : containers.Powerspec2D
          The 2D cylindrically averaged power spectrum.

        Returns
        -------
        ps1D : containers.Powerspec1D
           The 1D power spectum.
        """
        if not isinstance(ps2D, containers.Powerspec2D):
            raise ValueError(
                f"Input container must be instance of Powerspec2D (received {ps2D.__class__})"
            )
        ps2D.redistribute("pol")

        # Extract required data axes
        pol = ps2D.pol
        kpar = ps2D.kpar
        kperp = ps2D.kperp

        # Dereference the required datasets
        ps_2D = ps2D.ps2D[:].local_array
        mask_2D = ps2D.signal_mask[:].local_array
        weight_2D = ps2D.ps2D_weight[:].local_array

        # Define the 1D power spectrum container
        pspec_1D = containers.Powerspec1D(pol=pol, k=self.Nbins_3D - 1, attrs_from=ps2D)
        pspec_1D.redistribute("pol")
        pspec_1D.ps1D[:] = 0.0

        # loop over polarization
        for pp, psr in pspec_1D.ps1D[:].enumerate(axis=0):
            self.log.debug(f"Estimating 1D power spectrum for pol: {pol[psr]}")

            (
                pspec_1D.k1D[psr],
                pspec_1D.ps1D[psr],
                pspec_1D.ps1D_error[psr],
                pspec_1D.ps1D_var[psr],
            ) = get_1d_ps(
                ps_2D[pp],
                kperp,
                kpar,
                signal_window=mask_2D[pp],
                Nbins_3D=self.Nbins_3D,
                weight_cube=weight_2D[pp],
                logbins_3D=self.logbins_3D,
            )

        return pspec_1D


class SphericalPowerSpectrum3Dto1D(task.SingleTask):
    """Estimate the spherically averaged 1D power spectrum.

    This estimates the spherically averaged 1D power spectrum from a
    3D  power spectrum cube of shape [npol,kpara,kx,ky].
    Note: this task is for check consistency. The 1D ps estimated
    from 3D ps-cube or 2D ps-cube should match. So, one can just
    estimate 1D ps from either 3D or 2D cube.

    Attributes
    ----------
    bl_min : float
       The minimum baseline length in meter to include in power spectrum binning. Default: 14.0
    bl_max : float
       The minimum baseline length in meter to include in power spectrum binning. Default: 60.0
    Nbins_3D : int
       The number of bins in 3D spherical binning. Default: 9
    logbins_3D : bool, optional
        If True, use logarithmic binning in cylindrical averaging. Default: False
    delay_cut : float
        Throw away the delay modes below this cutoff during spherical averaging, unit sec.
        This is same for both polarization. Default: 300.0e-9
    """

    bl_min = config.Property(proptype=float, default=14.0)
    bl_max = config.Property(proptype=float, default=60.0)
    Nbins_3D = config.Property(proptype=int, default=9)
    logbins_3D = config.Property(proptype=bool, default=True)
    delay_cut = config.Property(proptype=float, default=300.0e-9)

    def setup(self, weight=None):
        """Set the weight to use as the inverse variance weight.

        Parameters
        ----------
        weight : `containers.Powerspec3D`
            weight power spectrum estimated from many
            noise simulation to be used as inverse variance weight.
            Note this is variance, we need to take inverse of this
            to have inverse variance weight.
        """
        if weight is not None:
            self.weight = weight
        else:
            self.weight = None

    def process(self, ps):
        """Estimate the spherically averaged power spectrum.

        Parameters
        ----------
        ps : containers.Powerspec3D
          The 3D power spectrum cube.

        Returns
        -------
        ps1D : containers.Powerspec1D
           The 1D power spectum.
        """
        if not isinstance(ps, containers.Powerspec3D):
            raise ValueError(
                f"Input container must be instance of Powerspec2D (received {ps.__class__})"
            )
        ps.redistribute("pol")
        
        # Extract required data axes
        pol = ps.index_map["pol"]
        kpar = ps.index_map["k_parallel"]
        u = ps.index_map["u"]
        v = ps.index_map["v"]
        uv_mask = ps.index_map["uv_mask"]
        redshift = ps.attrs["redshift"]
        nu_c = ps.attrs["freq_center"]
        wl = C / (nu_c * 1e6)  # m

        #  find out the kperp bins
        u_min_lambda = self.bl_min / wl
        u_max_lambda = self.bl_max / wl

        # Dereference the required datasets
        ps_3D = ps.ps3D[:].local_array

        if self.weight is None:
            weight = np.ones_like(ps_3D)
        else:
            # input weight is variance and we are taking the
            # inverse to have a inverse variance weight
            self.weight.redistribute("pol")
            weight = tools.invert_no_zero(self.weight.ps3D[:].local_array)

        # Define the 1D power spectrum container
        pspec_1D = containers.Powerspec1D(pol=pol, k=self.Nbins_3D - 1, attrs_from=ps)
        pspec_1D.redistribute("pol")
        pspec_1D.ps1D[:] = 0.0

        # loop over polarization
        for pp, psr in pspec_1D.ps1D[:].enumerate(axis=0):
            self.log.debug(f"Estimating 1D power spectrum for pol: {pol[psr]}")

            ps3D_flat = []
            weight_flat = []

            # loop over k_parallel axis
            for lde, de in enumerate(kpar):
                slc = (pp, lde, slice(None), slice(None))
                data = np.ascontiguousarray(ps_3D[slc])
                W = np.ascontiguousarray(weight[slc])

                # keep the non-zero visibilities between minimum and maximum baseline
                # for each delay channel and take the flatten array; i.e, the output is (nvis),
                # where nvis is number of vis between two limit

                ps_flat, uu_flat, vv_flat = reshape_data_cube(
                    data, u, v, u_min_lambda, u_max_lambda
                )
                # do the same for uv-spatial mask and weight
                # and apply the mask before passing to 2D ps estimate
                m_flat, _, _ = reshape_data_cube(
                    uv_mask, u, v, u_min_lambda, u_max_lambda
                )

                w_flat, _, _ = reshape_data_cube(W, u, v, u_min_lambda, u_max_lambda)

                ps3D_flat.append(ps_flat[m_flat])
                weight_flat.append(w_flat[m_flat])

            # Apply the mask here to the [u,v]
            uu_flat = uu_flat[m_flat]
            vv_flat = vv_flat[m_flat]

            # Reshape the array of shape [ndelay,nvis]
            ps3D_flat = np.array(ps3D_flat).reshape(kpar.size, -1)
            weight_flat = np.array(weight_flat).reshape(kpar.size, -1)

            # convert to Fourier modes and estimate kperp
            ku = u_to_kperp(uu_flat, redshift)  # convert the u-array to fourier modes
            kv = u_to_kperp(vv_flat, redshift)  # convert the v-array to fourier modes
            kperp = np.sqrt(ku**2 + kv**2)

            # Generate the signal window mask corresponding to the delay cut
            signal_mask = np.ones_like(ps3D_flat, dtype=bool)

            if self.delay_cut > 0.0:
                kpar_lim = delays_to_kpara(self.delay_cut, redshift)
                ibins = np.where((kpar > -kpar_lim) & (kpar < kpar_lim))
                signal_mask[ibins, :] = False

            # Estimate the 1D power spectrum
            (
                pspec_1D.k1D[pp],
                pspec_1D.ps1D[pp],
                pspec_1D.ps1D_error[pp],
                pspec_1D.ps1D_var[pp],
            ) = get_1d_ps(
                ps3D_flat,
                kperp,
                kpar,
                signal_window=signal_mask,
                Nbins_3D=self.Nbins_3D,
                weight_cube=weight_flat
                ** 2,  # square the weight for inverse variance weighted avg of power spectrum
                logbins_3D=self.logbins_3D,
            )
        return pspec_1D


def f2z(freq):
    """Convert frequency to redshift for the 21 cm line.

    Parameters
    ----------
    freq : float
      frequency in MHz

    Returns
    -------
    redshift: float
    """
    return f21 / freq - 1


def z2f(z):
    """Convert redshift to frequency for the 21 cm line.

    Parameters
    ----------
    z : float
     redshift

    Returns
    -------
    frequency: float
       frequency in MHz
    """
    return f21 / (z + 1)


def dRperp_dtheta(z):
    """Conversion factor from angular size (radian) to transverse comoving distance (Mpc) at a specific redshift: [Mpc / radians].

    Parameters
    ----------
    z : float
     redshift

    Returns
    -------
    comoving transverse distance: float. unit in Mpc
    """
    return cosmo.comoving_transverse_distance(z).value


def dRpara_df(z):
    """Conversion from frequency bandwidth to radial comoving distance at a specific redshift: [Mpc / Hz].

    Parameters
    ----------
    z : float
     redshift

    Returns
    -------
    Radial comoving distance: float. unit in Mpc
    """
    # Eqn A9 of Liu,A 2014A
    return (1 + z) ** 2.0 / cosmo.H(z).value * (C / 1e3) / (f21 * 1e6)


def delays_to_kpara(delay, z):
    """Conver delay in sec unit to k_parallel (comoving 1./Mpc along line of sight).

    Parameters
    ----------
    delay : np.array
      The inteferometric delay observed in units second.
    z : float
      The redshift of the expected 21cm emission.

    Returns
    -------
    kparr : np.array
       The spatial fluctuation scale parallel to the line of sight probed by
       the input delay (eta).unit [1/Mpc]
    """
    # Eqn A10 of Liu,A 2014A
    return (delay * 2 * np.pi) / dRpara_df(z)


def kpara_to_delay(kpara, z):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay in sec.

    Parameters
    ----------
    kpara : np.array
      The spatial fluctuation scale parallel to the line of sight in unit 1/Mpc.
    z : float
        The redshift of the expected 21cm emission.

    Returns
    -------
    delay : np.array
      The inteferometric delay in unit second
      which probes the spatial scale given by kparr.
    """
    # Eqn A10 of Liu,A 2014A
    return kpara * dRpara_df(z) / (2 * np.pi)


def u_to_kperp(
    u,
    z,
):
    """Convert baseline length u to k_perpendicular in unit [1/Mpc].

    Parameters
    ----------
    u : float
        The baseline separation of two interferometric antennas in units of wavelength
    z : float
        The redshift of the expected 21cm emission.

    Returns
    -------
    kperp : np.array
        The spatial fluctuation scale perpendicular to the line of sight
        probed by the baseline length u.
    """
    # Eqn A10 of Liu,A 2014A
    return 2 * np.pi * u / dRperp_dtheta(z)


def kperp_to_u(kperp, z):
    """Convert comsological k_perpendicular to baseline length u (wavelength unit).

    Parameters
    ----------
    kperp : np.array
      The spatial fluctuation scale perpendicular to the line of sight. unit 1/Mpc
    z : float
        The redshift of the expected 21cm emission.

    Returns
    -------
    u : np.array
     The baseline separation of two interferometric antennas in units of
     wavelength which probes the spatial scale given by kperp.

    """
    # Eqn A10 of Liu,A 2014A
    return kperp * dRperp_dtheta(z) / (2 * np.pi)


def jy_per_beam_to_kelvin(freq, bl_length):
    """Conversion factor from Jy/beam to kelvin unit.

    The conversion factor is C = (10**-26 * lambda**2)/(2 * K_boltzmann * omega_PSF),
    where omega_PSF is the beam area in sr unit.

    Parameters
    ----------
    freq : np.ndarray[freq]
        frequency in MHz unit
    bl_length : float
        baseline length in meter

    Returns
    -------
    C : np.ndarray[freq]
     The conversion factor from Jy/beam to Kelvin
    """
    Jy = 1.0e-26  # W m^-2 Hz^-1
    wl = units.c / (freq * 1e6)  # freq of the map in MHz

    # Estimate the PSF area, assuming a Gaussian PSF, based on max baseline
    # Bmaj=Bmin= PSF_arcsec; beam_area = (pi * Bmaj * Bmin)/(4 * log(2))
    PSF = 1.22 * wl / bl_length
    PSF = np.degrees(PSF)  # in deg
    omega_psf = (np.pi * PSF**2) / (4 * np.log(2))
    omega_psf_sr = omega_psf * (np.pi / 180.0) ** 2  # convert the PSF area to sr

    kB = units.k_B  # Boltzmann Const in J/k (1.38 * 10-23)
    return wl**2 * Jy / (2 * kB * omega_psf_sr)


def sza2dec(sza):
    """Convert sza to declination.

    Parameters
    ----------
    sza: np.ndarray[el]
      sin(zenith angle) or el.

    Returns
    -------
    dec: np.ndarray[nel]
      declination in degree.
    """
    return _LAT_LON["chime"][0] + np.degrees(np.arcsin(sza))


def noise_equivalent_bandwidth(window):
    """Calculate the relative equivalent noise bandwidth of window function.

    Following this:
    https://dsp.stackexchange.com/questions/70273/find-the-equivalent-noise-bandwidth

    Parameters
    ----------
    window : array_like
        A 1-Dimenaional array like.

    Returns
    -------
    float
    """
    return np.sum(window) ** 2 / (np.sum(window**2) * len(window))


def get_fourier_modes(ra, dec, delays, redshift):
    """Compute the spatial and line-of-sight Fourier modes.

    Parameters
    ----------
    ra : np.array[nra]
      The RA axis in deg unit.
    dec : np.narray[ndec]
      The DEC axis in deg unit.
    delays : np.array[ndelay]
      The delay axis in second unit.
    redshift : float
      redshift at the centre of the band.

    Returns
    -------
    kx : np.array[nra]
      The Fourier modes conjugate to RA axis, unit Mpc^-1.
    ky : np.array[ndec]
      The Fourier modes conjugate to DEC axis, unit Mpc^-1.
    u : np.array[nra]
      gridded u-coordinates
    v : np.array[nel]
      gridded v-coordinates
    k_parallel : np.array[ndelay]
      The Fourier modes conjugate to frequency axis, unit Mpc^-1.
    """
    nra = ra.size
    ndec = dec.size

    # Mean resolution in RA and DEC and convert that to radian
    res_ra_radian = np.deg2rad(np.mean(np.diff(ra)))
    res_dec_radian = np.deg2rad(np.mean(np.diff(dec)))

    DMz = dRperp_dtheta(redshift)  # in Mpc

    # Convert the RA and DEC resolution to Mpc unit
    d_RA_Mpc = DMz * res_ra_radian
    d_DEC_Mpc = DMz * res_dec_radian

    # Estimate the spatial Fourier modes
    k_x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nra, d=d_RA_Mpc))
    k_y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ndec, d=d_DEC_Mpc))

    # The gridded u and v coordinates
    u = kperp_to_u(k_x, redshift)
    v = kperp_to_u(k_y, redshift)

    # Estimate the line-of-sight Fourier modes
    k_para = delays_to_kpara(delays, redshift)

    return k_x, k_y, u, v, k_para


def image_to_uv(data, ra, dec, window="tukey-0.5"):
    """Spatial FFT along RA and DEC axes of the data cube.

    Parameters
    ----------
    data : np.ndarray[ra,el]
       The data, whose spatial FFT will be computed along RA and Dec axes.
    ra : np.array(ra)
      RA of the map in degrees.
    dec : np.array(dec)
      Dec of the map in degrees.
    window: window available in :func:`draco.util.tools.window_generalised()`, optional
       Apply an apodisation function. Default: 'tukey-0.5'.

    Returns
    -------
      data_cube : np.ndarray[kx,ky]
         The 2D spatial FFT of the data cube in (kx,ky) or (u,v) domain.
    """
    # Find the Fourier norm for the FFT
    # The norm is mentioned here - https://numpy.org/doc/2.0/reference/routines.fft.html
    FT_norm = 1 / float(np.prod(np.array(data.shape)))

    if window:
        x_ra = (ra - ra[0]) / (ra[-1] - ra[0])
        x_dec = (dec - dec[0]) / (dec[-1] - dec[0])
        w_ra = tools.window_generalised(x_ra, window=window)
        w_dec = tools.window_generalised(x_dec, window=window)

        # estimate the equivalent noise bandwidth for the tapering window
        NEB_ra = noise_equivalent_bandwidth(w_ra)
        NEB_dec = noise_equivalent_bandwidth(w_dec)
        taper_window = np.outer(w_ra[:, np.newaxis], w_dec[np.newaxis, :])
        data *= taper_window
        uv_map = np.fft.fftshift(np.fft.fft2(data))

    else:
        uv_map = np.fft.fftshift(np.fft.fft2(data))
        NEB_ra = NEB_dec = 1.0

    return uv_map * FT_norm, NEB_ra, NEB_dec


def vol_normalization(ra, dec, freq, redshift):
    """Estimate the volume normalization factor for the power spectrum.

    Parameters
    ----------
    ra : np.array[ra]
      The RA array in deg unit.
    dec : np.narray[dec]
      The DEC array in deg unit.
    freq : np.array[freq]
      The freq array in MHz.
    redshift : float
      redshift at the centre of the band.

    Returns
    -------
    norm : float
      The  Ppower spectrum normalization factor in Mpc^3 unit
    """
    nra = ra.size
    ndec = dec.size
    nfreq = freq.size

    # Mean resolution in RA and DEC and convert that to radian
    res_ra_radian = np.deg2rad(np.mean(np.diff(ra)))
    res_dec_radian = np.deg2rad(np.mean(np.diff(dec)))

    # Comoving distance
    DMz = dRperp_dtheta(redshift)  # in Mpc

    # Convert the RA and DEC resolution to Mpc unit
    dx_Mpc = DMz * res_ra_radian
    dy_Mpc = DMz * res_dec_radian
    Lx = nra * dx_Mpc  # survey length along RA [Mpc]
    Ly = ndec * dy_Mpc  # survey length along DEC [Mpc]

    ## Along line-of-sight direction
    chan_width = np.abs(np.diff(freq)).mean() * 1e6  # channel width in Hz
    dz_Mpc = dRpara_df(redshift) * chan_width  # Mpc
    Lz = dz_Mpc * nfreq  # survey length along line-of-sight [Mpc]

    return Lx * Ly * Lz



def nanaverage(d, w, axis=None):
    """Estimate the nanaverage data using the weight.

    Parameters
    ----------
    d : np.ndarray
     The data to average
    w : np.ndarray
     The weight to use during averaging.
    axis: int, optional
     Axis along which the average will be taken.

    Returns
    -------
    d_avg : np.ndarray
     The weighted average.
    """
    return np.nansum(d * w, axis=axis) / np.nansum(w, axis=axis)


def spatial_mask(k_x, k_y, ew_min, ew_max, ns_bl, wl_min, wl_max, redshift):
    """Estimate the mask in [u,v] or [kx,ky] domain.

    This will generate a mask in uv-domain, which is True
    for the uv-modes corresponding the cylinder east-west
    and north-south baselines, and outside of that region
    it is False. This spatial mask will be applied to the
    power spectrum, before collapsing kx and ky modes to kperp
    and then averaging.
    Note that applying this mask before averaging is crucial
    to avoid modes outside the cylinder.

    Parameters
    ----------
    k_x : np.array[nra]
      The Fourier modes conjugate to RA axis, unit Mpc^-1.
    k_y : np.array[ndec]
      The Fourier modes conjugate to DEC axis, unit Mpc^-1.
    ew_min : float
     Minimum east-west baseline in meter.
    ew_max : float
      Maximum east-west baseline in meter.
    ns_bl : float
      basline along north-south direction to include.
    wl_min : float
      minimum wavelength in meter.
    wl_max : float
      maximum wavelength in meter.
    redshift: float
       redshift at the center of the band.

    Returns
    -------
    fourier_mask : np.ndarray[u,v]
      The  fourier mask in [u,v] or [kx,ky] domain
    """
    # Estimate the uv-range
    ux_min = ew_min / wl_max
    ux_max = ew_max / wl_min
    vy_min = -ns_bl / wl_max
    vy_max = abs(vy_min)

    # Convert the uv_range to kx,ky modes
    kx_min = u_to_kperp(ux_min, redshift)
    kx_max = u_to_kperp(ux_max, redshift)

    ky_min = u_to_kperp(vy_min, redshift)
    ky_max = u_to_kperp(vy_max, redshift)

    # Mask along kx direction
    zone_x1 = (k_x >= kx_min) & (k_x <= kx_max)
    zone_x2 = (k_x >= -kx_max) & (k_x <= -kx_min)
    zone_x = zone_x1 | zone_x2

    # Mask along ky direction
    zone_y1 = (k_y >= ky_min) & (k_y <= ky_max)
    zone_y2 = (k_y >= -ky_max) & (k_y <= -ky_min)
    zone_y = zone_y1 | zone_y2

    # Now take the product of kx and ky mask and return it
    return zone_x[:, None] * zone_y[None, :]



def get_3D_ps(data_cube_1, data_cube_2, vol_norm_factor):
    """Estimate the cross power spectrum of two data cubes.

    The data cubes are complex. This will estimate the cross-correlation
    of two complex data cubes and return the real part of that.

    Parameters
    ----------
    data_cube_1 : np.ndarray[pol,delay,kx,ky]
      complex data cube in (delay,kx,ky) domain.
    data_cube_2 : np.ndarray[pol,delay,kx,ky]
      complex data cube in (delay,kx,ky) domain.
    vol_norm_factor : float
      power spectrum normalization factor in [Mpc^3]

    Returns
    -------
    ps_cube_real: np.ndarray[pol,delay,kx,ky]
       The real part of the power spectrum
    """
    if data_cube_1 is None and data_cube_2 is None:
        raise NameError("Atleast one data cube must be provided")

    if data_cube_2 is None:
        ps_cube_real = (np.conj(data_cube_1) * data_cube_1).real

    else:
        ps_cube_real = (data_cube_1 * np.conj(data_cube_2)).real

    return ps_cube_real * vol_norm_factor


def reshape_data_cube(data_cube, u, v, bl_min, bl_max):
    """Keep non-zero visibility cube between min and max baslines (in wavelength unit).

    For each delay channel, it will keep the non-zero visibilities between min and max
    baseline length and flatten the array, i.e, the output is of shape (nvis),
    where nvis is number of vis between two limits.

    Parameters
    ----------
    data_cube : np.ndarray[kx,ky]
     The data cube to reshape and flatten.
    u : np.ndarray[kx]
     The u-coordinates in wavelength unit.
    v : np.ndarray[ky]
     The v-coordinates in wavelength unit.
    bl_min : float
     The min baseline length in wavelength unit.
    bl_max : float
     The max baseline length in wavelength unit.

    Returns
    -------
    ft_cube : np.ndarray[nvis]
     The flatten data cube.
    uu : np.ndarray[nvis]
     The flatten u-coordinates
    vv : np.ndarray[nvis]
     The flatten v-coordinates
    """
    g_uu, g_vv = np.meshgrid(v, u)
    g_ru = np.sqrt(g_uu**2 + g_vv**2)
    bl_idx = (g_ru >= bl_min) & (g_ru <= bl_max)
    uu = g_uu[bl_idx]
    vv = g_vv[bl_idx]
    ft_cube = data_cube[bl_idx].flatten()

    return ft_cube, uu.flatten(), vv.flatten()


def get_2d_ps(ps_cube, weight, kperp_bins, uu, vv, redshift):
    """Estimate the cylindrically averaged 2D power spectrum.

    Parameters
    ----------
    ps_cube : np.ndarray[nbl]
      The power spectrum array to average in cylindrical bins.
    weight : np.ndarray[nbl]
      The weight to be used in averaging.
       If None, then use unit unifrom weight.
    kperp_bins : float
      The kperp values of each bin, at which the power spectrum will be
       calculated. Unit: Mpc^-1
    uu : np.ndarray[u]
      The flatten u-coordinate in wavelength.
    vv : np.ndarray[v]
      The flatten v-coordinate in wavelength.
    redshift : float
      The redshift corresponding to the band center.

    Returns
    -------
    ps_2D : np.ndarray[kperp]
      The  binned  power along k_perp (cylindrical binning).

    ps_2D_w : np.ndarray[kperp]
      The binned  weight along k_perp (cylindrical binning).

    """
    # Weight for inverse variance weighted avg
    w = weight

    # Find the bin indices and determine which radial bin each cell lies in.
    ku = u_to_kperp(uu, redshift)  # convert the u-array to fourier modes
    kv = u_to_kperp(vv, redshift)  # convert the v-array to fourier modes
    ru = np.sqrt(ku**2 + kv**2)

    # Digitize the bins
    bin_indx = np.digitize(ru, bins=kperp_bins)

    # Define empty list to store the binned 2D PS
    ps_2D = []
    ps_2D_w = []

    # Now bin in 2D ##
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in np.arange(len(kperp_bins) - 1) + 1:
            p = np.nansum(w[bin_indx == i] * ps_cube[bin_indx == i]) / np.sum(
                w[bin_indx == i]
            )
            ps_2D.append(p)
            ps_2D_w.append(np.sum(w[bin_indx == i]))

    return np.array(ps_2D), np.array(ps_2D_w)


def get_1d_ps(
    ps_2D,
    k_perp,
    k_para,
    weight_cube,
    signal_window=None,
    Nbins_3D=10,
    logbins_3D=True,
):
    """Compute the 3D spherically averaged power spectrum.

    Parameters
    ----------
    ps_2D :  np.ndarray[kpar,kperp]
     The cylindrically averaged 2D power spectrum.
    k_perp : np.array[kperp]
     The k_perp array after cylindrically binning.
    k_para : np.array[kpar]
     The k_parallel array.
    weight_cube :  np.ndarray[kpar,kperp]
      The weight array to use during spherical averaging.
    signal_window :  np.ndarray[kpar,kperp]
      The signal window mask.
    Nbins_3D : int
      The number of 3D bins
    logbins_3D : bool
      Bin in logarithmic space if True.

    Returns
    -------
    k1d : np.array[Nbins_3d]
      The K-values corresponding to the bin center
    ps_3D : np.array[Nbins_3d]
     The spherically avg power spectrum.
    ps_3D_err : np.array[Nbins_3d]
      The error in the PS (sample variance)
    """
    # Estimate the 1D k-modes
    kpp, kll = np.meshgrid(k_perp, k_para)
    k = np.sqrt(kpp**2 + kll**2)

    # apply the window mask
    if signal_window is not None:
        k = k[signal_window]
        ps_2D = ps_2D[signal_window]
        w = weight_cube[signal_window]

    # Take the min and max of 1D k-modes
    kmin = k[k > 0].min()
    kmax = k.max()

    # estimat the bins
    if logbins_3D:
        kbins = np.logspace(np.log10(kmin), np.log10(kmax), Nbins_3D)
    else:
        kbins = np.linspace(kmin, kmax, Nbins_3D)

    # Flatten the arrays
    p1D = ps_2D.flatten()
    w1D = w.flatten()
    k1D = k.flatten()

    # find the indices of bins
    indices = np.digitize(k1D, kbins)

    # Define the empty lists to save the
    # averaged power spec and other relevant quantities
    ps_3D = []
    ps_3D_err = []
    k3D = []
    variance = []
    # Average the modes falls in each bin
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in np.arange(len(kbins) - 1) + 1:
            w_b = w1D[indices == i]
            p = np.nansum(w_b * p1D[indices == i]) / np.sum(w_b)
            p_err = np.sqrt(np.sum(w_b**2 * p**2) / np.sum(w_b) ** 2)
            k_mean_b = nanaverage(k1D[indices == i], w_b)
            k3D.append(k_mean_b)
            var = 1 / np.sum(w_b)

            ps_3D.append(p)
            ps_3D_err.append(p_err)
            variance.append(var)

    # convert the list to array
    k3D = np.array(k3D)
    ps_3D = np.array(ps_3D)
    ps_3D_err = np.array(ps_3D_err)
    variance = np.array(variance)

    return k3D, ps_3D, ps_3D_err, variance
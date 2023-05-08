"""Power spectrum estimation from ringmap."""

import numpy as np
from astropy import constants as const
import scipy.signal.windows as windows
from astropy.cosmology import Planck15, default_cosmology
from astropy import units as un

from draco.core import containers, task
from caput import mpiarray, config
from draco.analysis.delay import match_axes

# Defining the cosmology - Planck15 of astropy
cosmo = Planck15
f21 = 1420.405752 * un.MHz  # MHz
c = const.c.value  # m/s

# Lat/Lon of CHIME
_LAT_LON = {
    "chime": [49.3207125, -119.623670],
}


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
        in the weight dataset. Default: True.
    apply_window : bool, optional
        Whether to apply apodisation to frequency axis. Default: True.
    window_name : window available in scipy.signal.windows, optional
        Apodisation to perform on frequency axis. Default: "nuttall".
    """

    apply_pixel_mask = config.Property(proptype=bool, default=True)
    apply_window = config.Property(proptype=bool, default=True)
    window_name = config.enum(
        ["nuttall", "blackman_nuttall", "blackman_harris"], default="nuttall"
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
                "Input container must be instance of "
                "RingMap (received %s)" % rm.__class__
            )

        freq_spacing = np.abs(np.diff(rm.freq[:])).min()
        ndelay = len(rm.freq)

        # Compute delays in us
        self.delays = np.fft.fftshift(np.fft.fftfreq(ndelay, d=freq_spacing))

        # Find the relevant axis positions
        # Extract the required axes
        data_axes = list(rm.map.attrs["axis"])
        ax_freq = data_axes.index("freq")
        ax_ra = data_axes.index("ra")

        # Create a view of the dataset with the relevant axes at the back,
        # and all other axes compressed
        data_view = np.moveaxis(
            rm.datasets["map"][:].local_array,
            [ax_ra, ax_freq],
            [-2, -1],
        )
        data_view = data_view.reshape(-1, data_view.shape[-2], data_view.shape[-1])
        data_view = mpiarray.MPIArray.wrap(data_view, axis=2, comm=rm.comm)
        nbase = int(np.prod(data_view.shape[:-2]))
        data_view = data_view.redistribute(axis=0)

        # ... do the same for the weights, but we also need to make the weights full
        # size
        weight_full = np.zeros(rm.datasets["map"][:].shape, dtype=rm.weight.dtype)
        weight_full[:] = match_axes(rm.datasets["map"], rm.weight)
        weight_view = np.moveaxis(weight_full, [ax_ra, ax_freq], [-2, -1])
        weight_view = weight_view.reshape(
            -1, weight_view.shape[-2], weight_view.shape[-1]
        )
        weight_view = mpiarray.MPIArray.wrap(weight_view, axis=2, comm=rm.comm)
        weight_view = weight_view.redistribute(axis=0)

        # Initialise the spectrum container
        delay_spectrum = containers.DelayTransform(
            baseline=nbase,
            sample=rm.index_map["ra"],
            delay=self.delays,
            attrs_from=rm,
            weight_boost=1.0,
        )
        delay_spectrum.redistribute("baseline")
        delay_spectrum.spectrum[:] = 0.0
        bl_axes = [da for da in data_axes if da not in ["ra", "freq"]]

        # Copy the index maps for all the flattened axes into the output container, and
        # write out their order into an attribute so we can reconstruct this easily
        # when loading in the spectrum
        for ax in bl_axes:
            delay_spectrum.create_index_map(ax, rm.index_map[ax])
        delay_spectrum.attrs["baseline_axes"] = bl_axes

        # Create the index map for frequency axis
        delay_spectrum.create_index_map("freq", rm.freq)

        # Do delay transform
        for lbi, bi in delay_spectrum.spectrum[:].enumerate(axis=0):
            self.log.debug(
                f"Estimating the delay spectrum of each baseline {bi}/{nbase} by FFT "
            )

            # Get the local selections
            data = data_view.local_array[lbi]
            weight = weight_view.local_array[lbi]

            if self.apply_pixel_mask:
                pixel_mask = np.where(weight > 0.0, 1.0, weight)
                data *= pixel_mask

            # Do FFT along freq
            if self.apply_window:
                window = windows.get_window(self.window_name, ndelay)
                w = window[np.newaxis, :]
                yspec = np.fft.fftshift(np.fft.ifft(data * w, axis=-1), axes=-1)

            else:
                yspec = np.fft.fftshift(np.fft.ifft(data, axis=-1), axes=-1)

            delay_spectrum.spectrum[bi] = yspec

        return delay_spectrum


class SpatialTransformDelayMap(task.SingleTask):
    """Spatial transform the delay map from (RA,DEC) to (u,v) domain.

    This transforms the delay data cube to spatial (u,v) domain
    by taking a 2D FFT along (RA,DEC) axes.

    Attributes
    ----------
    apply_spatial_window : bool, optional
        Whether to apply apodisation to spatial axes. If true,
        a 2D Tukey tapering window will be applied to the data. Default: True.

    """

    apply_spatial_window = config.Property(proptype=bool, default=True)

    def process(self, ds):
        """Estimate the spatial transform of the delay map.

        Parameters
        ----------
        ds : containers.DelayTransform
          The delay map, whose spatial transform will be estimated.

        Returns
        -------
        spatial cube : containers.SpatialTransform
           The data cube in (delay,u,v) domain.
        """
        if not isinstance(ds, containers.DelayTransform):
            raise ValueError(
                "Input container must be instance of "
                "DelayTransform (received %s)" % ds.__class__
            )

        ds.redistribute("delay")

        # Extract required data axes
        delays = ds.index_map["delay"]  # micro-sec
        el = ds.index_map["el"]
        pol = ds.index_map["pol"]
        ra = ds.index_map["sample"]
        freq = ds.index_map["freq"]  # MHz
        dec = sza2dec(el)

        # Unpack the baseline axis of the delay spectrum
        # and reshape it as (pol,delay,ra,el)
        axes = list(ds.attrs["baseline_axes"])
        shp = tuple([ds.index_map[ax].size for ax in axes])
        data_view = ds.spectrum[:].local_array.reshape(shp + (ra.size,) + (-1,))[0, :]
        data_view = np.swapaxes(data_view, 1, 3)
        data_view = mpiarray.MPIArray.wrap(data_view, axis=1, comm=ds.comm)
        data_view = data_view.redistribute(axis=1)

        # Estimate the Fourier modes
        nu_c = freq[int(freq.size / 2.0)]
        redshift = f21.value / nu_c - 1
        kx, ky, u, v, k_parallel = get_fourier_modes(ra, dec, delays * 1e-6, redshift)

        # Initialise the spatial transformed data cube container
        vis_cube = containers.SpatialDelayCube(
            pol=pol, delay=delays, kx=kx, ky=ky, attrs_from=ds
        )
        vis_cube.redistribute("delay")
        vis_cube.data_tau_uv[:] = 0.0

        # Create the index map for ra, el and freq axes
        # Note that the 'sample' axis is the 'ra' axis in
        # DelayTransform container for a ringmap
        map_axes = ["sample", "el", "freq"]
        for ax in map_axes:
            vis_cube.create_index_map(ax, ds.index_map[ax])

        # Save the central freq of the band and the
        # corresponding redshift in the attrs
        vis_cube.attrs["freq_center"] = nu_c
        vis_cube.attrs["redshift"] = redshift

        # Create index map for (u,v) coordinates and k_parallel
        fourier_axes = ["u", "v", "k_parallel"]
        for ff in fourier_axes:
            vis_cube.create_index_map(ff, locals()[ff])

        # Spatial transform to Fourier domain
        # loop over pol
        for pp, psr in enumerate(pol):
            self.log.debug(f"Estimating spatial FFT for pol: {psr}")
            # loop over delays
            for lde, de in vis_cube.data_tau_uv[:].enumerate(axis=1):
                slc = (pp, lde, slice(None), slice(None))
                data = np.ascontiguousarray(data_view.local_array[slc])
                data_uv = image_to_uv(data, window=self.apply_spatial_window)
                vis_cube.data_tau_uv[pp, de] = data_uv

        return vis_cube


class CrossPowerSpectrum3D(task.SingleTask):
    """Estimate the 3D cross power spectrum of two data cubes .

    This estimates the 3D cross power spectrum of two data cubes by taking
    real part of correlation two complex data cubes and normalize that
    be the volume of the data cube in Mpc^3. The unit of the power spectrum
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
        if type(data_1) != type(data_2):
            raise TypeError(
                f"type(data_1) (={type(data_1)}) must match "
                f"type(data_2) (={type(data_2)}"
            )

        data_1.redistribute("delay")
        data_2.redistribute("delay")

        # Extract required data axes
        pol = data_1.index_map["pol"]
        el = data_1.index_map["el"]
        ra = data_1.index_map["sample"]
        freq = data_1.index_map["freq"]  # MHz
        dec = sza2dec(el)
        redshift = data_1.attrs["redshift"]
        nu_c = data_1.attrs["freq_center"]  # MHz

        # Estimate the PSF area, assuming a Gaussian PSF, based on max baseline
        # Bmaj=Bmin= PSF_arcsec; beam_area = (pi * Bmaj * Bmin)/(4 * log(2))
        wl = const.c.value / (nu_c * 1e6)  # m
        PSF = wl / 100.0  # The max baseline is assumed 100.0m and hard coded here
        PSF_arcsec = np.degrees(PSF) * 3600  # in arcsec
        omega_psf = (np.pi * PSF_arcsec**2) / ((4 * np.log(2)))
        omega_psf_sr = (omega_psf * un.arcsec**2).to("sr")

        # Estimate the conversion factor from Jy/beam to Kelvin
        factor = jy_per_beam_to_kelvin(freq, omega_psf_sr)

        # Dereference the required datasets and
        vis_cube_1 = data_1.data_tau_uv[:].local_array
        vis_cube_2 = data_2.data_tau_uv[:].local_array

        # Compute power spectrum normalization factor
        ps_norm = vol_normalization(ra, dec, freq, redshift)

        # we estimate only XX and YY power spectrum of the data
        co_pol = ["XX", "YY"]

        # Initialise the 3D power spectrum container
        ps_cube = containers.Powerspec3D(
            pol=np.array(co_pol), axes_from=data_1, attrs_from=data_1
        )
        ps_cube.redistribute("delay")
        ps_cube.ps3D[:] = 0.0

        # Save the power spectrum normalization factor in the attrs
        ps_cube.attrs["ps_norm"] = ps_norm

        # Create index map for (u,v) coordinates and k_parallel axis
        fourier_axes = ["u", "v", "k_parallel"]
        for ax in fourier_axes:
            ps_cube.create_index_map(ax, data_1.index_map[ax])

        # Estimate the cross power spectrum
        for pp, psr in enumerate(co_pol):
            self.log.debug(f"Estimating power spectrum for pol: {psr}")
            pol_id = list(pol).index(psr)

            for lde, de in ps_cube.ps3D[:].enumerate(axis=1):
                slc = (pol_id, lde, slice(None), slice(None))
                cube_1 = np.ascontiguousarray(vis_cube_1[slc] * factor[lde])
                cube_2 = np.ascontiguousarray(vis_cube_2[slc] * factor[lde])
                power_3D = get_ps(cube_1, cube_2, vol_norm_factor=ps_norm)
                ps_cube.ps3D[pp, de] = power_3D

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
        ps = super().process(data, data)
        return ps


class CylindricalPowerSpectrum2D(task.SingleTask):
    """Estimate the cylindrically averaged 2D power spectrum.

    This estimates the cylindrically averaged 2D power spectrum from a
    3D power spectrum cube.

    Attributes
    ----------
    weight_cube : str
        The name of an hdf5 file containing the weight cube to be used in averaging.
        For inverse variance weighting, weight_cube = 1/sigma**2,
        Note that, weight_cube should be of the same shape as the input  power spectrum cube.
        TODO : If a filename is provided,  then it should be loaded during setup and will be
        used during power spectrum averaging. Currently we do not have inverse variance cube, so
        the default is set to None and this is the only option for now.
        Default : None
    bl_min : float
       The minimum baseline length in meter to include in power spectrum binning. Default: 14.0
    bl_max : float
       The minimum baseline length in meter to include in power spectrum binning. Default: 60.0
    Nbins_2D : int
       The number of bins in 2D cylindrical binning. Default: 30
    logbins_2D : bool, optional
        If True, use logarithmic binning in cylindrical averaging. Default: False
    delay_cut : float
        Throw away the delay modes below this cutoff during spherical averaging, unit sec.
        This is same for both polarization. Default: 300.0e-9
    """

    weight_cube = config.Property(proptype=str, default=None)
    bl_min = config.Property(proptype=float, default=14.0)
    bl_max = config.Property(proptype=float, default=60.0)
    Nbins_2D = config.Property(proptype=int, default=30)
    logbins_2D = config.Property(proptype=bool, default=False)
    delay_cut = config.Property(proptype=float, default=300.0e-9)

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
                "Input container must be instance of "
                "Powerspec3D (received %s)" % ps.__class__
            )

        ps.redistribute("delay")

        # Extract required data axes
        pol = ps.index_map["pol"]
        kpar = ps.index_map["k_parallel"]
        u = ps.index_map["u"]
        v = ps.index_map["v"]
        redshift = ps.attrs["redshift"]
        nu_c = ps.attrs["freq_center"]
        wl = const.c.value / (nu_c * 1e6)  # m

        # find out the kperp bins
        u_min_lambda = self.bl_min / wl
        u_max_lambda = self.bl_max / wl
        kperp_min = u_to_kperp(u_min_lambda, redshift, cosmo=cosmo).value
        kperp_max = u_to_kperp(u_max_lambda, redshift, cosmo=cosmo).value

        if self.logbins_2D:
            kperp = np.logspace(
                np.log10(kperp_min), np.log10(kperp_max), self.Nbins_2D + 1
            )
        else:
            kperp = np.linspace(kperp_min, kperp_max, self.Nbins_2D + 1)

        # Dereference the required datasets
        ps_3D = ps.ps3D[:].view(np.ndarray)

        # Define the 2D power spectrum container
        pspec_2D = containers.Powerspec2D(
            pol=pol, kpar=kpar, kperp=kperp, attrs_from=ps
        )
        pspec_2D.redistribute("kpar")
        pspec_2D.ps2D[:] = 0.0

        # save the delay cut value in attrs
        pspec_2D.attrs["delay_cut"] = self.delay_cut

        # loop over polarization
        for pp, psr in enumerate(pol):
            self.log.debug(f"Estimating 2D power spectrum for pol: {psr}")

            for lde, de in pspec_2D.ps2D[:].enumerate(axis=1):
                slc = (pp, lde, slice(None), slice(None))
                data = np.ascontiguousarray(ps_3D[slc])

                # keep the non-zero visibilities between minimum and maximum baseline
                # for each delay channel and take the flatten array; i.e, the output is (nvis),
                # where nvis is number of vis between two limit

                ps3D_flat, uu, vv = reshape_data_cube(
                    data, u, v, u_min_lambda, u_max_lambda
                )

                # Cylindrical 2D power spectrum
                pspec_2D.ps2D[pp, de], pspec_2D.ps2D_weight[pp, de] = get_2d_ps(
                    ps3D_flat,
                    w=self.weight_cube,
                    kperp_bins=kperp,
                    uu=uu,
                    vv=vv,
                    redshift=redshift,
                )

        # Generate the signal window mask
        # Note that, we are not applying this mask to the 2D power spectrum
        # Instead, we save this as a dataset in the container, which can
        # be applied during plotting
        pspec_2D.redistribute("pol")
        pspec_2D.signal_mask[:] = 1.0

        if self.delay_cut > 0.0:
            kpar_lim = delay_to_kpara(self.delay_cut * un.s, redshift, cosmo=cosmo)
            ibins = np.where((kpar > -kpar_lim.value) & (kpar < kpar_lim.value))
            for ii, jj in enumerate(pol):
                pspec_2D.signal_mask[ii, ibins, :] = 0.0

        return pspec_2D


class SphericalPowerSpectrum1D(task.SingleTask):
    """Estimate the spherically averaged 1D power spectrum.

    This estimates the spherically averaged 1D power spectrum from a
    2D cylindrical power spectrum.

    Attributes
    ----------
    Nbins_3D : int
       The number of bins in 3D spherical binning. Default: 10
    logbins_3D : bool, optional
        If True, use logarithmic binning in cylindrical averaging. Default: False

    """

    Nbins_3D = config.Property(proptype=int, default=10)
    logbins_3D = config.Property(proptype=bool, default=False)

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
                "Input container must be instance of "
                "Powerspec2D (received %s)" % ps2D.__class__
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

        # Define the 2D power spectrum container
        pspec_1D = containers.Powerspec1D(pol=pol, k=self.Nbins_3D, attrs_from=ps2D)
        pspec_1D.redistribute("pol")
        pspec_1D.ps1D[:] = 0.0

        # loop over polarization
        for pp, psr in pspec_1D.ps1D[:].enumerate(axis=0):
            self.log.debug(f"Estimating 1D power spectrum for pol: {psr}")

            pspec_1D.k1D[psr], pspec_1D.ps1D[psr], pspec_1D.ps1D_error[psr] = get_3d_ps(
                ps_2D[pp],
                kperp,
                kpar,
                signal_window=mask_2D[pp],
                Nbins_3D=self.Nbins_3D,
                weight_cube=weight_2D[pp],
                logbins_3D=self.logbins_3D,
            )

        return pspec_1D


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

    DMz = cosmo.comoving_transverse_distance(redshift).value  # in Mpc

    # Convert the RA and DEC resolution to Mpc unit
    d_RA_Mpc = DMz * res_ra_radian
    d_DEC_Mpc = DMz * res_dec_radian

    # Estimate the spatial Fourier modes
    k_x = 2 * np.pi * (np.fft.fftfreq(nra, d=d_RA_Mpc))
    k_y = 2 * np.pi * (np.fft.fftfreq(ndec, d=d_DEC_Mpc))
    k_x = np.fft.fftshift(k_x)
    k_y = np.fft.fftshift(k_y)

    # The gridded u and v coordinates
    u = (DMz * k_x) / (2 * np.pi)
    v = (DMz * k_y) / (2 * np.pi)

    # Estimate the line-of-sight Fourier modes
    k_parallel = delay_to_kpara(delays * un.s, redshift, cosmo=cosmo)

    return k_x, k_y, u, v, k_parallel.value


def image_to_uv(data, window=True, window_name="tukey", alpha=0.5):
    """Spatial FFT along RA and DEC axes of the data cube.

    Parameters
    ----------
    data : np.ndarray[ra,el]
       The data, whose spatial FFT will be computed along RA and DEC axes.
    window : bool.
       If True apply a spatial apodisation function. Default: True.
    window_name : 'Tukey'
       Apply Tukey spatial tapering window. Default: 'Tukey'.
    alpha : float
       Shape parameter of the Tukey window.
       0 = rectangular window, 1 = Hann window. We are taking 0.5 (default), which lies in between.

    Returns
    -------
      data_cube : np.ndarray[kx,ky]
         The 2D spatial FFT of the data cube in (kx,ky) or (u,v) domain.
    """
    from scipy.signal import windows

    FT_norm = 1 / float(np.prod(np.array(data.shape)))

    if window:
        window_func = getattr(windows, window_name)
        w_dec = window_func(data.shape[0], alpha=alpha)  # taper in the DEC direction
        w_ra = window_func(data.shape[1], alpha=alpha)  # taper in the RA direction
        taper_window = np.outer(
            w_dec[:, np.newaxis], w_ra[np.newaxis, :]
        )  # Make the 2D tapering function by taking the outer product
        data *= taper_window

    uv_map = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data)))

    return uv_map * FT_norm


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
    DMz = cosmo.comoving_transverse_distance(redshift).value  # in Mpc

    # Mean resolution in RA and DEC and convert that to radian
    res_ra_radian = np.deg2rad(np.mean(np.diff(ra)))
    res_dec_radian = np.deg2rad(np.mean(np.diff(dec)))

    # Convert the RA and DEC resolution to Mpc unit
    d_RA_Mpc = DMz * res_ra_radian
    d_DEC_Mpc = DMz * res_dec_radian
    Lx = nra * d_RA_Mpc  # survey length along RA [Mpc]
    Ly = ndec * d_DEC_Mpc  # survey length along DEC [Mpc]

    ## Along line-of-sight direction
    chan_width = abs(freq[0] - freq[1])  # [MHz]
    c = const.c.to("km/s").value  # [km/s]
    Hz = cosmo.H(redshift).value  # [km/s/Mpc]
    d_z = (
        c * (1 + redshift) ** 2 * chan_width / Hz / f21.value
    )  # [Mpc] The sampling interval along the line-of-sight direction. Ref.[liu2014].Eq.(A9)
    Lz = d_z * nfreq  # survey length along line-of-sight [Mpc]

    norm = Lx * Ly * Lz

    return norm


def jy_per_beam_to_kelvin(freq, beam_area=None):
    """Conversion factor from Jy/beam to kelvin unit.

    The conversion factor is C = (10**-26 * lambda**2)/(2 * K_boltzmann * omega_PSF),
    where omega_PSF is the beam area in sr unit.

    Parameters
    ----------
    freq : np.ndarray[freq]
        frequency in MHz unit
    beam_area : float
        synthesized beam area in sr unit

    Returns
    -------
    C : np.ndarray[freq]
     The conversion factor from Jy/beam to Kelvin
    """
    Jy = 1.0e-26  # W m^-2 Hz^-1
    c = const.c.value  # m/s
    wl = c / (freq * 1e6)  # freq of the map in MHz
    kB = const.k_B.value  # Boltzmann Const in J/k (1.38 * 10-23)
    C = wl**2 * Jy / (2 * kB * beam_area.value)
    return C


def get_ps(data_cube_1, data_cube_2, vol_norm_factor):
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
    where nvis is number of vis between two limit

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


def get_2d_ps(ps_cube, w, kperp_bins, uu, vv, redshift):
    """Estimate the cylindrically averaged 2D power spectrum.

    Parameters
    ----------
    ps_cube : np.ndarray[nvis]
      The power spectrum array to average in cylindrical bins.
    w : np.ndarray[nvis]
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
    # Check the weight
    if w is None:
        w = np.ones_like(ps_cube)

    # Find the bin indices and determine which radial bin each cell lies in.
    ku = u_to_kperp(uu, redshift)  # convert the u-array to fourier modes
    kv = u_to_kperp(vv, redshift)  # convert the v-array to fourier modes

    ru = np.sqrt(ku.value**2 + kv.value**2)

    # Digitize the bins
    bin_indx = np.digitize(ru, bins=kperp_bins)

    # Define empty list to store the binned 2D PS
    ps_2D = []
    ps_2D_w = []

    # Now bin in 2D ##
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in np.arange(len(kperp_bins)) + 1:
            p = np.nansum(w[bin_indx == i] * ps_cube[bin_indx == i]) / np.sum(
                w[bin_indx == i]
            )
            ps_2D.append(p)
            ps_2D_w.append(np.sum(w[bin_indx == i]))

    return np.array(ps_2D), np.array(ps_2D_w)


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


def get_3d_ps(
    ps_2D,
    k_perp,
    k_para,
    signal_window=None,
    Nbins_3D=100,
    weight_cube=None,
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
     The k_parallel array, only the positive half.
    signal_window :  np.ndarray[kpar,kperp]
      The signal window mask.
    Nbins_3D : int
      The number of 3D bins
    weight_cube :  np.ndarray[kpar,kperp]
      The weight array to use during spherical averaging.
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
    kpp, kll = np.meshgrid(k_perp, k_para)
    k = np.sqrt(kpp**2 + kll**2)
    kmin = k[k > 0].min()
    kmax = k.max()

    # Weight cube
    if weight_cube is None:
        w = 1 * np.ones_like(ps_2D)

    else:
        w = weight_cube

    # Signal window
    if signal_window is not None:
        k *= signal_window
        ps_2D *= signal_window
        w *= signal_window

    ps_3D = []
    ps_3D_err = []
    k1D = []

    # bins
    if logbins_3D:
        kbins = np.logspace(np.log10(kmin), np.log10(kmax), Nbins_3D + 1)
    else:
        kbins = np.linspace(kmin, kmax, Nbins_3D + 1)

    # Flatten the arrays
    p1D = ps_2D.flatten()
    w1D = w.flatten()
    ks = k.flatten()

    # Digitize the bins
    indices = np.digitize(ks, kbins)

    # 3D binning
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in np.arange(len(kbins) - 1) + 1:
            w_b = w1D[indices == i]
            p = np.nansum(w_b * p1D[indices == i]) / np.sum(w_b)
            p_err = np.sqrt(2 * np.sum(w_b**2 * p**2) / np.sum(w_b) ** 2)
            k_mean_b = nanaverage(ks[indices == i], w_b)
            k1D.append(k_mean_b)
            ps_3D.append(p)
            ps_3D_err.append(p_err)

    k1D = np.array(k1D)
    ps_3D = np.array(ps_3D)
    ps_3D_err = np.array(ps_3D_err)

    # replacing any nan with 0
    k1D[np.isnan(k1D)] = 0.0
    ps_3D[np.isnan(ps_3D)] = 0.0
    ps_3D_err[np.isnan(ps_3D_err)] = 0.0

    return k1D, ps_3D, ps_3D_err


def delay_to_kpara(delay, z, cosmo=None):
    """Conver delay in sec unit to k_parallel (comoving 1./Mpc along line of sight).

    Parameters
    ----------
    delay : Astropy Quantity object with units equivalent to time.
        The inteferometric delay observed in units compatible with time.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to planck2015 year in "little h" units

    Returns
    -------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight probed by the input delay (eta).
    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (
        delay * (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z)) / (const.c * (1 + z) ** 2)
    ).to("1/Mpc")


def kpara_to_delay(kpara, z, cosmo=None):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay in sec.

    Parameters
    ----------
    kpara : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    delay : Astropy Quantity units equivalent to time
        The inteferometric delay which probes the spatial scale given by kparr.
    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (
        kpara * const.c * (1 + z) ** 2 / (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))
    ).to("s")


def kperp_to_u(kperp, z, cosmo=None):
    """Convert comsological k_perpendicular to baseline length u (wavelength unit).

    Parameters
    ----------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    u : float
        The baseline separation of two interferometric antennas in units of
        wavelength which probes the spatial scale given by kperp
    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return kperp * cosmo.comoving_transverse_distance(z) / (2 * np.pi)


def u_to_kperp(u, z, cosmo=None):
    """Convert baseline length u to k_perpendicular.

    Parameters
    ----------
    u : float
        The baseline separation of two interferometric antennas in units of wavelength
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight probed by the baseline length u.
    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return 2 * np.pi * u / cosmo.comoving_transverse_distance(z)

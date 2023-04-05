"""Wavelet power spectrum estimation."""

import numpy as np
import scipy.linalg as la
import pywt

from caput import config

from ..core import containers, task
from .delay import flatten_axes
from ..util import _fast_tools


class WaveletSpectrumEstimator(task.SingleTask):
    """Estimate a continuous wavelet power spectrum from the data.

    This requires the input of the underlying data, *and* an estimate of its delay
    spectrum to allow us to in-fill any masked frequencies.
    """

    dataset = config.Property(proptype=str, default="vis")
    average_axis = config.Property(proptype=str)
    ndelay = config.Property(proptype=int, default=128)
    wavelet = config.Property(proptype=str, default="morl")

    def process(
        self,
        data: containers.FreqContainer,
        dspec: containers.DelaySpectrum,
    ) -> containers.WaveletSpectrum:
        """Estimate the wavelet power spectrum.

        Parameters
        ----------
        data
            Must have a frequency axis, and the axis to average over. Any other axes
            will be flattened in the output.
        dspec
            The delay spectrum. The flattened `baseline` axis must match the remaining
            axes in `data`.

        Returns
        -------
        wspec
            The wavelet spectrum. The non-frequency and averaging axes have been
            collapsed into `baseline`.
        """

        dset_view, bl_axes = flatten_axes(
            data[self.dataset], [self.average_axis, "freq"]
        )
        weight_view, _ = flatten_axes(
            data.weight,
            [self.average_axis, "freq"],
            match_dset=data[self.dataset],
        )

        nbase = dset_view.global_shape[0]

        df = np.abs(data.freq[1] - data.freq[0])
        delay_scales = np.arange(1, self.ndelay + 1) / (2 * df * self.ndelay)

        wv_scales = pywt.frequency2scale(self.wavelet, delay_scales * df)

        wspec = containers.WaveletSpectrum(
            baseline=nbase,
            axes_from=data,
            attrs_from=data,
            delay=delay_scales,
        )
        # Copy the index maps for all the flattened axes into the output container, and
        # write out their order into an attribute so we can reconstruct this easily
        # when loading in the spectrum
        for ax in bl_axes:
            wspec.create_index_map(ax, data.index_map[ax])
        wspec.attrs["baseline_axes"] = bl_axes

        wspec.redistribute("baseline")
        dspec.redistribute("baseline")
        ws = wspec.spectrum[:].local_array
        ds = dspec.spectrum[:].local_array

        # Construct the freq<->delay mapping Fourier matrix
        F = np.exp(
            -2.0j
            * np.pi
            * dspec.index_map["delay"][np.newaxis, :]
            * data.freq[:, np.newaxis]
        )

        for ii in range(dset_view.shape[0]):
            d = dset_view.local_array[ii]
            w = weight_view.local_array[ii]

            # Construct an averaged frequency mask
            Ni = w.mean(axis=0)

            D = ds[ii]
            Df = (F * D[np.newaxis, :]) @ F.T.conj()
            iDf = la.inv(Df)
            Ci = iDf + np.diag(Ni)

            # Solve for the infilled data
            d_infill = la.solve(
                Ci,
                Ni[:, np.newaxis] * d.T,
                assume_a="pos",
                overwrite_a=True,
                overwrite_b=True,
            ).T

            wd, s = pywt.cwt(
                d_infill,
                scales=wv_scales,
                wavelet=self.wavelet,
                axis=-1,
                sampling_period=df,
                method="fft",
            )

            ws[ii] = wd.var(axis=1)
            _fast_tools._fast_var(wd, ws[ii])

        return wspec

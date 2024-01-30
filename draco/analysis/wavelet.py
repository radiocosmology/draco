"""Wavelet power spectrum estimation."""

import os

import numpy as np
import pywt
import scipy.fft as fft
import scipy.linalg as la
from caput import config, mpiutil

from ..core import containers, task
from ..util import _fast_tools
from .delay import flatten_axes


class WaveletSpectrumEstimator(task.SingleTask):
    """Estimate a continuous wavelet power spectrum from the data.

    This requires the input of the underlying data, *and* an estimate of its delay
    spectrum to allow us to in-fill any masked frequencies.
    """

    dataset = config.Property(proptype=str, default="vis")
    average_axis = config.Property(proptype=str)
    ndelay = config.Property(proptype=int, default=128)
    wavelet = config.Property(proptype=str, default="morl")
    chunks = config.Property(proptype=int, default=4)

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
        workers = int(os.environ.get("OMP_NUM_THREADS", 1))

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
        ww = wspec.weight[:].local_array
        ds = dspec.spectrum[:].local_array

        # Construct the freq<->delay mapping Fourier matrix
        F = np.exp(
            -2.0j
            * np.pi
            * dspec.index_map["delay"][np.newaxis, :]
            * data.freq[:, np.newaxis]
        )

        for ii in range(dset_view.shape[0]):
            self.log.info(f"Transforming baseline {ii} of {dset_view.shape[0]}")
            d = dset_view.local_array[ii]
            w = weight_view.local_array[ii]

            # Construct an averaged frequency mask and use it to set the output
            # weights
            Ni = w.mean(axis=0)
            ww[ii] = Ni

            # Construct a Wiener filter to in-fill the data
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

            # Doing the cwt and calculating the variance can eat a bunch of
            # memory. Break it up into chunks to try and control this
            for _, s, e in mpiutil.split_m(wv_scales.shape[0], self.chunks).T:
                with fft.set_workers(workers):
                    wd, _ = pywt.cwt(
                        d_infill,
                        scales=wv_scales[s:e],
                        wavelet=self.wavelet,
                        axis=-1,
                        sampling_period=df,
                        method="fft",
                    )

                # Calculate and set the variance
                _fast_tools._fast_var(wd, ws[ii, s:e])

        return wspec

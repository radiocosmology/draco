"""Implement HyFoReS to correct bandpass gains.

Two tasks are needed.
The first task should be chosen from one of the options below.

DelayFilterHyFoReSBandpassHybridVis :
    Estimate bandpass gains and their window matrix from unfiltered hybrid vis.

DelayFilterHyFoReSBandpassHybridVisMask :
    Same as the previous task but adding a pixel mask to mask sidelobes.

HyFoReSBandpassHybridVis :
    Same as the first task but does not implement the delay filter.
    This allows RA median subtraction before HyFoReS.
    Namely, having a separate task to delay filter hybrid vis.
    Then apply RA median subtraction, followed by HyFoReS.

HyFoReSBandpassHybridVisMask :
    This task does not implement the delay filter.
    Otherwise same as DelayFilterHyFoReSBandpassHybridVisMask.

HyFoReSBandpassHybridVisMaskKeepSource :
    This task masks source sidelobes but keeps their mainlobes.
    Otherwise same as HyFoReSBandpassHybridVisMask.
    The files to construct the mask are provided as the inputs.

The second task is DelayFilterHyFoReSBandpassHybridVisClean.
It compensates the bandpass window and subtracts the residual gain errors.

Currently, the best choice to run HyFoReSBandpassHybridVisMaskKeepSource.
Then run DelayFilterHyFoReSBandpassHybridVisClean.
Contact Haochen for an example config file.
"""

import time

import numpy as np
from caput import config, task, units
from mpi4py import MPI
from scipy import linalg as la

from draco.analysis.ringmapmaker import find_grid_indices
from draco.core import containers

from ..core import io
from ..util import tools


class DelayFilterHyFoReSBandpassHybridVis(task.SingleTask):
    """Estimate bandpass gains and their window matrix from unfiltered hybrid vis.

    HyFoReS uses the unfiltered visibilities as estimated foregrounds and use the
    delay filtered visibilities as estimated signals. It cross correlates the two
    to estimate the bandpass errors in the postfiltered data.

    This implementation fixed the issue of noise bias in gain estimation.

    Attributes
    ----------
    atten_threshold : float
        Used by the DAYENU filter.
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    atten_threshold = config.Property(proptype=float, default=0.0)

    def setup(self, manager):
        """Extract the minimum baseline separation from the telescope class.

        Parameters
        ----------
        manager : io.TelescopeConvertible
            Telescope/manager used to extract the baseline distances
            to calculate the minimum separation in the north-south direction
            needed to compute aliases.
        """
        # Determine the layout of the visibilities on the grid.
        telescope = io.get_telescope(manager)
        _, _, _, min_ysep = find_grid_indices(telescope.baselines)

        # Save the minimum north-south separation
        self.min_ysep = min_ysep

    def process(self, hv, source):
        """Apply the DAYENU filter and then HyFoReS.

        Parameters
        ----------
        hv: containers.HybridVisStream
            The data the filter will be applied to.
        source: containers.HybridVisStream
            The filter of HybridVisStream to be applied.

        Returns
        -------
        gain_window: containers.VisBandpassWindow
            Estimated bandpass gains and their window matrix.
        """
        # First apply the DEYANU filter
        # Distribute over products
        hv.redistribute(["ra", "time"])
        source.redistribute(["ra", "time"])

        # Validate that both hybrid beamformed visibilites match
        if not np.array_equal(source.freq, hv.freq):
            raise ValueError("Frequencies do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["el"], hv.index_map["el"]):
            raise ValueError("Elevations do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["ew"], hv.index_map["ew"]):
            raise ValueError("EW baselines do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["pol"], hv.index_map["pol"]):
            raise ValueError("Polarisations do not match for hybrid visibilities.")

        if not np.array_equal(source.ra, hv.ra):
            raise ValueError("Right Ascension do not match for hybrid visibilities.")

        npol, nfreq, new, _, ntime = hv.vis.local_shape

        # Dereference the required datasets
        vis = hv.vis[:].local_array
        # Do not modify the weight of the original input
        weight = hv.weight[:].local_array.copy()
        filt = source.filter[:].local_array

        # create an empty dataset to store the post filtered vis.
        # Keeping vis as the unfiltered visibilities.
        post_vis = np.zeros_like(vis)

        # loop over products
        for tt in range(ntime):
            t0 = time.time()
            self.log.debug(f"Filter time {tt} of {ntime}.")

            # new stands for number east-west
            for xx in range(new):

                for pp in range(npol):

                    flag = weight[pp, :, xx, tt] > 0.0

                    if not np.any(flag):
                        continue

                    # Grab datasets for this pol and ew baseline
                    tvis = np.ascontiguousarray(vis[pp, :, xx, :, tt])

                    # Grab the filter for this pol and ew baseline
                    NF = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    # Make sure that any frequencies unmasked during filter generation
                    # are also unmasked in the data
                    valid_freq_flag = np.any(np.abs(NF) > 0.0, axis=0)

                    if not np.any(valid_freq_flag):
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    missing_freq = np.flatnonzero(valid_freq_flag & ~flag)
                    if missing_freq.size > 0:
                        self.log.warning(
                            "Missing the following frequencies that were "
                            "assumed valid during filter generation: "
                            f"{missing_freq}"
                        )
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    # Apply the filter
                    post_vis[pp, :, xx, :, tt] = np.matmul(NF, tvis)

                    # Flag frequencies with large attenuation
                    if self.atten_threshold > 0.0:
                        diag = np.abs(np.diag(NF))
                        nonzero_diag_flag = diag > 0.0
                        if np.any(nonzero_diag_flag):
                            med_diag = np.median(diag[nonzero_diag_flag])
                            flag_low = diag > (self.atten_threshold * med_diag)
                            weight[pp, :, xx, tt] *= flag_low.astype(
                                weight.dtype
                            )  # this masking is only done on the weight
                            # Now apply this masking to the filtered visibilities as well
                            atten_mask = flag_low.astype(vis.dtype)
                            post_vis[pp, :, xx, :, tt] *= atten_mask[:, np.newaxis]

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # Now estimate bandpass gains and their window
        # First create variable to store values

        yN = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # numerator of the estimated gains
        N = np.zeros(
            (npol, new, nfreq, nfreq), dtype=np.complex128
        )  # numerator of the window matrix
        D = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # denominator of both estimated gains and window matrix

        el_mask = self.aliased_el_mask(
            hv
        )  # generate a mask along the el axis to mask out aliased regions

        self.log.debug("Start computing the estimated gains.")
        t0 = time.time()
        for pp in range(npol):

            for ff in range(nfreq):

                for xx in range(new):
                    # grab datasets
                    # original data
                    tvis = np.ascontiguousarray(vis[pp, ff, xx, ...])
                    tpost_vis = np.ascontiguousarray(
                        post_vis[pp, ff, xx, ...]
                    )  # estimated signal

                    weight_mask = (
                        weight[pp, ff, xx, :] > 0.0
                    )  # post-filtered vis (estimated signal) weight
                    tpost_vis_masked = (
                        tpost_vis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                    )  # apply weight and liasing mask to estimated signal

                    # masked foreground template = original data - estimated signal
                    tfg_temp_mask = (
                        tvis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                        - tpost_vis_masked
                    )  #

                    yN[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tpost_vis_masked
                    )  # step 2 numerator
                    D[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tfg_temp_mask
                    )  # step 2 denominator

                self.log.debug(
                    f"Gains estimated for Polarization {pp} of {npol} and freq {ff} of {nfreq}."
                )
        self.log.debug(
            f"Gain estimation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        self.log.debug("Start computing the window.")
        t0 = time.time()
        for pp in range(npol):

            for xx in range(new):
                for tt in range(ntime):
                    # grab datasets
                    vis_f_l = np.ascontiguousarray(
                        vis[pp, :, xx, :, tt]
                    )  # original data
                    sg_f_l = np.ascontiguousarray(
                        post_vis[pp, :, xx, :, tt]
                    )  # estimated signal
                    filt_f_f = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    weight_mask_N = (
                        weight[pp, :, xx, tt] > 0.0
                    )  # this accounts for RFI, daytime, and other masks

                    # get estimated foreground template
                    fg_f_l = (
                        (vis_f_l - sg_f_l)
                        * weight_mask_N[:, np.newaxis]
                        * el_mask[np.newaxis, :]
                    )

                    N[pp, xx, :, :] += np.matmul(np.conj(fg_f_l), fg_f_l.T) * filt_f_f

                self.log.debug(
                    f"Window summed for Polarization {pp} of {npol} and East-West baseline {xx} of {new}."
                )

        self.log.debug(
            f"Window computation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        # create variables to reduce the partial sums
        yN_sum = np.zeros_like(yN)
        N_sum = np.zeros_like(N)
        D_sum = np.zeros_like(D)

        self.comm.Allreduce(yN, yN_sum, op=MPI.SUM)
        self.comm.Allreduce(N, N_sum, op=MPI.SUM)
        self.comm.Allreduce(D, D_sum, op=MPI.SUM)

        y = yN_sum * tools.invert_no_zero(D_sum)
        W = N_sum * tools.invert_no_zero(D_sum[:, :, :, np.newaxis])

        # put the estimated gains and window in a container
        bp_gain_win = containers.VisBandpassWindowBaseline(
            pol=hv.pol,
            ew=hv.ew,
            freq=hv.freq,
        )
        bp_gain_win.bandpass[:] = y
        bp_gain_win.window[:] = W

        return bp_gain_win

    def aliased_el_mask(self, hv):
        """Return a mask for the el axis to mask zenith angles beyond the aliased horizon.

        Computed using the maximum frequency in the hybrid beamformed visibilities.

        Parameters
        ----------
        hv: HybridVisStream
            Input container for which we will provide a aliased region mask along the el axis

        Returns
        -------
        mask: a mask to flag aliased regions along the el axis
        """
        freq = np.max(hv.freq)
        horizon_limit = self.get_horizon_limit(freq)
        el = hv.index_map["el"]

        # mask = np.abs(el) < horizon_limit
        return np.abs(el) < horizon_limit

    def get_horizon_limit(self, freq):
        """Calculate the value of sin(za) where the southern horizon aliases.

        Parameters
        ----------
        freq : np.ndarray[nfreq,]
            Frequency in MHz.

        Returns
        -------
        horizon_limit : np.ndarray[nfreq,]
            This is the value of sin(za) where the southern horizon aliases.
            Regions of sky where ``|sin(za)|`` is greater than or equal to
            this value will contain aliases.
        """
        return units.c / (freq * 1e6 * self.min_ysep) - 1.0


class DelayFilterHyFoReSBandpassHybridVisMask(DelayFilterHyFoReSBandpassHybridVis):
    """Same as DelayFilterHyFoReSBandpassHybridVis but adding a pixel mask.

    Attributes
    ----------
    atten_threshold : float
        Used by the DAYENU filter.
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    atten_threshold = config.Property(proptype=float, default=0.0)

    def process(self, hv, source, maskf):
        """Apply the pixel mask, DAYENU filter, and HyFoReS.

        Parameters
        ----------
        hv: containers.HybridVisStream
            The data the filter will be applied to.
        source: containers.HybridVisStream
            The filter of HybridVisStream to be applied.
        maskf: containers.RingMapMask
            A pixel track mask to mask out sidelobes that could bias gain estimation.

        Returns
        -------
        gain_window: containers.VisBandpassWindow
            Estimated bandpass gains and their window matrix.
        """
        # First apply the DEYANU filter
        # Distribute over products
        hv.redistribute(["ra", "time"])
        source.redistribute(["ra", "time"])
        maskf.redistribute(["ra", "time"])

        # Validate that both hybrid beamformed visibilites match
        if not np.array_equal(source.freq, hv.freq):
            raise ValueError("Frequencies do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["el"], hv.index_map["el"]):
            raise ValueError("Elevations do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["ew"], hv.index_map["ew"]):
            raise ValueError("EW baselines do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["pol"], hv.index_map["pol"]):
            raise ValueError("Polarisations do not match for hybrid visibilities.")

        if not np.array_equal(source.ra, hv.ra):
            raise ValueError("Right Ascension do not match for hybrid visibilities.")

        npol, nfreq, new, _, ntime = hv.vis.local_shape

        # Dereference the required datasets
        vis = hv.vis[:].local_array.copy()
        # Ringmap's RA and DEC are swapped
        mask = np.swapaxes(maskf.mask[:].local_array, -1, -2)
        # Do not modify the weight of the original input
        weight = hv.weight[:].local_array.copy()
        filt = source.filter[:].local_array

        # create an empty dataset to store the post filtered visibilities
        # Keeping vis as the unfiltered visibilities.
        post_vis = np.zeros_like(vis)

        # loop over products
        for tt in range(ntime):
            t0 = time.time()
            self.log.debug(f"Filter time {tt} of {ntime}.")

            # new stands for number east-west
            for xx in range(new):

                for pp in range(npol):

                    flag = weight[pp, :, xx, tt] > 0.0

                    if not np.any(flag):
                        continue

                    # Grab datasets for this pol and ew baseline
                    tvis = np.ascontiguousarray(vis[pp, :, xx, :, tt])

                    # Grab the filter for this pol and ew baseline
                    NF = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    # Make sure that any frequencies unmasked during filter generation
                    # are also unmasked in the data
                    valid_freq_flag = np.any(np.abs(NF) > 0.0, axis=0)

                    if not np.any(valid_freq_flag):
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    missing_freq = np.flatnonzero(valid_freq_flag & ~flag)
                    if missing_freq.size > 0:
                        self.log.warning(
                            "Missing the following frequencies that were "
                            "assumed valid during filter generation: "
                            f"{missing_freq}"
                        )
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    # Apply the filter
                    post_vis[pp, :, xx, :, tt] = np.matmul(NF, tvis)

                    # Flag frequencies with large attenuation
                    if self.atten_threshold > 0.0:
                        diag = np.abs(np.diag(NF))
                        nonzero_diag_flag = diag > 0.0
                        if np.any(nonzero_diag_flag):
                            med_diag = np.median(diag[nonzero_diag_flag])
                            flag_low = diag > (self.atten_threshold * med_diag)
                            weight[pp, :, xx, tt] *= flag_low.astype(
                                weight.dtype
                            )  # this masking is only done on the weight
                            # Now apply this masking to the filtered visibilities as well
                            atten_mask = flag_low.astype(vis.dtype)
                            post_vis[pp, :, xx, :, tt] *= atten_mask[:, np.newaxis]

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # Now compute the gains and their window
        # First create variable to store values

        yN = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # numerator of the estimated gains
        N = np.zeros(
            (npol, new, nfreq, nfreq), dtype=np.complex128
        )  # numerator of the window matrix
        D = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # denominator of both estimated gains and window matrix

        el_mask = self.aliased_el_mask(
            hv
        )  # generate a mask along the el axis to mask out aliased regions

        # Apply the pixel mask
        post_vis *= ~mask[:, :, np.newaxis, :, :]
        vis *= ~mask[:, :, np.newaxis, :, :]

        self.log.debug("Start computing the estimated gains.")
        t0 = time.time()
        for pp in range(npol):

            for ff in range(nfreq):

                for xx in range(new):
                    # grab datasets
                    tvis = np.ascontiguousarray(vis[pp, ff, xx, ...])  # original data
                    tpost_vis = np.ascontiguousarray(
                        post_vis[pp, ff, xx, ...]
                    )  # estimated signal

                    weight_mask = (
                        weight[pp, ff, xx, :] > 0.0
                    )  # post-filtered vis (estimated signal) weight
                    tpost_vis_masked = (
                        tpost_vis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                    )  # apply weight and liasing mask to estimated signal

                    # masked foreground template = original data - estimated signal
                    tfg_temp_mask = (
                        tvis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                        - tpost_vis_masked
                    )

                    yN[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tpost_vis_masked
                    )  # step 2 numerator
                    D[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tfg_temp_mask
                    )  # step 2 denominator

                self.log.debug(
                    f"Gains estimated for Polarization {pp} of {npol} and freq {ff} of {nfreq}."
                )
        self.log.debug(
            f"Gain estimation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        self.log.debug("Start computing the window.")
        t0 = time.time()
        for pp in range(npol):
            # step 3
            for xx in range(new):
                for tt in range(ntime):
                    # grab datasets
                    vis_f_l = np.ascontiguousarray(
                        vis[pp, :, xx, :, tt]
                    )  # original data
                    sg_f_l = np.ascontiguousarray(
                        post_vis[pp, :, xx, :, tt]
                    )  # estimated signal
                    filt_f_f = np.ascontiguousarray(filt[pp, :, :, xx, tt])  # filter

                    weight_mask_N = (
                        weight[pp, :, xx, tt] > 0.0
                    )  # this accounts for RFI, daytime, and other masks

                    # get estimated foreground template
                    fg_f_l = (
                        (vis_f_l - sg_f_l)
                        * weight_mask_N[:, np.newaxis]
                        * el_mask[np.newaxis, :]
                    )

                    N[pp, xx, :, :] += np.matmul(np.conj(fg_f_l), fg_f_l.T) * filt_f_f

                self.log.debug(
                    f"Window summed for Polarization {pp} of {npol} and East-West baseline {xx} of {new}."
                )

        self.log.debug(
            f"Window computation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        # create variables to reduce the partial sums
        yN_sum = np.zeros_like(yN)
        N_sum = np.zeros_like(N)
        D_sum = np.zeros_like(D)

        self.comm.Allreduce(yN, yN_sum, op=MPI.SUM)
        self.comm.Allreduce(N, N_sum, op=MPI.SUM)
        self.comm.Allreduce(D, D_sum, op=MPI.SUM)

        y = yN_sum * tools.invert_no_zero(D_sum)
        W = N_sum * tools.invert_no_zero(D_sum[:, :, :, np.newaxis])

        # put the estimated gains and window in a container
        bp_gain_win = containers.VisBandpassWindowBaseline(
            pol=hv.pol,
            ew=hv.ew,
            freq=hv.freq,
        )
        bp_gain_win.bandpass[:] = y
        bp_gain_win.window[:] = W

        return bp_gain_win


class HyFoReSBandpassHybridVis(DelayFilterHyFoReSBandpassHybridVis):
    """Same as DelayFilterHyFoReSBandpassHybridVis just without the Delay filter.

    It relies on an external step to perform the delay transform to get the estimated vis.

    Attributes
    ----------
    atten_threshold : float
        Used by the DAYENU filter.
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    atten_threshold = config.Property(proptype=float, default=0.0)

    def process(self, hv, pf_hv):
        """Uses HyFoReS to estimate the bandpass errors and their window matrix.

        Parameters
        ----------
        hv: containers.HybridVisStream
            Pre-filtered data.
        pf_hv: containers.HybridVisStream
            Post-filtered and crosstalk corrected data.

        Returns
        -------
        gain_window: containers.VisBandpassWindow
            Estimated bandpass gains and their window matrix.
        """
        # Distribute over products
        hv.redistribute(["ra", "time"])
        pf_hv.redistribute(["ra", "time"])

        npol, nfreq, new, _, ntime = hv.vis.local_shape

        # Dereference the required datasets
        vis = hv.vis[:].local_array.copy()
        post_vis = pf_hv.vis[:].local_array.copy()
        # weight of the post-filtered data
        weight = pf_hv.weight[:].local_array.copy()
        # TODO: consider the effect of crosstalk subtraction on the window
        filt = hv.filter[:].local_array.copy()

        yN = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # numerator of the estimated gains
        N = np.zeros(
            (npol, new, nfreq, nfreq), dtype=np.complex128
        )  # numerator of the window matrix
        D = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # denominator of both estimated gains and window matrix

        el_mask = self.aliased_el_mask(
            hv
        )  # generate a mask along the el axis to mask out aliased regions

        self.log.debug("Start computing the estimated gains.")
        t0 = time.time()
        for pp in range(npol):

            for ff in range(nfreq):

                for xx in range(new):
                    # grab datasets
                    tvis = np.ascontiguousarray(vis[pp, ff, xx, ...])  # original data
                    tpost_vis = np.ascontiguousarray(
                        post_vis[pp, ff, xx, ...]
                    )  # estimated signal

                    weight_mask = (
                        weight[pp, ff, xx, :] > 0.0
                    )  # post-filtered vis (estimated signal) weight
                    tpost_vis_masked = (
                        tpost_vis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                    )  # apply weight and liasing mask to estimated signal

                    # masked foreground template = original data - estimated signal
                    tfg_temp_mask = (
                        tvis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                        - tpost_vis_masked
                    )

                    yN[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tpost_vis_masked
                    )  # step 2 numerator
                    D[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tfg_temp_mask
                    )  # step 2 denominator

                self.log.debug(
                    f"Gains estimated for Polarization {pp} of {npol} and freq {ff} of {nfreq}."
                )
        self.log.debug(
            f"Gain estimation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        self.log.debug("Start computing the window.")
        t0 = time.time()
        for pp in range(npol):
            # step 3
            for xx in range(new):
                for tt in range(ntime):
                    # grab datasets
                    vis_f_l = np.ascontiguousarray(
                        vis[pp, :, xx, :, tt]
                    )  # original data
                    sg_f_l = np.ascontiguousarray(
                        post_vis[pp, :, xx, :, tt]
                    )  # estimated signal
                    filt_f_f = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    weight_mask_N = (
                        weight[pp, :, xx, tt] > 0.0
                    )  # this accounts for RFI, daytime, and other masks

                    # get estimated foreground template
                    fg_f_l = (
                        (vis_f_l - sg_f_l)
                        * weight_mask_N[:, np.newaxis]
                        * el_mask[np.newaxis, :]
                    )

                    N[pp, xx, :, :] += np.matmul(np.conj(fg_f_l), fg_f_l.T) * filt_f_f

                self.log.debug(
                    f"Window summed for Polarization {pp} of {npol} and East-West baseline {xx} of {new}."
                )

        self.log.debug(
            f"Window computation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        # create variables to reduce the partial sums
        yN_sum = np.zeros_like(yN)
        N_sum = np.zeros_like(N)
        D_sum = np.zeros_like(D)

        self.comm.Allreduce(yN, yN_sum, op=MPI.SUM)
        self.comm.Allreduce(N, N_sum, op=MPI.SUM)
        self.comm.Allreduce(D, D_sum, op=MPI.SUM)

        y = yN_sum * tools.invert_no_zero(D_sum)
        W = N_sum * tools.invert_no_zero(D_sum[:, :, :, np.newaxis])

        # put the estimated gains and window in a container
        bp_gain_win = containers.VisBandpassWindowBaseline(
            pol=hv.pol,
            ew=hv.ew,
            freq=hv.freq,
        )
        bp_gain_win.bandpass[:] = y
        bp_gain_win.window[:] = W

        return bp_gain_win


class HyFoReSBandpassHybridVisMask(DelayFilterHyFoReSBandpassHybridVis):
    """Same as DelayFilterHyFoReSBandpassHybridVisMask but without the Delay filter.

    It relies on an external step to perform the delay transform to get the estimated vis.

    Attributes
    ----------
    atten_threshold : float
        Used by the DAYENU filter.
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    atten_threshold = config.Property(proptype=float, default=0.0)

    def process(self, hv, pf_hv, maskf):
        """Uses HyFoReS to estimate the bandpass errors and their window matrix.

        Parameters
        ----------
        hv: containers.HybridVisStream
            Pre-filtered data.
        pf_hv: containers.HybridVisStream
            Post-filtered and crosstalk corrected data.
        maskf: containers.RingMapMask
            A pixel track mask to mask out sidelobes that could bias gain estimation.

        Returns
        -------
        gain_window: containers.VisBandpassWindow
            Estimated bandpass gains and their window matrix.
        """
        # Distribute over products
        hv.redistribute(["ra", "time"])
        pf_hv.redistribute(["ra", "time"])
        maskf.redistribute(["ra", "time"])

        npol, nfreq, new, _, ntime = hv.vis.local_shape

        # Dereference the required datasets
        vis = hv.vis[:].local_array.copy()
        # Ringmap's RA and DEC are swapped
        mask = np.swapaxes(maskf.mask[:].local_array, -1, -2)
        # Get filtered data
        post_vis = pf_hv.vis[:].local_array.copy()
        # weight of the post-filtered data
        weight = pf_hv.weight[:].local_array.copy()
        # TODO: consider the effect of crosstalk subtraction on the window
        filt = hv.filter[:].local_array.copy()

        yN = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # numerator of the estimated gains
        N = np.zeros(
            (npol, new, nfreq, nfreq), dtype=np.complex128
        )  # numerator of the window matrix
        D = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # denominator of both estimated gains and window matrix

        el_mask = self.aliased_el_mask(
            hv
        )  # generate a mask along the el axis to mask out aliased regions

        # Apply the pixel mask
        post_vis *= ~mask[:, :, np.newaxis, :, :]
        vis *= ~mask[:, :, np.newaxis, :, :]

        self.log.debug("Start computing the estimated gains.")
        t0 = time.time()
        for pp in range(npol):
            # step 2
            for ff in range(nfreq):

                for xx in range(new):
                    # grab datasets
                    tvis = np.ascontiguousarray(vis[pp, ff, xx, ...])  # original data
                    tpost_vis = np.ascontiguousarray(
                        post_vis[pp, ff, xx, ...]
                    )  # estimated signal

                    weight_mask = (
                        weight[pp, ff, xx, :] > 0.0
                    )  # post-filtered vis (estimated signal) weight
                    tpost_vis_masked = (
                        tpost_vis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                    )  # apply weight and liasing mask to estimated signal

                    # masked foreground template = original data - estimated signal
                    tfg_temp_mask = (
                        tvis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                        - tpost_vis_masked
                    )

                    yN[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tpost_vis_masked
                    )  # step 2 numerator
                    D[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tfg_temp_mask
                    )  # step 2 denominator

                self.log.debug(
                    f"Gains estimated for Polarization {pp} of {npol} and freq {ff} of {nfreq}."
                )
        self.log.debug(
            f"Gain estimation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        self.log.debug("Start computing the window.")
        t0 = time.time()
        for pp in range(npol):

            for xx in range(new):
                for tt in range(ntime):
                    # grab datasets
                    vis_f_l = np.ascontiguousarray(
                        vis[pp, :, xx, :, tt]
                    )  # original data
                    sg_f_l = np.ascontiguousarray(
                        post_vis[pp, :, xx, :, tt]
                    )  # estimated signal
                    filt_f_f = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    weight_mask_N = (
                        weight[pp, :, xx, tt] > 0.0
                    )  # this accounts for RFI, daytime, and other masks

                    # get estimated foreground template
                    fg_f_l = (
                        (vis_f_l - sg_f_l)
                        * weight_mask_N[:, np.newaxis]
                        * el_mask[np.newaxis, :]
                    )

                    N[pp, xx, :, :] += np.matmul(np.conj(fg_f_l), fg_f_l.T) * filt_f_f

                self.log.debug(
                    f"Window summed for Polarization {pp} of {npol} and East-West baseline {xx} of {new}."
                )

        self.log.debug(
            f"Window computation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        # create variables to reduce the partial sums
        yN_sum = np.zeros_like(yN)
        N_sum = np.zeros_like(N)
        D_sum = np.zeros_like(D)

        self.comm.Allreduce(yN, yN_sum, op=MPI.SUM)
        self.comm.Allreduce(N, N_sum, op=MPI.SUM)
        self.comm.Allreduce(D, D_sum, op=MPI.SUM)

        y = yN_sum * tools.invert_no_zero(D_sum)
        W = N_sum * tools.invert_no_zero(D_sum[:, :, :, np.newaxis])

        # put the estimated gains and window in a container
        bp_gain_win = containers.VisBandpassWindowBaseline(
            pol=hv.pol,
            ew=hv.ew,
            freq=hv.freq,
        )
        bp_gain_win.bandpass[:] = y
        bp_gain_win.window[:] = W

        return bp_gain_win


class HyFoReSBandpassHybridVisMaskKeepSource(DelayFilterHyFoReSBandpassHybridVis):
    """Same as HyFoReSBandpassHybridVisMask but uses a different pixel mask.

    The mask here only masks out sources' sidelobes but keeps their main lobes.

    It relies on an external step to perform the delay transform to get the estimated vis.

    Attributes
    ----------
    atten_threshold : float
        Used by the DAYENU filter.
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    atten_threshold = config.Property(proptype=float, default=0.0)

    def process(self, hv, pf_hv, maskf, masksf):
        """Uses HyFoReS to estimate the bandpass errors and their window matrix.

        Parameters
        ----------
        hv: containers.HybridVisStream
            Pre-filtered data.
        pf_hv: containers.HybridVisStream
            Post-filtered and crosstalk corrected data.
        maskf: containers.RingMapMask
            A pixel track mask to mask out sidelobes that could bias gain estimation.
        masksf: containers.RingMapMask
            A pixel transit mask to keep sources' main lobes.

        Returns
        -------
        gain_window: containers.VisBandpassWindow
            Estimated bandpass gains and their window matrix.
        """
        # Distribute over products
        hv.redistribute(["ra", "time"])
        pf_hv.redistribute(["ra", "time"])
        maskf.redistribute(["ra", "time"])
        masksf.redistribute(["ra", "time"])

        npol, nfreq, new, _, ntime = hv.vis.local_shape

        # Dereference the required datasets
        vis = hv.vis[:].local_array.copy()
        # Ringmap's RA and DEC are swapped
        mask = np.swapaxes(maskf.mask[:].local_array, -1, -2)
        masks = np.swapaxes(masksf.mask[:].local_array, -1, -2)
        # Get filtered data
        post_vis = pf_hv.vis[:].local_array.copy()
        # weight of the post-filtered data
        weight = pf_hv.weight[:].local_array.copy()
        # TODO: consider the effect of crosstalk subtraction on the window
        filt = hv.filter[:].local_array.copy()

        yN = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # numerator of the estimated gains
        N = np.zeros(
            (npol, new, nfreq, nfreq), dtype=np.complex128
        )  # numerator of the window matrix
        D = np.zeros(
            (npol, new, nfreq), dtype=np.complex128
        )  # denominator of both estimated gains and window matrix

        el_mask = self.aliased_el_mask(
            hv
        )  # generate a mask along the el axis to mask out aliased regions

        # Apply the pixel mask, keeping the main lobes but not the sidelobes
        post_vis *= ~np.logical_and(
            mask[:, :, np.newaxis, :, :], ~masks[:, :, np.newaxis, :, :]
        )
        vis *= ~np.logical_and(
            mask[:, :, np.newaxis, :, :], ~masks[:, :, np.newaxis, :, :]
        )

        self.log.debug("Start computing the estimated gains.")
        t0 = time.time()
        for pp in range(npol):

            for ff in range(nfreq):

                for xx in range(new):
                    # grab datasets
                    tvis = np.ascontiguousarray(vis[pp, ff, xx, ...])  # original data
                    tpost_vis = np.ascontiguousarray(
                        post_vis[pp, ff, xx, ...]
                    )  # estimated signal

                    weight_mask = (
                        weight[pp, ff, xx, :] > 0.0
                    )  # post-filtered vis (estimated signal) weight
                    tpost_vis_masked = (
                        tpost_vis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                    )  # apply weight and liasing mask to estimated signal

                    # masked foreground template = original data - estimated signal
                    tfg_temp_mask = (
                        tvis * weight_mask[np.newaxis, :] * el_mask[:, np.newaxis]
                        - tpost_vis_masked
                    )

                    yN[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tpost_vis_masked
                    )  # step 2 numerator
                    D[pp, xx, ff] = np.vdot(
                        tfg_temp_mask, tfg_temp_mask
                    )  # step 2 denominator

                self.log.debug(
                    f"Gains estimated for Polarization {pp} of {npol} and freq {ff} of {nfreq}."
                )
        self.log.debug(
            f"Gain estimation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        self.log.debug("Start computing the window.")
        t0 = time.time()
        for pp in range(npol):

            for xx in range(new):
                for tt in range(ntime):
                    # grab datasets
                    vis_f_l = np.ascontiguousarray(
                        vis[pp, :, xx, :, tt]
                    )  # original data
                    sg_f_l = np.ascontiguousarray(
                        post_vis[pp, :, xx, :, tt]
                    )  # estimated signal
                    filt_f_f = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    weight_mask_N = (
                        weight[pp, :, xx, tt] > 0.0
                    )  # this accounts for RFI, daytime, and other masks

                    # get estimated foreground template
                    fg_f_l = (
                        (vis_f_l - sg_f_l)
                        * weight_mask_N[:, np.newaxis]
                        * el_mask[np.newaxis, :]
                    )

                    N[pp, xx, :, :] += np.matmul(np.conj(fg_f_l), fg_f_l.T) * filt_f_f

                self.log.debug(
                    f"Window summed for Polarization {pp} of {npol} and East-West baseline {xx} of {new}."
                )

        self.log.debug(
            f"Window computation finished. Took {time.time() - t0:0.3f} seconds in total."
        )

        # create variables to reduce the partial sums
        yN_sum = np.zeros_like(yN)
        N_sum = np.zeros_like(N)
        D_sum = np.zeros_like(D)

        self.comm.Allreduce(yN, yN_sum, op=MPI.SUM)
        self.comm.Allreduce(N, N_sum, op=MPI.SUM)
        self.comm.Allreduce(D, D_sum, op=MPI.SUM)

        y = yN_sum * tools.invert_no_zero(D_sum)
        W = N_sum * tools.invert_no_zero(D_sum[:, :, :, np.newaxis])

        # put the estimated gains and window in a container
        bp_gain_win = containers.VisBandpassWindowBaseline(
            pol=hv.pol,
            ew=hv.ew,
            freq=hv.freq,
        )
        bp_gain_win.bandpass[:] = y
        bp_gain_win.window[:] = W

        return bp_gain_win


class DelayFilterHyFoReSBandpassHybridVisClean(task.SingleTask):
    """Compensates bandpass gain windows and subtracts foreground residuals.

    This task first compensates the bandpass window to obtain the unwindowed
    bandpass gains and then subtracts foreground residuals in the DAYENU-
    filtered visibilities.

    Attributes
    ----------
    cutoff : float
        Used by HyFoReS foreground residual subtraction.
        The cutoff for the singular value when pseudo-inverting
        the window matrix. Default is 1e-1.
        If the cutoff is 0.0, then do not compensate the window.
    atten_threshold : float
        Used by the DAYENU filter.
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    calculate_cov : bool
        Calculate the frequency-frequency noise covariance due to filtering.
    """

    cutoff = config.Property(proptype=float, default=1e-1)
    atten_threshold = config.Property(proptype=float, default=0.0)

    calculate_cov = config.Property(proptype=bool, default=False)

    def process(self, hv, source, bp):
        """Compensates the bandpass window and subtracts foreground residuals.

        Parameters
        ----------
        hv: containers.HybridVisStream
            The data the filter will be applied to.
        source: containers.HybridVisStream
            The filter of HybridVisStream to be applied.
        bp: containers.VisBandpassWindow
                Estimated bandpass gains and their window matrix from
                DelayFilterHyFoReSBandpassHybridVis.

        Returns
        -------
        vis: containers.HybridVisStream
            DAYENU-filtered hybrid beamformed visibilities with
            foreground residuals subtracted.
        comp_bandpass: containers.VisBandpassCompensate
                window-compensated gains along with the singular values
                of the window, rank of the inverted window, and cutoff
        """
        # Validate that both hybrid beamformed visibilites match
        if not np.array_equal(source.freq, hv.freq):
            raise ValueError("Frequencies do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["el"], hv.index_map["el"]):
            raise ValueError("Elevations do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["ew"], hv.index_map["ew"]):
            raise ValueError("EW baselines do not match for hybrid visibilities.")

        if not np.array_equal(source.index_map["pol"], hv.index_map["pol"]):
            raise ValueError("Polarisations do not match for hybrid visibilities.")

        if not np.array_equal(source.ra, hv.ra):
            raise ValueError("Right Ascension do not match for hybrid visibilities.")

        # If requested, add freq_cov dataset.
        if self.calculate_cov:
            if "complex_filter" in source.datasets:
                hv.add_dataset("complex_freq_cov")
            else:
                hv.add_dataset("freq_cov")
            hv.freq_cov[:] = 0.0

        # Distribute over products
        hv.redistribute(["ra", "time"])
        source.redistribute(["ra", "time"])

        npol, nfreq, new, _, ntime = hv.vis.local_shape

        # Compensate the window for the estimated bandpass gains
        y = bp.bandpass[:]
        W = bp.window[:]

        s_val = np.zeros((npol, new, nfreq))
        W_pinv = np.zeros_like(W)
        rank = np.zeros((npol, new))
        g = np.zeros_like(y)

        # if cutoff is 0.0, then do not compensate the window
        if self.cutoff == 0.0:
            g = y
            self.log.debug("Skip compensating the window")
        else:

            self.log.debug("Start compensating the window")

            for pp in range(npol):

                for xx in range(new):

                    # save the singular values for debugging or inspection
                    s_val[pp, xx] = la.svd(W[pp, xx, :, :], compute_uv=False)
                    # TODO: use la.solve(W, y)
                    W_pinv[pp, xx, :, :], rank[pp, xx] = la.pinv(
                        W[pp, xx, :, :], atol=self.cutoff, return_rank=True
                    )
                    g[pp, xx, :] = np.dot(W_pinv[pp, xx, :, :], y[pp, xx, :])

            self.log.debug("Gain window compensated")

        # Put unwinded bandpass gains in a container.
        # This container is not needed in later pipeline tasks.
        # Save for debugging purposes only.
        comp_bandpass = containers.VisBandpassCompensateBaseline(
            pol=hv.pol,
            ew=hv.ew,
            freq=hv.freq,
        )
        comp_bandpass.sval[:] = s_val
        comp_bandpass.comp_bandpass[:] = g
        comp_bandpass.attrs["rank"] = rank
        comp_bandpass.attrs["cutoff"] = self.cutoff

        # Apply the DAYENU filter
        vis = hv.vis[:].local_array
        weight = hv.weight[:].local_array
        filt = source.filter[:].local_array
        if self.calculate_cov:
            freq_cov = hv.freq_cov[:].local_array

        # loop over products
        for tt in range(ntime):
            t0 = time.time()
            self.log.debug(f"Filter time {tt} of {ntime}.")

            for xx in range(new):

                for pp in range(npol):

                    flag = weight[pp, :, xx, tt] > 0.0

                    if not np.any(flag):
                        continue

                    # Grab the filter for this pol and ew baseline
                    NF = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    # Make sure that any frequencies unmasked during filter generation
                    # are also unmasked in the data
                    valid_freq_flag = np.any(np.abs(NF) > 0.0, axis=0)

                    if not np.any(valid_freq_flag):
                        # Skip samples where filter is entirely zero
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    missing_freq = np.flatnonzero(valid_freq_flag & ~flag)
                    if missing_freq.size > 0:
                        self.log.warning(
                            "Missing the following frequencies that were "
                            "assumed valid during filter generation: "
                            f"{missing_freq}"
                        )
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    # Construct gain correction
                    diag_m = 1 - g[pp, xx, :]

                    # Apply the gain correction to visibilities and visibility variance
                    tvis = vis[pp, :, xx, :, tt] * diag_m[:, np.newaxis]
                    tvar = (
                        tools.invert_no_zero(weight[pp, :, xx, tt])
                        * np.abs(diag_m) ** 2
                    )

                    # Apply the filter
                    vis[pp, :, xx, :, tt] = np.matmul(NF, tvis)

                    # Apply filter squared to the variance and covariance
                    weight[pp, :, xx, tt] = tools.invert_no_zero(
                        np.matmul(np.abs(NF) ** 2, tvar)
                    )

                    if self.calculate_cov:
                        freq_cov[pp, :, :, xx, tt] = np.matmul(NF * tvar, NF.T.conj())

                    # Flag frequencies with large attenuation
                    if self.atten_threshold > 0.0:
                        diag = np.abs(np.diag(NF))
                        nonzero_diag_flag = diag > 0.0
                        if np.any(nonzero_diag_flag):
                            med_diag = np.median(diag[nonzero_diag_flag])
                            flag_low = diag > (self.atten_threshold * med_diag)
                            weight[pp, :, xx, tt] *= flag_low.astype(
                                weight.dtype
                            )  # This masking is only done on the weight

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # Problems saving to disk when distributed over last axis
        hv.redistribute("freq")

        return hv, comp_bandpass

import numpy as np
import ch_util
from ch_util import ephemeris
from astropy import constants as const
import h5py
from draco.core import containers
import pickle
from scipy import linalg as la
import os
import scipy
from draco.analysis.ringmapmaker import find_grid_indices

import time
import scipy.interpolate
from caput import config
from cora.util import units
from mpi4py import MPI

from ..core import io, task
from ..util import tools
from . import transform

##################################################################
# Comments:
# vis has the shape (pol, freq, ew, el, time)
# filter has the shape (pol, freq, freq, ew, time)
# vis_weight has the shape (pol, freq, ew, time)
# time is basically ra
#
# HyFoReS
#
# Pipeline task 1:
#
# inputs: vis, filter
#
# step 1: get estimated signal (apply delay filter to vis): loop over pol, ew, time: post_vis[pol, :, ew, :, time] = filter[pol, :, :, ew, time].dot(vis[pol, :, ew, :, time]) --> distributed time axes work independently
#
# step 2: estimate bandpass gains: 
#                               Both numerator and denominator are 0 if the freq channel is masked
#                               Numerator:   loop over pol, freq: yN[pol, freq] = vis[pol, freq,...].vdot(post_vis[pol, freq,...]) --> sum over distributed time axes
#                               Denominator: loop over pol, freq: yD[pol, freq] = vis[pol, freq,...].vdot(vis[pol, freq,...])      --> sum over distributed time axes
#                               Estimated g: y[:, :] = yN[:,:]/yN[:,:] 
#
# step 3: compute the window matrix:
#                               Numerator: loop over pol N[pol, :, :] = sum over ew, time: vis[pol,:, ew, :, time].conj().dot(vis[pol,:, ew, :, time].T())*filter[pol,:,:,ew, time] # the last product is element-wise product --> sum over distributed time axes
#                                                  # this can be computed under the pol loop of the previous step
#                               Dominator: loop over pol, freq_1: D[pol, freq_1] = vis[pol, freq_1, ...].vdot(vis[pol, freq_1, ...]) --> sum over distributed time axes # note this is the same as yD above, do so no need to compute again
#                               Window:    loop over pol, freq_1: W[pol,freq_1,:] = N[pol, freq_1, :]/D[pol, freq_1]
#
# outputs: the estimated gains and window (new container required)
#
#
# Pipeline task 2:
#
# inputs: vis, filter, estimate gains and window
#
# step 4: pseudo-invert the window:
#                               Singular values:  loop over pol: u[pol], s_val[pol], vh[pol] = LA.svd(W[pol,:,:]) # need the singular value to determine the pseduo-inverse cut off: cut_off = 3e-2 by default
#                               Pseudo-inverse:   loop over pol: W_pinv[pol,:,:], rank[pol] = LA.pinv(W[pol,:,:], atol = cut_off, return_rank = True) # return rank should be Nfreq - Nfreq_mask - Nfreq_delay_cut
#                               Unwindowed gains: loop over pol: g[pol, :] = W_pinv[pol,:,:].dot(y[pol, :])
# 
# step 5: subtract foregrounds:
#                               cleaned vis: loop over pol, ew, time: vis_cleaned[pol, :, ew, :, time] = post_vis[pol, :, ew, :, time] - filter[pol, :, :, ew, time].dot(np.diag(g[pol,:]).dot(vis[pol, :, ew, :, time])) --> distributed time axes work independently
#                               # post_vis[pol, :, ew, :, time] is computed the same way as step 1
#
# output: cleaned vis
#
# Required Draco containers: VisBandpassWindowBaseline, VisBandpassCompensateBaseline
#
###################################################################


class DelayFilterHyFoReSBandpassHybridVis(task.SingleTask): 
    """Apply HyFoReS to estimate bandpass gains and compute their window matrix from unfiltered hybrid beam formed visibilities. 

    This task builds on ApplyDelayFilterHybridVis. HyFoReS uses the unfiltered visibilities 
    as estimated foregrounds and use the delay filtered visibilities as estimated signals. It
    cross correlates the two to estimate the bandpass errors in the postfiltered data.

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
        xind, yind, min_xsep, min_ysep = find_grid_indices(telescope.baselines)

        # Save the minimum north-south separation
        self.min_ysep = min_ysep
    
    def process(self, hv, source):
        """First apply the DAYENU filter to a HybridVisStream.
           Then use HyFoReS to estimate the bandpass errors and compute their window matrix

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
        hv.redistribute(["ra", "time"])   ###Q1:so this is how to distribute over nodes? what is the difference between the first and second element (ra vs time)?
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

        npol, nfreq, new, nel, ntime = hv.vis.local_shape ###Q2:what is local_shape? Shape of the data distributed to a node, not the total shape?

        # Dereference the required datasets ### N:dereference means accessing the stored value where a pointer is pointing to. #Q3:why dereferencing and why is a pointer involved? A: to make operations faster and compatible with MPI
        vis = hv.vis[:].local_array 
        weight = hv.weight[:].local_array.copy() # Do not modify the weight of the original input. Modify its copy.
        filt = source.filter[:].local_array

        # create an empty dataset to store the post filtered visibilities. Keeping vis as the unfiltered visibilities.
        post_vis = np.zeros_like(vis)

        # loop over products
        for tt in range(ntime):   #Q4:why looping over the last index first? what if I don't follow this order in my own code? #A:looking at the loops alone, this makes it slower
            t0 = time.time()
            self.log.debug(f"Filter time {tt} of {ntime}.")

            for xx in range(new):

                for pp in range(npol):

                    flag = weight[pp, :, xx, tt] > 0.0   ### N:so this is how to tell a frequency is flagged or not

                    # N:Skip fully masked samples ### no frequency is available --> skip (cond 1/3)
                    if not np.any(flag):
                        continue

                    # Grab datasets for this pol and ew baseline
                    tvis = np.ascontiguousarray(vis[pp, :, xx, :, tt]) ### N:ensures an array is stored in contiguous memory, following the C-order (row-major) layout. This is to make sure looping over the column index is fast.
                    tvar = tools.invert_no_zero(weight[pp, :, xx, tt]) ### inverse of the weight is variance (keep zeros zeros). Looks like the variance is taken over DEC or el.

                    # Grab the filter for this pol and ew baseline
                    NF = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    # Make sure that any frequencies unmasked during filter generation
                    # are also unmasked in the data
                    valid_freq_flag = np.any(np.abs(NF) > 0.0, axis=0)

                    if not np.any(valid_freq_flag):   ### if the filter is entirely zero --> skip (cond 2/3)
                        # Skip samples where filter is entirely zero 
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    missing_freq = np.flatnonzero(valid_freq_flag & ~flag) ### find the indices of frequencies needed by the delay filter but missing from the unfiltered data --> skip (cond 3/3)
                    if missing_freq.size > 0:
                        self.log.warning(
                            "Missing the following frequencies that were "
                            "assumed valid during filter generation: "
                            f"{missing_freq}"
                        )
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    # Apply the filter
                    post_vis[pp, :, xx, :, tt] = np.matmul(NF, tvis) ### compute using contiguous filter and visibility data ### this must have changed hv.vis
                    # Do not care about the weight for now
                    #weight[pp, :, xx, tt] = tools.invert_no_zero( ### variance multiples with the filter value squared
                        #np.matmul(np.abs(NF) ** 2, tvar)
                    #)
                    # Flag frequencies with large attenuation ### mask frequencies where in the filtered data they do not have enough same-frequency contributions from the unfiltered data
                    if self.atten_threshold > 0.0:
                        diag = np.abs(np.diag(NF))
                        nonzero_diag_flag = diag > 0.0
                        if np.any(nonzero_diag_flag):
                            med_diag = np.median(diag[nonzero_diag_flag])
                            flag_low = diag > (self.atten_threshold * med_diag)
                            weight[pp, :, xx, tt] *= flag_low.astype(weight.dtype) ### this masking is only done on the weight
                            # Now apply this masking to the filtered visibilities as well
                            atten_mask = flag_low.astype(vis.dtype)
                            post_vis[pp, :, xx, :, tt] *= atten_mask[:, np.newaxis]

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # Now implement HyFoReS step 2 and 3 in the comments
        # First create variable to store values

        yN = np.zeros((npol, new, nfreq), dtype = np.complex128) # numerator of the estimated gains
        N = np.zeros((npol, new, nfreq, nfreq), dtype = np.complex128) # numerator of the window matrix
        D = np.zeros((npol, new, nfreq), dtype = np.complex128) # denominator of both estimated gains and window matrix

        el_mask = self.aliased_el_mask(hv) # generate a mask along the el axis to mask out aliased regions
        
        self.log.debug(f"Start computing the estimated gains.")
        t0 = time.time()
        for pp in range(npol):
                # step 2
            for ff in range(nfreq):
            
                for xx in range(new):
                    # grab datasets
                    tvis = np.ascontiguousarray(vis[pp, ff, xx, ...]) # ensures an array is stored in contiguous memory, following the C-order (row-major) layout #Q5:does this work for arrays with > 2 dimensions? Do I need to do this? # not necessary here
                    tpost_vis = np.ascontiguousarray(post_vis[pp, ff, xx, ...])

                    weight_mask = weight[pp, ff, xx, :] > 0.0 # this is technically the post-filtered vis weight, which should be what we want. # this accounts for RFI, daytime, and other masks # does not account for aliasing
                    tvis_masked = tvis*weight_mask[np.newaxis,:]*el_mask[:, np.newaxis] # applying the aliasing mask as well
                    tpost_vis_masked = tpost_vis*weight_mask[np.newaxis,:]*el_mask[:, np.newaxis]

                    yN[pp, xx, ff] = np.vdot(tvis_masked,tpost_vis_masked) # step 2 numerator
                    D[pp, xx, ff] = np.vdot(tvis_masked,tvis_masked) # step 2 denominator
    
                self.log.debug(f"Gains estimated for Polarization {pp} of {npol} and freq {ff} of {nfreq}.")
        self.log.debug(f"Gain estimation finished. Took {time.time() - t0:0.3f} seconds in total.")


        self.log.debug(f"Start computing the window.")
        t0 = time.time()
        for pp in range(npol):
            # step 3
            for xx in range(new):
                for tt in range(ntime):
                    # grab datasets
                    vis_f_l = np.ascontiguousarray(vis[pp,:, xx, :, tt])
                    filt_f_f = np.ascontiguousarray(filt[pp,:,:,xx, tt])

                    weight_mask_N = weight[pp, :, xx, tt] > 0.0 # this accounts for RFI, daytime, and other masks # does not account for aliasing
                    vis_f_l_masked = vis_f_l*weight_mask_N[:, np.newaxis]*el_mask[np.newaxis, :] # apply the aliasing mask

                    N[pp, xx, :, :] += np.matmul(np.conj(vis_f_l_masked), vis_f_l_masked.T)*filt_f_f
                    # the line above does vis[pp,:, xx, :, tt].conj().dot(vis[pp,:, xx, :, tt].T())*filt[pp,:,:,xx, tt]

                self.log.debug(f"Window summed for Polarization {pp} of {npol} and East-West baseline {xx} of {new}.")

        self.log.debug(f"Window computation finished. Took {time.time() - t0:0.3f} seconds in total.")

        # create variables to reduce the partial sums
        yN_sum = np.zeros_like(yN)
        N_sum = np.zeros_like(N)
        D_sum = np.zeros_like(D)

        self.comm.Allreduce(yN, yN_sum, op=MPI.SUM)
        self.comm.Allreduce(N, N_sum, op=MPI.SUM)
        self.comm.Allreduce(D, D_sum, op=MPI.SUM)

        y = np.zeros_like(yN_sum)
        y = yN_sum*tools.invert_no_zero(D_sum)
        W = np.zeros_like(N_sum)
        W = N_sum*tools.invert_no_zero(D_sum[:, :, :, np.newaxis])

        # put the estimated gains and window in a container
        bp_gain_win = containers.VisBandpassWindowBaseline(
                pol= hv.pol,
                ew = hv.ew,
                freq= hv.freq,
                )
        bp_gain_win.bandpass[...] = y # new question: do I need [...] here? should check out other examples
        bp_gain_win.window[...] = W

        return bp_gain_win

    def aliased_el_mask(self, hv):
        """Return a mask for the el axis to mask out zenith angles beyond the aliased horizon
        Computed using the maximum frequency in the hybrid beamformed visibilities.

        Parameters
        ----------
        hv: HybridVisStream
        Input container for which we want to provide a aliased region mask along the el axis

        Returns
        -------
        mask: a mask to flag aliased regions along the el axis
        """

        freq = np.max(hv.freq)
        horizon_limit = self.get_horizon_limit(freq)
        el = hv.index_map["el"]
        mask = np.abs(el) < horizon_limit

        return mask


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
        
        return scipy.constants.c / (freq * 1e6 * self.min_ysep) - 1.0


# Add the option to compensate the window or not.

class DelayFilterHyFoReSBandpassHybridVisClean(task.SingleTask):
    """ Second pipeline task of HyFoReS: Subtract foreground residuals using the estimated bandpass gains .

    This task first compensates the bandpass window to obtain the unwindowed
    bandpass gains and then subtracts foreground residuals in the DAYENU-
    filtered visibilities.

    Attributes
    ----------
    cutoff : float
        Used by HyFoReS foreground residual subtraction.
        The cutoff for the singular value when pseudo-inverting 
        the window matrix. Default is 3e-2.
    atten_threshold : float
        Used by the DAYENU filter.
        Mask any frequency where the diagonal element of the filter
        is less than this fraction of the median value over all
        unmasked frequencies.  Default is 0.0 (i.e., do not mask
        frequencies with low attenuation).
    """

    # if the cutoff is 0.0, then do not compensate the window
    cutoff = config.Property(proptype=float, default=1e-1)
    atten_threshold = config.Property(proptype=float, default=0.0)

    def process(self, hv, source, bp):
        """First compensate for the bandpass estimate window
           Then subtract foreground residuals in the NAYENU-filtered
           visibilities due to bandpass gains.

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

        npol, nfreq, new, nel, ntime = hv.vis.local_shape ### Q:what is local_shape? Shape of the data distributed to a node, not the total shape?

        # Step 4
        # step 4: pseudo-invert the window:
        # Singular values:  loop over pol: u[pol], s_val[pol], vh[pol] = LA.svd(W[pol,:,:]) # need the singular value to determine the pseduo-inverse cut off: cut_off = 3e-2 by default
        # Pseudo-inverse:   loop over pol: W_pinv[pol,:,:], rank[pol] = LA.pinv(W[pol,:,:], atol = cut_off, return_rank = True) # return rank should be Nfreq - Nfreq_mask - Nfreq_delay_cut
        # Unwindowed gains: loop over pol: g[pol, :] = W_pinv[pol,:,:].dot(y[pol, :])
        # Compensate the window for the estimated bandpass gains
        y = bp.bandpass[...]
        W = bp.window[...]

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

                    u, s_val[pp, xx], vh = la.svd(W[pp, xx, :,:]) #Q6: is scipy.linalg imported as LA somewhere in the pipeline? #la lower case # create new module in draco analysis
                    W_pinv[pp, xx, :, :], rank[pp, xx] = la.pinv(W[pp, xx, :, :], atol = self.cutoff, return_rank = True)
                    g[pp, xx, :] = np.dot(W_pinv[pp, xx, :, :], y[pp, xx, :])

            self.log.debug("Gain window compensated")
        
        # put unwinded bandpass gains in a container. This container is not needed in later pipeline tasks. Save for debug purposes only.
        comp_bandpass = containers.VisBandpassCompensateBaseline(
                pol= hv.pol,
                ew = hv.ew,
                freq= hv.freq,
                )
        comp_bandpass.sval[...] = s_val
        comp_bandpass.comp_bandpass[...] = g
        comp_bandpass.attrs['rank'] = rank
        comp_bandpass.attrs['cutoff'] = self.cutoff
        

        # Step 5
		# get estimated signal (apply delay filter to vis): loop over pol, ew, time: post_vis[pol, :, ew, :, time] = filter[pol, :, :, ew, time].dot(vis[pol, :, ew, :, time]) --> distributed time axes work independently
		# cleaned vis: loop over pol, ew, time: vis_cleaned[pol, :, ew, :, time] = post_vis[pol, :, ew, :, time] - filter[pol, :, :, ew, time].dot(np.diag(g[pol,:]).dot(vis[pol, :, ew, :, time])) --> distributed time axes work independently
		# two lines together: vis_cleaned[pol, :, ew, :, time] = np.dot(filter[pol, :, :, ew, time], np.dot(np.eye(nfreq) - np.diag(np.real(g[pol,:])), vis[pol, :, ew, :, time]))
        # Dereference the required datasets ### N:dereference means accessing the stored value where a pointer is pointing to. Q:why is a pointer involved?
        
        # the following code will modify both vis and weight (foreground filtering and gain corrections)
        vis = hv.vis[:].local_array
        weight = hv.weight[:].local_array
        filt = source.filter[:].local_array

        # loop over products
        for tt in range(ntime):  
            t0 = time.time()
            self.log.debug(f"Filter time {tt} of {ntime}.")

            for xx in range(new):

                for pp in range(npol):

                    flag = weight[pp, :, xx, tt] > 0.0   ### N:so this is how to tell is a frequency is flagged or not

                    # N:Skip fully masked samples ### no frequency is available --> skip (cond 1/3)
                    if not np.any(flag):
                        continue

                    # Grab datasets for this pol and ew baseline
                    tvis = np.ascontiguousarray(vis[pp, :, xx, :, tt]) ### N:ensures an array is stored in contiguous memory, following the C-order (row-major) layout. This is to make sure looping over the column index is fast.
                    tvar = tools.invert_no_zero(weight[pp, :, xx, tt]) ### inverse of the weight is variance (keep zeros zeros). Looks like the variance is taken over DEC or el.

                    # Grab the filter for this pol and ew baseline
                    NF = np.ascontiguousarray(filt[pp, :, :, xx, tt])

                    # Make sure that any frequencies unmasked during filter generation
                    # are also unmasked in the data
                    valid_freq_flag = np.any(np.abs(NF) > 0.0, axis=0)

                    if not np.any(valid_freq_flag):   ### if the filter is entirely zero --> skip (cond 2/3)
                        # Skip samples where filter is entirely zero 
                        weight[pp, :, xx, tt] = 0.0
                        continue

                    missing_freq = np.flatnonzero(valid_freq_flag & ~flag) ### find the indices of frequencies needed by the delay filter but missing from the unfiltered data --> skip (cond 3/3)
                    if missing_freq.size > 0:
                        self.log.warning(
                            "Missing the following frequencies that were "
                            "assumed valid during filter generation: "
                            f"{missing_freq}"
                        )
                        weight[pp, :, xx, tt] = 0.0
                        continue
                    
                    # Implement Step 5 
                    diag_m = 1 - g[pp, xx, :]
                    vis[pp, :, xx, :, tt] = np.matmul(NF, diag_m[:, np.newaxis]*tvis) ### compute using contiguous filter and visibility data ### this must have changed hv.vis
                    weight[pp, :, xx, tt] = tools.invert_no_zero( ### variance multiples with the filter value squared
                        np.matmul(np.abs(NF) ** 2, (np.abs(diag_m[:])**2)*tvar))
                    # Flag frequencies with large attenuation ### mask frequencies where in the filtered data they do not have enough same-frequency contributions from the unfiltered data
                    if self.atten_threshold > 0.0:
                        diag = np.abs(np.diag(NF))
                        nonzero_diag_flag = diag > 0.0
                        if np.any(nonzero_diag_flag):
                            med_diag = np.median(diag[nonzero_diag_flag])
                            flag_low = diag > (self.atten_threshold * med_diag)
                            weight[pp, :, xx, tt] *= flag_low.astype(weight.dtype) ### this masking is only done on the weight

            self.log.debug(f"Took {time.time() - t0:0.3f} seconds in total.")

        # Problems saving to disk when distributed over last axis
        hv.redistribute("freq") ### why redistribute again? How does this work when the data are aready distributed over the ra axis on different nodes?

        return hv, comp_bandpass

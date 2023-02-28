"""Sensitivity Analysis Tasks."""

import numpy as np

from caput import config

from ..core import task, io, containers
from ..util import tools


class ComputeSystemSensitivity(task.SingleTask):
    """Compute the sensitivity of beamformed visibilities.

    Parameters
    ----------
    exclude_intracyl : bool
        Exclude the intracylinder baselines in the sensitivity estimate.
        Default is to use all baselines.  Note that a RuntimeError
        will be raised if exclude_intracyl is True and the visibilities
        have already been stacked over cylinder.
    """

    exclude_intracyl = config.Property(proptype=bool, default=False)

    def setup(self, telescope):
        """Save the telescope model.

        Parameters
        ----------
        telescope : TransitTelescope
            Telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, data):
        """Estimate the sensitivity of the input data.

        Parameters
        ----------
        data : TODContainer
            Must have a weight property that contains an
            estimate of the inverse variance of the noise
            in each visibility.  The visibilities can be
            stacked to any level of redundancy.

        Returns
        -------
        metrics : SystemSensitivity
            Contains the measured and radiometric estimates of
            the noise in the beamformed visibilities.
        """
        # Ensure we are distributed over frequency. Get shape of visibilities.
        data.redistribute("freq")

        nfreq, nstack, ntime = data.vis.local_shape

        # Extract the input flags.  If container has a gain dataset,
        # then also check for the default gain 1.0 + 0.0j as this indicates
        # that an input was masked for a particular time and frequency.
        inpflg = data.input_flags[:].view(np.ndarray).astype(bool)
        niff = 1

        if "gain" in data.datasets:
            # Derive frequency dependent flags from gains
            gainflg = data.gain[:].view(np.ndarray) != (1.0 + 0.0j)
            inpflg = np.swapaxes(inpflg[np.newaxis, :, :] & gainflg, 0, 1)
            # Flatten frequency and time axis so we can use numpy's unique
            inpflg = inpflg.reshape(inpflg.shape[0], -1)
            niff = nfreq

        # Find unique sets of input flags
        uniq_inpflg, index_cnt = np.unique(inpflg, return_inverse=True, axis=1)

        # Calculate redundancy for each unique set of input flags
        cnt = tools.calculate_redundancy(
            uniq_inpflg.astype(np.float32),
            data.prod,
            data.reverse_map["stack"]["stack"],
            data.stack.size,
        )

        # Determine stack axis
        stack_new, stack_flag = tools.redefine_stack_index_map(
            self.telescope, data.input, data.prod, data.stack, data.reverse_map["stack"]
        )

        if not np.all(stack_flag):
            self.log.warning(
                "There are %d stacked baselines that are masked "
                "in the telescope instance." % np.sum(~stack_flag)
            )

        ps = data.prod[stack_new["prod"]]
        conj = stack_new["conjugate"]

        prodstack = ps.copy()
        prodstack["input_a"] = np.where(conj, ps["input_b"], ps["input_a"])
        prodstack["input_b"] = np.where(conj, ps["input_a"], ps["input_b"])

        # Figure out mapping between inputs in data file and inputs in telescope
        tel_index = tools.find_inputs(
            self.telescope.input_index, data.input, require_match=False
        )

        # Use the mapping to extract polarisation and EW position of each input
        input_pol = np.array(
            [
                self.telescope.polarisation[ti] if ti is not None else "N"
                for ti in tel_index
            ]
        )

        ew_position = np.array(
            [
                self.telescope.feedpositions[ti, 0] if ti is not None else 0.0
                for ti in tel_index
            ]
        )

        # Next we determine indices into the stack axis for each polarisation product
        # The next three lines result in XY and YX being
        # combined into a single polarisation product
        pa, pb = input_pol[prodstack["input_a"]], input_pol[prodstack["input_b"]]
        pol_a = np.where(pa <= pb, pa, pb)
        pol_b = np.where(pa <= pb, pb, pa)

        baseline_pol = np.core.defchararray.add(pol_a, pol_b)

        if self.exclude_intracyl:
            baseline_flag = (
                ew_position[prodstack["input_a"]] != ew_position[prodstack["input_b"]]
            )
        else:
            baseline_flag = np.ones(prodstack.size, dtype=bool)

        pol_uniq = [bp for bp in np.unique(baseline_pol) if "N" not in bp]
        pol_index = [
            np.flatnonzero((baseline_pol == up) & baseline_flag) for up in pol_uniq
        ]
        npol = len(pol_uniq)

        auto_flag = (prodstack["input_a"] == prodstack["input_b"]).astype(np.float32)

        if self.exclude_intracyl and (np.sum(auto_flag) == npol):
            raise ValueError(
                "You have requested the exclusion of "
                "intracylinder baselines,  however it appears "
                "that the visibilities have already been stacked "
                "over cylinder, preventing calculation of the "
                "radiometric estimate."
            )

        # Dereference the weight dataset
        bweight = data.weight[:].view(np.ndarray)
        bflag = bweight > 0.0

        # Initialize arrays
        var = np.zeros((nfreq, npol, ntime), dtype=np.float32)
        counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

        # Average over selected baseline per polarization
        for pp, ipol in enumerate(pol_index):
            pcnt = cnt[ipol, :]
            pscale = 2.0 - auto_flag[ipol, np.newaxis]

            # Loop over frequencies to reduce memory usage
            for ff in range(nfreq):
                fslc = slice((ff % niff) * ntime, ((ff % niff) + 1) * ntime)
                pfcnt = pcnt[:, index_cnt[fslc]]

                pvar = tools.invert_no_zero(bweight[ff, ipol, :])
                pflag = bflag[ff, ipol, :].astype(np.float32)

                var[ff, pp, :] = np.sum(pfcnt**2 * pscale * pflag * pvar, axis=0)

                counter[ff, pp, :] = np.sum(pfcnt * pscale * pflag, axis=0)

        # Normalize
        var *= tools.invert_no_zero(counter**2)

        # Determine which of the stack indices correspond to autocorrelations
        auto_stack_id = np.flatnonzero(auto_flag)
        auto_input = prodstack["input_a"][auto_stack_id]
        auto_pol = input_pol[auto_input]

        auto_cnt = cnt[auto_stack_id, :][:, index_cnt]
        auto_cnt = np.swapaxes(auto_cnt.reshape(-1, niff, ntime), 0, 1)
        num_feed = auto_cnt * bflag[:, auto_stack_id, :].astype(np.float32)

        auto = data.vis[:, auto_stack_id, :].local_array.real

        # Construct the radiometric estimate of the noise by taking the sum
        # of the product of pairs of (possibly stacked) autocorrelations.
        radiometer = np.zeros((nfreq, npol, ntime), dtype=np.float32)
        radiometer_counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

        for ii, (ai, pi) in enumerate(zip(auto_input, auto_pol)):
            for jj, (aj, pj) in enumerate(zip(auto_input, auto_pol)):
                if self.exclude_intracyl and (ew_position[ai] == ew_position[aj]):
                    # Exclude intracylinder baselines
                    continue

                # Combine XY and YX into single polarisation product
                pp = pol_uniq.index(pi + pj) if pi <= pj else pol_uniq.index(pj + pi)

                # Weight by the number of feeds that were averaged
                # together to obtain each stacked autocorrelation
                nsq = num_feed[:, ii, :] * num_feed[:, jj, :]

                radiometer[:, pp, :] += nsq * auto[:, ii, :] * auto[:, jj, :]

                radiometer_counter[:, pp, :] += nsq

        # Calculate number of independent samples from the
        # integration time, frequency resolution, and fraction of packets lost
        tint = np.median(np.abs(np.diff(data.time)))
        dnu = np.median(data.index_map["freq"]["width"]) * 1e6

        if ("flags" in data) and ("frac_lost" in data["flags"]):
            frac_lost = data["flags"]["frac_lost"][:].local_array
        else:
            frac_lost = np.zeros((1, 1), dtype=np.float32)

        nint = dnu * tint * (1.0 - frac_lost[:, np.newaxis, :])

        # Normalize by the number of independent samples
        # and the total number of baselines squared
        radiometer *= tools.invert_no_zero(nint * radiometer_counter**2)

        # Create output container
        metrics = containers.SystemSensitivity(
            pol=np.array(pol_uniq, dtype="<U2"),
            axes_from=data,
            attrs_from=data,
            comm=data.comm,
            distributed=data.distributed,
        )

        metrics.redistribute("freq")

        # In order to write generic code for generating the radiometric
        # estimate of the sensitivity, we had to sum over the upper and lower triangle
        # of the visibility matrix.  Below we multiply by sqrt(2) in order to
        # obtain the sensitivity of the real component.
        metrics.radiometer[:] = np.sqrt(2.0 * radiometer)
        metrics.measured[:] = np.sqrt(2.0 * var)

        # Save the total number of baselines that were averaged in the weight dataset
        metrics.weight[:] = counter

        # Save the fraction of missing samples
        metrics.frac_lost[:] = frac_lost

        return metrics

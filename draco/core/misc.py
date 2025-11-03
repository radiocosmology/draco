"""Miscellaneous pipeline tasks with no where better to go.

Tasks should be proactively moved out of here when there is a thematically
appropriate module, or enough related tasks end up in here such that they can
all be moved out into their own module.
"""

import os

import h5py
import numpy as np
from caput import config, mpiutil, weighted_median

from ..core import containers, io, task
from ..util import tools


class ApplyGain(task.SingleTask):
    """Apply a set of gains to a timestream or sidereal stack.

    Attributes
    ----------
    inverse : bool, optional
        Apply the gains directly, or their inverse.
    update_weight : bool, optional
        Scale the weight array with the updated gains.
    smoothing_length : float, optional
        Smooth the gain timestream across the given number of seconds.
        Not supported (ignored) for Sidereal Streams.
    """

    inverse = config.Property(proptype=bool, default=True)
    update_weight = config.Property(proptype=bool, default=False)
    smoothing_length = config.Property(proptype=float, default=None)

    def process(self, tstream, gain):
        """Apply gains to the given timestream.

        Smoothing the gains is not supported for SiderealStreams.

        Parameters
        ----------
        tstream : TimeStream like or SiderealStream
            Time stream to apply gains to. The gains are applied in place.
        gain : StaticGainData, GainData, SiderealGainData, CommonModeGainData
            or CommonModeSiderealGainData. Gains to apply.

        Returns
        -------
        tstream : TimeStream or SiderealStream
            The timestream with the gains applied.
        """
        tstream.redistribute("freq")
        gain.redistribute("freq")

        if tstream.is_stacked and not isinstance(
            gain, containers.CommonModeGainData | containers.CommonModeSiderealGainData
        ):
            raise ValueError(
                f"Cannot apply input-dependent gains to stacked data: {tstream!s}"
            )

        if isinstance(gain, containers.StaticGainData):
            # Extract gain array and add in a time axis
            gain_arr = gain.gain[:][..., np.newaxis]

            # Get the weight array if it's there
            weight_arr = (
                gain.weight[:][..., np.newaxis] if gain.weight is not None else None
            )

        elif isinstance(
            gain,
            containers.GainData
            | containers.SiderealGainData
            | containers.CommonModeGainData
            | containers.CommonModeSiderealGainData,
        ):
            # Extract gain array
            gain_arr = gain.gain[:]

            # Regularise any crazy entries
            gain_arr = np.nan_to_num(gain_arr)

            # Get the weight array if it's there
            weight_arr = gain.weight[:] if gain.weight is not None else None

            if isinstance(
                gain,
                containers.SiderealGainData | containers.CommonModeSiderealGainData,
            ):
                # Check that we are defined at the same RA samples
                if (gain.ra != tstream.ra).any():
                    raise RuntimeError(
                        "Gain data and sidereal stream defined at different RA samples."
                    )

            else:
                # We are using a time stream

                # Check that we are defined at the same time samples
                if (gain.time != tstream.time).any():
                    raise RuntimeError(
                        "Gain data and timestream defined at different time samples."
                    )

                # Smooth the gain data if required
                if self.smoothing_length is not None:
                    # Turn smoothing length into a number of samples
                    tdiff = gain.time[1] - gain.time[0]
                    samp = int(np.ceil(self.smoothing_length / tdiff))

                    # Ensure smoothing length is odd
                    l = 2 * (samp // 2) + 1

                    # Turn into 2D array (required by smoothing routines)
                    gain_r = gain_arr.reshape(-1, gain_arr.shape[-1])

                    # Get smoothing weight mask, if it exists
                    if weight_arr is not None:
                        wmask = (weight_arr > 0.0).astype(np.float64)
                    else:
                        wmask = np.ones(gain_r.shape, dtype=np.float64)

                    # Smooth amplitude and phase separately
                    smooth_amp = weighted_median.moving_weighted_median(
                        np.abs(gain_r), weights=wmask, size=(1, l)
                    )
                    smooth_phase = weighted_median.moving_weighted_median(
                        np.angle(gain_r), weights=wmask, size=(1, l)
                    )

                    # Recombine and reshape back to original shape
                    gain_arr = smooth_amp * np.exp(1.0j * smooth_phase)
                    gain_arr = gain_arr.reshape(gain.gain[:].shape)

                    # Smooth weight array if it exists
                    if weight_arr is not None:
                        # Smooth
                        shp = weight_arr.shape
                        weight_arr = weighted_median.moving_weighted_median(
                            weight_arr.reshape(-1, shp[-1]), weights=wmask, size=(1, l)
                        ).reshape(shp)
                        # Ensure flagged values remain flagged
                        weight_arr[wmask == 0] = 0.0

        else:
            raise RuntimeError("Format of `gain` argument is unknown.")

        # Regularise any crazy entries
        gain_arr = np.nan_to_num(gain_arr)

        # Invert the gains as we need both the gains and the inverse to update
        # the visibilities and the weights
        inverse_gain_arr = tools.invert_no_zero(gain_arr)

        # Apply gains to visibility matrix
        self.log.info("Applying inverse gain." if self.inverse else "Applying gain.")
        gvis = inverse_gain_arr if self.inverse else gain_arr
        if isinstance(gain, containers.SiderealGainData):
            # Need a prod_map for sidereal streams
            tools.apply_gain(
                tstream.vis[:], gvis, out=tstream.vis[:], prod_map=tstream.prod
            )
        elif isinstance(
            gain, containers.CommonModeGainData | containers.CommonModeSiderealGainData
        ):
            # Apply the gains to all 'prods/stacks' directly:
            tstream.vis[:] *= np.abs(gvis[:, np.newaxis, :]) ** 2
        else:
            tools.apply_gain(tstream.vis[:], gvis, out=tstream.vis[:])

        # Apply gains to the weights
        if self.update_weight:
            self.log.info("Applying gain to weight.")
            gweight = np.abs(gain_arr if self.inverse else inverse_gain_arr) ** 2
        else:
            gweight = np.ones_like(gain_arr, dtype=np.float64)

        if weight_arr is not None:
            gweight *= (weight_arr[:] > 0.0).astype(np.float64)

        if isinstance(gain, containers.SiderealGainData):
            # Need a prod_map for sidereal streams
            tools.apply_gain(
                tstream.weight[:], gweight, out=tstream.weight[:], prod_map=tstream.prod
            )
        elif isinstance(
            gain, containers.CommonModeGainData | containers.CommonModeSiderealGainData
        ):
            # Apply the gains to all 'prods/stacks' directly:
            tstream.weight[:] *= gweight[:, np.newaxis, :] ** 2
        else:
            tools.apply_gain(tstream.weight[:], gweight, out=tstream.weight[:])

        # Update units if they were specified
        convert_units_to = gain.gain.attrs.get("convert_units_to")
        if convert_units_to is not None:
            tstream.vis.attrs["units"] = convert_units_to

        return tstream


class AccumulateList(task.MPILoggedTask):
    """Accumulate the inputs into a list and return as a group.

    If `group_size` is None, return when the task *finishes*. Otherwise,
    return every time `group_size` inputs have been accumulated.

    Attributes
    ----------
    group_size
        If this is set, this task will return the list of accumulated
        data whenever it reaches this length. If not set, wait until
        no more input is received and then return everything.
    """

    group_size = config.Property(proptype=int, default=None)

    def __init__(self):
        super().__init__()
        self._items = []

    def next(self, input_):
        """Append an input to the list of inputs."""
        self._items.append(input_)

        if self.group_size is not None:
            if len(self._items) >= self.group_size:
                output = self._items
                self._items = []

                return output

        return None

    def finish(self):
        """Remove the internal reference.

        Prevents the items from hanging around after the task finishes.
        """
        items = self._items
        del self._items

        # If the group_size was set, then items will either be an empty list
        # or an incomplete list (with the incorrect number of outputs), so
        # in that case return None to prevent the pipeline from crashing.
        return items if self.group_size is None else None


class CheckMPIEnvironment(task.MPILoggedTask):
    """Check that the current MPI environment can communicate across all nodes."""

    timeout = config.Property(proptype=int, default=240)

    def setup(self):
        """Send random messages between all ranks.

        Tests to ensure that all messages are received within a specified amount
        of time, and that the messages received are the same as those sent (i.e.
        nothing was corrupted).
        """
        import time

        comm = self.comm
        n = 500000  # Corresponds to a 4 MB buffer
        results = []

        sends = np.arange(comm.size * n, dtype=np.float64).reshape(comm.size, n)
        recvs = np.empty_like(sends)

        # Send and receive across all ranks
        for i in range(comm.size):
            send = (comm.rank + i) % comm.size
            recv = (comm.rank - i) % comm.size

            results.append(comm.Irecv(recvs[recv, :], recv))
            comm.Isend(sends[comm.rank, :], send)

        start_time = time.time()

        while time.time() - start_time < self.timeout:
            success = all(r.get_status() for r in results)

            if success:
                break

            time.sleep(5)

        if not success:
            self.log.critical(
                f"MPI test failed to respond in {self.timeout} seconds. Aborting..."
            )
            comm.Abort()

        if not (recvs == sends).all():
            self.log.critical("MPI test did not receive the correct data. Aborting...")
            comm.Abort()

        # Stop successful processes from finshing if any task has failed
        comm.Barrier()

        self.log.debug(
            f"MPI test successful after {time.time() - start_time:.1f} seconds"
        )


class MakeCopy(task.SingleTask):
    """Make a copy of the passed container."""

    def process(self, data):
        """Return a copy of the given container.

        Parameters
        ----------
        data : containers.ContainerBase
            The container to copy.
        """
        return data.copy()


class WaitUntil(task.MPILoggedTask):
    """Wait until the the requires before forwarding inputs.

    This simple synchronization task will forward on whatever inputs it gets, however, it won't do
    this until it receives any requirement to it's setup method. This allows certain parts of the
    pipeline to be delayed until a piece of data further up has been generated.
    """

    def setup(self, input_):
        """Accept, but don't save any input."""
        self.log.info("Received the requirement, starting to forward inputs")
        pass

    def next(self, input_):
        """Immediately forward any input."""
        return input_


class PassOn(task.MPILoggedTask):
    """Unconditionally forward a tasks input.

    While this seems like a pointless no-op it's useful for connecting tasks in complex
    topologies.
    """

    def next(self, input_):
        """Immediately forward any input."""
        return input_


class DebugInfo(task.MPILoggedTask, task.SetMPILogging):
    """Output some useful debug info."""

    def __init__(self):
        import logging

        # Set the default log levels to something reasonable for debugging
        self.level_rank0 = logging.DEBUG
        self.level_all = logging.INFO
        task.SetMPILogging.__init__(self)
        task.MPILoggedTask.__init__(self)

        ip = self._get_external_ip()

        self.log.info(f"External IP is {ip}")

        if self.comm.rank == 0:
            versions = self._get_package_versions()

            for name, version in versions:
                self.log.info(f"Package: {name:40s} version={version}")

    def _get_external_ip(self) -> str:
        # Reference here:
        # https://community.cloudflare.com/t/can-1-1-1-1-be-used-to-find-out-ones-public-ip-address/14971/6

        # Setup a resolver to point to Cloudflare
        import dns.resolver

        r = dns.resolver.Resolver()
        r.nameservers = ["1.1.1.1"]

        # Get IP from cloudflare chaosnet TXT record, and parse the response
        res = r.resolve("whoami.cloudflare", "TXT", "CH", tcp=True, lifetime=15)

        return str(res[0]).replace('"', "")

    def _get_package_versions(self) -> list[tuple[str, str]]:
        import json
        import subprocess

        p = subprocess.run(["pip", "list", "--format", "json"], stdout=subprocess.PIPE)

        package_info = json.loads(p.stdout)

        package_list = []

        for p in package_info:
            package_list.append((p["name"], p["version"]))

        return package_list


class CombineBeamTransfers(task.SingleTask):
    """Combine beam transfer matrices for frequency sub-bands.

    Here are the recommended steps for generating BTMs in sub-bands (useful
    if the BTMs over the full band would take too much walltime for a single
    batch job):
    1. Create a telescope manager for the desired full set of BTMs (by running
    drift-makeproducts with a config file that doesn't generate any BTMs.)
    Make note of the `lmax` and `mmax` computed by this telescope manager
    2. Generate separate BTMs for sub-bands which cover the desired full band.
    In each separate config file, set `force_lmax` and `force_mmax` to the
    values for the full telescope, to ensure that the combined BTMs will exactly
    match what would be generated with a single job for the full band.
    3. Run this task.

    Attributes
    ----------
    partial_btm_configs : list
        List of config files for partial BTMs.
    overwrite : bool
        Whether to overwrite existing m files. Default: False.
    pol_pair_copy : list
        If specified, only copy BTM elements with polarization in the list.
        (For example, to only copy co-pol elements for CHIME: ["XX", "YY"].)
        Default: None.
    """

    partial_btm_configs = config.Property(proptype=list)
    overwrite = config.Property(proptype=bool, default=False)
    pol_pair_copy = config.Property(proptype=list, default=None)

    def process(self, bt_full):
        """Combine beam transfer matrices and write to new set of files.

        Parameters
        ----------
        bt_full : ProductManager or BeamTransfer
            Beam transfer manager for desired output beam transfer matrices.
        """
        from drift.core import manager
        from drift.core.telescope import PolarisedTelescope

        try:
            import bitshuffle.h5

            BITSHUFFLE_IMPORTED = True
        except ImportError:
            BITSHUFFLE_IMPORTED = False

        self.manager_full = bt_full
        self.beamtransfer_full = io.get_beamtransfer(bt_full)
        self.telescope_full = io.get_telescope(bt_full)

        self.n_partials = len(self.partial_btm_configs)

        self.beamtransfer_partial = []
        self.telescope_partial = []

        for file in self.partial_btm_configs:
            man = manager.ProductManager.from_config(file)
            self.beamtransfer_partial.append(man.beamtransfer)
            self.telescope_partial.append(man.telescope)
            self.log.info(
                f"Loaded product manager for {file}",
            )

        # Get lists of frequencies included in full BTM and each partial BTM
        self.desired_freqs = list(
            self.telescope_full.frequencies[self.telescope_full.included_freq]
        )
        self.partial_freqs = []
        for tel in self.telescope_partial:
            self.partial_freqs.append(list(tel.frequencies[tel.included_freq]))

        # Check that total set of frequencies in partial BTMs exactly matches
        # frequencies in desired BTM
        if len(self.desired_freqs) != len(np.concatenate(self.partial_freqs)):
            raise ValueError(
                "Provided BTMs have different number of frequencies than set of "
                "desired frequencies!"
            )
        if not np.allclose(
            np.sort(self.desired_freqs), np.sort(np.concatenate(self.partial_freqs))
        ):
            raise ValueError(
                "Frequencies of provided BTMs do not match set of desired frequencies!"
            )

        # Compute some quantities needed to define shapes of output files
        nf_inc = len(self.telescope_full.included_freq)
        nb_inc = len(self.telescope_full.included_baseline)
        np_inc = len(self.telescope_full.included_pol)
        nl = self.telescope_full.lmax + 1

        # Determine compression arguments for output BTM files
        if self.beamtransfer_full.truncate and BITSHUFFLE_IMPORTED:
            compression_kwargs = {
                "compression": bitshuffle.h5.H5FILTER,
                "compression_opts": (0, bitshuffle.h5.H5_COMPRESS_LZ4),
            }
        else:
            compression_kwargs = {"compression": "lzf"}

        # Determine which elements of baseline axis to copy.
        # pol_mask selects elements we *do not* want to copy.
        if self.pol_pair_copy is not None:
            if not isinstance(self.telescope_full, PolarisedTelescope):
                raise Exception("pol_pair_copy only works for polarized telescopes!")
            pol_pairs = [
                p[0] + p[1]
                for p in self.telescope_full.polarisation[
                    self.telescope_full.uniquepairs[
                        self.telescope_full.included_baseline
                    ]
                ]
            ]
            pol_mask = [p not in self.pol_pair_copy for p in pol_pairs]

        # Make output directories for new BTMs
        self.beamtransfer_full._generate_dirs()

        # Interate over m's local to this rank.
        # BTM for lower m's are generally larger, so we use method="alt"
        # in mpirange for better load-balancing.
        for mi in mpiutil.mpirange(self.telescope_full.mmax + 1, method="alt"):
            # Check whether file exists
            if os.path.exists(self.beamtransfer_full._mfile(mi)) and not self.overwrite:
                self.log.info(f"BTM file for m = {mi} exists. Skipping...")
                continue

            f_out = h5py.File(self.beamtransfer_full._mfile(mi), "w")

            # Create beam_m dataset
            dsize = (nf_inc, 2, nb_inc, np_inc, nl - mi)
            csize = (1, 2, min(10, nb_inc), np_inc, nl - mi)
            f_out.create_dataset(
                "beam_m", dsize, chunks=csize, dtype=np.complex128, **compression_kwargs
            )

            # Write a few useful attributes.
            f_out.attrs["m"] = mi
            f_out.attrs["frequencies"] = self.telescope_full.frequencies

            # Loop over partial BTMs
            for i in range(self.n_partials):
                # Get indices of partial frequencies within full frequency list
                nl_sub = self.telescope_partial[i].lmax + 1
                freq_sub = self.partial_freqs[i]
                freq_sub_idx = [self.desired_freqs.index(f) for f in freq_sub]

                # Copy partial BTMs into correct slice of full BTM output.
                # Full BTM ell axis may stretch beyond partial BTM ell axis,
                # but only elements of the full axis up to nl_sub-mi will be nonzero.
                if mi <= self.telescope_partial[i].mmax:
                    with h5py.File(
                        self.beamtransfer_partial[i]._mfile(mi), "r"
                    ) as f_sub:
                        # If copying specific polarizations, first load BTM elements
                        # from partial-BTM file, set unwanted pols to zero, then write
                        # to new file (necessary due to h5py restrictions on slicing)
                        if self.pol_pair_copy is not None:

                            block_to_copy = f_sub["beam_m"][..., :nl_sub]
                            block_to_copy[:, :, pol_mask, :, :] = 0.0
                            f_out["beam_m"][
                                freq_sub_idx, :, :, :, : nl_sub - mi
                            ] = block_to_copy
                            del block_to_copy
                        # If copying all polarizations, do it all at once
                        else:
                            f_out["beam_m"][freq_sub_idx, :, :, : nl_sub - mi] = f_sub[
                                "beam_m"
                            ][..., :nl_sub]

            # Close output file
            f_out.close()

            self.log.debug(f"Wrote file for m = {mi}")

        mpiutil.barrier()
        if mpiutil.rank0:
            self.log.info("New BTMs complete!")

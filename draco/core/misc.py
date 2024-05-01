"""Miscellaneous pipeline tasks with no where better to go.

Tasks should be proactively moved out of here when there is a thematically
appropriate module, or enough related tasks end up in here such that they can
all be moved out into their own module.
"""

import numpy as np
from caput import config

from ..core import containers, task
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
            gain, (containers.CommonModeGainData, containers.CommonModeSiderealGainData)
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
            (
                containers.GainData,
                containers.SiderealGainData,
                containers.CommonModeGainData,
                containers.CommonModeSiderealGainData,
            ),
        ):
            # Extract gain array
            gain_arr = gain.gain[:]

            # Regularise any crazy entries
            gain_arr = np.nan_to_num(gain_arr)

            # Get the weight array if it's there
            weight_arr = gain.weight[:] if gain.weight is not None else None

            if isinstance(
                gain,
                (containers.SiderealGainData, containers.CommonModeSiderealGainData),
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
                    import scipy.signal as ss

                    # Turn smoothing length into a number of samples
                    tdiff = gain.time[1] - gain.time[0]
                    samp = int(np.ceil(self.smoothing_length / tdiff))

                    # Ensure smoothing length is odd
                    l = 2 * (samp // 2) + 1

                    # Turn into 2D array (required by smoothing routines)
                    gain_r = gain_arr.reshape(-1, gain_arr.shape[-1])

                    # Smooth amplitude and phase separately
                    smooth_amp = ss.medfilt2d(np.abs(gain_r), kernel_size=[1, l])
                    smooth_phase = ss.medfilt2d(np.angle(gain_r), kernel_size=[1, l])

                    # Recombine and reshape back to original shape
                    gain_arr = smooth_amp * np.exp(1.0j * smooth_phase)
                    gain_arr = gain_arr.reshape(gain.gain[:].shape)

                    # Smooth weight array if it exists
                    if weight_arr is not None:
                        shp = weight_arr.shape
                        weight_arr = ss.medfilt2d(
                            weight_arr.reshape(-1, shp[-1]), kernel_size=[1, l]
                        ).reshape(shp)

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
            gain, (containers.CommonModeGainData, containers.CommonModeSiderealGainData)
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
            gain, (containers.CommonModeGainData, containers.CommonModeSiderealGainData)
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
    """Accumulate the inputs into a list and return when the task *finishes*."""

    def __init__(self):
        super().__init__()
        self._items = []

    def next(self, input_):
        """Append an input to the list of inputs."""
        self._items.append(input_)

    def finish(self):
        """Remove the internal reference.

        Prevents the items from hanging around after the task finishes.
        """
        items = self._items
        del self._items

        return items


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

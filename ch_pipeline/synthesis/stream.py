"""
============================================================
Timestream Simulation (:mod:`~ch_pipeline.synthesis.stream`)
============================================================

.. currentmodule:: ch_pipeline.synthesis.stream

Tasks for simulating sidereal and time stream data.

A typical pattern would be to turn a map into a
:class:`containers.SiderealStream` with the :class:`SimulateSidereal` task, then
expand any redundant products with :class:`ExpandProducts` and finally generate
a set of time stream files with :class:`MakeTimeStream`.

Tasks
=====

.. autosummary::
    :toctree: generated/

    SimulateSidereal
    ExpandProducts
    MakeTimeStream
"""

import numpy as np

from cora.util import hputil
from caput import mpiutil, pipeline, config, mpiarray

from ch_util import ephemeris

from ..core import containers, task


class SimulateSidereal(task.SingleTask):
    """Create a simulated sidereal dataset from an input map.
    """

    done = False

    def setup(self, beamtransfer):
        """Setup the simulation.

        Parameters
        ----------
        bt : BeamTransfer
            Beam Transfer maanger.
        """
        self.beamtransfer = beamtransfer
        self.telescope = beamtransfer.telescope

    def process(self, map_):
        """Simulate a SiderealStream

        Parameters
        ----------
        map : :class:`containers.Map`
            The sky map to process to into a sidereal stream. Frequencies in the map, must match the Beam Transfer matrices.

        Returns
        -------
        ss : SiderealStream
            Stacked sidereal day.
        feeds : list of CorrInput
            Description of the feeds simulated.
        """

        if self.done:
            raise pipeline.PipelineStopIteration

        # Read in telescope system
        bt = self.beamtransfer
        tel = self.telescope

        lmax = tel.lmax
        mmax = tel.mmax
        nfreq = tel.nfreq
        npol = tel.num_pol_sky

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)

        lm, sm, em = mpiutil.split_local(mmax + 1)

        # Set the minimum resolution required for the sky.
        ntime = 2 * mmax + 1

        freqmap = map_.index_map['freq'][:]
        row_map = map_.map[:]

        if (tel.frequencies != freqmap['centre']).all():
            raise RuntimeError('Frequencies in map file (%s) do not match those in Beam Transfers.' % mapfile)

        # Calculate the alm's for the local sections
        row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape((lfreq, npol * (lmax + 1), lmax + 1))

        # Trim off excess m's and wrap into MPIArray
        row_alm = row_alm[..., :(mmax + 1)]
        row_alm = mpiarray.MPIArray.wrap(row_alm, axis=0)

        # Perform the transposition to distribute different m's across processes. Neat
        # tip, putting a shorter value for the number of columns, trims the array at
        # the same time
        col_alm = row_alm.redistribute(axis=2)

        # Transpose and reshape to shift m index first.
        col_alm = col_alm.transpose((2, 0, 1)).reshape((None, nfreq, npol, lmax + 1))

        # Create storage for visibility data
        vis_data = mpiarray.MPIArray((mmax + 1, nfreq, bt.ntel), axis=0, dtype=np.complex128)
        vis_data[:] = 0.0

        # Iterate over m's local to this process and generate the corresponding
        # visibilities
        for mp, mi in vis_data.enumerate(axis=0):
            vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp].view(np.ndarray))

        # Rearrange axes such that frequency is last (as we want to divide
        # frequencies across processors)
        row_vis = vis_data.transpose((0, 2, 1))

        # Parallel transpose to get all m's back onto the same processor
        col_vis_tmp = row_vis.redistribute(axis=2)
        col_vis_tmp = col_vis_tmp.reshape((mmax + 1, 2, tel.npairs, None))

        # Transpose the local section to make the m's the last axis and unwrap the
        # positive and negative m at the same time.
        col_vis = mpiarray.MPIArray((tel.npairs, nfreq, ntime), axis=1, dtype=np.complex128)
        col_vis[:] = 0.0
        col_vis[..., 0] = col_vis_tmp[0, 0]
        for mi in range(1, mmax + 1):
            col_vis[..., mi] = col_vis_tmp[mi, 0]
            col_vis[..., -mi] = col_vis_tmp[mi, 1].conj()  # Conjugate only (not (-1)**m - see paper)

        del col_vis_tmp

        # Fourier transform m-modes back to get final timestream.
        vis_stream = np.fft.ifft(col_vis, axis=-1) * ntime
        vis_stream = vis_stream.reshape((tel.npairs, lfreq, ntime))
        vis_stream = vis_stream.transpose((1, 0, 2)).copy()

        # Try and fetch out the feed index and info from the telescope object.
        try:
            feed_index = tel.input_index
        except AttributeError:
            feed_index = tel.nfeed

        # Construct container and set visibility data
        sstream = containers.SiderealStream(freq=freqmap, ra=ntime, input=feed_index,
                                            prod=tel.uniquepairs, distributed=True, comm=map_.comm)
        sstream.vis[:] = mpiarray.MPIArray.wrap(vis_stream, axis=0)
        sstream.weight[:] = 1.0

        self.done = True

        return sstream


def _list_of_timeranges(dlist):

    if not isinstance(list, dlist):
        pass


class ExpandProducts(task.SingleTask):
    """Un-wrap collated products to full triangle.
    """

    def setup(self, telescope):
        """Get a reference to the telescope class.

        Parameters
        ----------
        tel : :class:`drift.core.TransitTelescope`
            Telescope object.
        """
        self.telescope = telescope

    def process(self, sstream):
        """Transform a sidereal stream to having a full product matrix.

        Parameters
        ----------
        sstream : :class:`containers.SiderealStream`
            Sidereal stream to unwrap.

        Returns
        -------
        new_sstream : :class:`containers.SiderealStream`
            Unwrapped sidereal stream.
        """

        sstream.redistribute('freq')

        ninput = len(sstream.input)

        prod = np.array([ (fi, fj) for fi in range(ninput) for fj in range(fi, ninput)])

        new_stream = containers.SiderealStream(prod=prod, axes_from=sstream)
        new_stream.redistribute('freq')
        new_stream.vis[:] = 0.0
        new_stream.weight[:] = 0.0

        # Iterate over all feed pairs and work out which is the correct index in the sidereal stack.
        for pi, (fi, fj) in enumerate(prod):

            unique_ind = self.telescope.feedmap[fi, fj]
            conj = self.telescope.feedconj[fi, fj]

            # unique_ind is less than zero it has masked out
            if unique_ind < 0:
                continue

            prod_stream = sstream.vis[:, unique_ind]
            new_stream.vis[:, pi] = prod_stream.conj() if conj else prod_stream

            new_stream.weight[:, pi] = 1.0

        return new_stream


class MakeTimeStream(task.SingleTask):
    """Generate a series of time streams files from a sidereal stream.

    Parameters
    ----------
    start_time, end_time : float or datetime
        Start and end times of the timestream to simulate. Needs to be either a
        `float` (UNIX time) or a `datetime` objects in UTC.
    integration_time : float, optional
        Integration time in seconds. Takes precedence over `integration_frame_exp`.
    integration_frame_exp: int, optional
        Specify the integration time in frames. The integration time is
        `2**integration_frame_exp * 2.56 us`.
    samples_per_file : int, optional
        Number of samples per file.
    """

    start_time = config.Property(proptype=ephemeris.ensure_unix)
    end_time = config.Property(proptype=ephemeris.ensure_unix)

    integration_time = config.Property(proptype=float, default=None)
    integration_frame_exp = config.Property(proptype=int, default=23)

    samples_per_file = config.Property(proptype=int, default=1024)

    _cur_time = 0.0  # Hold the current file start time

    def setup(self, sstream):
        """Get the sidereal stream to turn into files.

        Parameters
        ----------
        sstream : SiderealStream
        """
        self.sstream = sstream

        # Initialise the current start time
        self._cur_time = self.start_time

    def process(self):
        """Create a timestream file.

        Returns
        -------
        tstream : :class:`andata.CorrData`
            Time stream object.
        """

        from ..util import regrid

        # First check to see if we have reached the end of the requested time,
        # and if so stop the iteration.
        if self._cur_time > self.end_time:
            raise pipeline.PipelineStopIteration

        # Calculate the integration time
        if self.integration_time is not None:
            int_time = self.integration_time
        else:
            int_time = 2.56e-6 * 2**self.integration_frame_exp

        # Calculate number of samples in file and timestamps
        nsamp = min(int(np.ceil((self.end_time - self._cur_time) / int_time)), self.samples_per_file)
        timestamps = self._cur_time + (np.arange(nsamp) + 1) * int_time  # +1 as timestamps are at the end of each sample

        # Construct the time axis index map
        if self.integration_time is not None:
            time = timestamps
        else:
            _time_dtype = [('fpga_count', np.uint64), ('ctime', np.float64)]
            time = np.zeros(nsamp, _time_dtype)
            time['ctime'] = timestamps
            time['fpga_count'] = (timestamps - self.start_time) / int_time * 2**self.integration_frame_exp

        # Make the timestream container
        tstream = containers.make_empty_corrdata(axes_from=self.sstream, time=time)

        # Make the interpolation array
        ra = ephemeris.transit_RA(tstream.time)
        lza = regrid.lanczos_forward_matrix(self.sstream.ra, ra, periodic=True)
        lza = lza.T.astype(np.complex64)

        # Apply the interpolation matrix to construct the new timestream, place
        # the output directly into the container
        np.dot(self.sstream.vis[:], lza, out=tstream.vis[:])

        # Set the weights array to the maximum value for CHIME
        tstream.weight[:] = 255.0

        # Increment the current start time for the next iteration
        self._cur_time += nsamp * int_time

        # Output the timestream
        return tstream

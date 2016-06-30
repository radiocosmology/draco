"""
=============================================================
Data transformations (:mod:`~ch_pipeline.analysis.transform`)
=============================================================

.. currentmodule:: ch_pipeline.anaysis.transform

A collection of miscellaneous transformations to do on data, from grouping
frequencies and products to performing the m-mode transform.

Tasks
=====

.. autosummary::
    :toctree: generated/

    FrequencyRebin
    SelectFreq
    CollateProducts
    MModeTransform
"""
import numpy as np
from caput import mpiarray, config

from ch_util import tools, andata

from ..core import containers, task


class FrequencyRebin(task.SingleTask):
    """Rebin neighbouring frequency channels.

    Parameters
    ----------
    channel_bin : int
        Number of channels to in together.
    """

    channel_bin = config.Property(proptype=int, default=1)

    def process(self, ss):
        """Take the input dataset and rebin the frequencies.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sb : SiderealStream
        """

        if 'freq' not in ss.index_map:
            raise RuntimeError('Data does not have a frequency axis.')

        if len(ss.freq) % self.channel_bin != 0:
            raise RuntimeError("Binning must exactly divide the number of channels.")

        # Get all frequencies onto same node
        ss.redistribute('time')

        # Calculate the new frequency centres and widths
        fc = ss.index_map['freq']['centre'].reshape(-1, self.channel_bin).mean(axis=-1)
        fw = ss.index_map['freq']['width'].reshape(-1, self.channel_bin).sum(axis=-1)

        freq_map = np.empty(fc.shape[0], dtype=ss.index_map['freq'].dtype)
        freq_map['centre'] = fc
        freq_map['width'] = fw

        # Create new container for rebinned stream
        if isinstance(ss, containers.ContainerBase):
            sb = ss.__class__(freq=freq_map, axes_from=ss)
        elif isinstance(ss, andata.CorrData):
            sb = containers.make_empty_corrdata(freq=freq_map, axes_from=ss, distributed=True,
                                                distributed_axis=2, comm=ss.comm)
        else:
            raise RuntimeError("I don't know how to deal with data type %s" % ss.__class__.__name__)

        # Get all frequencies onto same node
        sb.redistribute('time')

        # Copy over the tag attribute
        sb.attrs['tag'] = ss.attrs['tag']

        # Rebin the arrays, do this with a loop to save memory
        for fi in range(len(ss.freq)):

            # Calculate rebinned index
            ri = fi / self.channel_bin

            sb.vis[ri] += ss.vis[fi] * ss.weight[fi]
            sb.gain[ri] += ss.gain[fi] / self.channel_bin  # Don't do weighted average for the moment

            sb.weight[ri] += ss.weight[fi]

            # If we are on the final sub-channel then divide the arrays through
            if (fi + 1) % self.channel_bin == 0:
                sb.vis[ri] *= tools.invert_no_zero(sb.weight[ri])

        sb.redistribute('freq')

        return sb


class CollateProducts(task.SingleTask):
    """Extract and order the correlation products for map-making.

    The task will take a sidereal task and format the products that are needed
    or the map-making. It uses a BeamTransfer instance to figure out what these
    products are, and how they should be ordered. It similarly selects only the
    required frequencies.

    It is important to note that while the input
    :class:`SiderealStream` can contain more feeds and frequencies
    than are contained in the BeamTransfers, the converse is not
    true. That is, all the frequencies and feeds that are in the
    BeamTransfers must be found in the timestream object.
    """

    def setup(self, bt):
        """Set the BeamTransfer instance to use.

        Parameters
        ----------
        bt : BeamTransfer
        """

        self.beamtransfer = bt
        self.telescope = bt.telescope

    def process(self, ss):
        """Select and reorder the products.

        Parameters
        ----------
        ss : SiderealStream

        Returns
        -------
        sp : SiderealStream
            Dataset containing only the required products.
        """

        ss_keys = ss.index_map['input'][:]

        # Figure the mapping between inputs for the beam transfers and the file
        try:
            bt_keys = self.telescope.feeds
        except AttributeError:
            bt_keys = np.arange(self.telescope.nfeed)

        def find_key(key_list, key):
            try:
                return map(tuple, list(key_list)).index(tuple(key))
            except TypeError:
                return list(key_list).index(key)
            except ValueError:
                return None

        input_ind = [ find_key(bt_keys, sk) for sk in ss_keys]

        # Figure out mapping between the frequencies
        bt_freq = self.telescope.frequencies
        ss_freq = ss.freq['centre']

        freq_ind = [ find_key(ss_freq, bf) for bf in bt_freq]

        sp_freq = ss.freq[freq_ind]

        sp = containers.SiderealStream(
            freq=sp_freq, input=len(bt_keys), prod=self.telescope.uniquepairs,
            axes_from=ss, attrs_from=ss, distributed=True, comm=ss.comm
        )

        # Ensure all frequencies and products are on each node
        ss.redistribute('ra')
        sp.redistribute('ra')

        sp.vis[:] = 0.0
        sp.weight[:] = 0.0

        # Iterate over products in the sidereal stream
        for ss_pi in range(len(ss.index_map['prod'])):

            # Get the feed indices for this product
            ii, ij = ss.index_map['prod'][ss_pi]

            # Map the feed indices into ones for the Telescope class
            bi, bj = input_ind[ii], input_ind[ij]

            # If either feed is not in the telescope class, skip it.
            if bi is None or bj is None:
                continue

            sp_pi = self.telescope.feedmap[bi, bj]
            feedconj = self.telescope.feedconj[bi, bj]

            # Skip if product index is not valid
            if sp_pi < 0:
                continue

            # Accumulate visibilities, conjugating if required
            if not feedconj:
                sp.vis[:, sp_pi] += ss.weight[freq_ind, ss_pi] * ss.vis[freq_ind, ss_pi]
            else:
                sp.vis[:, sp_pi] += ss.weight[freq_ind, ss_pi] * ss.vis[freq_ind, ss_pi].conj()

            # Accumulate weights
            sp.weight[:, sp_pi] += ss.weight[freq_ind, ss_pi]

        # Divide through by weights to get properly weighted visibility average
        sp.vis[:] *= tools.invert_no_zero(sp.weight[:])

        # Switch back to frequency distribution
        ss.redistribute('freq')
        sp.redistribute('freq')

        return sp


class SelectFreq(task.SingleTask):
    """Select a subset of frequencies from the data.

    Attributes
    ----------
    frequencies : list
        List of frequency indices.
    """

    frequencies = config.Property(proptype=list)

    def process(self, data):
        """Selet a subset of the frequencies.

        Parameters
        ----------
        data : containers.ContainerBase
            A data container with a frequency axis.

        Returns
        -------
        newdata : containers.ContainerBase
            New container with trimmed frequencies.
        """

        freq_map = data.index_map['freq'][self.frequencies]
        data.redistribute(['ra', 'time'])

        newdata = data.__class__(freq=freq_map, axes_from=data, attrs_from=data)
        newdata.redistribute(['ra', 'time'])

        for name, dset in data.datasets.items():

            if 'freq' in dset.attrs['axis']:
                newdata.datasets[name][:] = data.datasets[name][self.frequencies, ...]
            else:
                newdata.datasets[name][:] = data.datasets[name][:]

        return newdata


class MModeTransform(task.SingleTask):
    """Transform a sidereal stream to m-modes.

    Currently ignores any noise weighting.
    """

    def process(self, sstream):
        """Perform the m-mode transform.

        Parameters
        ----------
        sstream : containers.SiderealStream
            The input sidereal stream.

        Returns
        -------
        mmodes : containers.MModes
        """

        sstream.redistribute('freq')

        # Sum the noise variance over time samples, this will become the noise
        # variance for the m-modes
        weight_sum = sstream.weight[:].sum(axis=-1)

        # Construct the array of m-modes
        marray = _make_marray(sstream.vis[:])
        marray = mpiarray.MPIArray.wrap(marray[:], axis=2, comm=sstream.comm)

        # Create the container to store the modes in
        mmax = marray.shape[0] - 1
        ma = containers.MModes(mmax=mmax, axes_from=sstream, comm=sstream.comm)
        ma.redistribute('freq')

        # Assign the visibilities and weights into the container
        ma.vis[:] = marray
        ma.weight[:] = weight_sum[np.newaxis, np.newaxis, :, :]

        ma.redistribute('m')

        return ma


def _make_marray(ts):
    # Construct an array of m-modes from a sidereal time stream
    mmodes = np.fft.fft(ts, axis=-1) / ts.shape[-1]
    marray = _pack_marray(mmodes)

    return marray


def _pack_marray(mmodes, mmax=None):
    # Pack an FFT into the correct format for the m-modes (i.e. [m, freq, +/-,
    # baseline])

    if mmax is None:
        mmax = mmodes.shape[-1] / 2

    shape = mmodes.shape[:-1]

    marray = np.zeros((mmax + 1, 2) + shape, dtype=np.complex128)

    marray[0, 0] = mmodes[..., 0]

    mlimit = min(mmax, mmodes.shape[-1] / 2)  # So as not to run off the end of the array
    for mi in range(1, mlimit - 1):
        marray[mi, 0] = mmodes[..., mi]
        marray[mi, 1] = mmodes[..., -mi].conj()

    return marray

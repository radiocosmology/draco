"""Miscellaneous transformations to do on data, from grouping frequencies and
products to performing the m-mode transform.

Tasks
=====

.. autosummary::
    :toctree:

    FrequencyRebin
    SelectFreq
    CollateProducts
    MModeTransform
"""
import numpy as np
from caput import mpiarray, config

from ..core import containers, task, io
from ..util import tools


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
        ss : containers.SiderealStream or containers.TimeStream
            Input data to rebin. Can also be an `andata.CorrData` instance,
            however the output will be a `containers.TimeStream` instance.

        Returns
        -------
        sb : containers.SiderealStream or containers.TimeStream
            Rebinned data. Type should match the input.
        """

        if 'freq' not in ss.index_map:
            raise RuntimeError('Data does not have a frequency axis.')

        if len(ss.freq) % self.channel_bin != 0:
            raise RuntimeError("Binning must exactly divide the number of channels.")

        # Get all frequencies onto same node
        ss.redistribute(['time', 'ra'])

        # Calculate the new frequency centres and widths
        fc = ss.index_map['freq']['centre'].reshape(-1, self.channel_bin).mean(axis=-1)
        fw = ss.index_map['freq']['width'].reshape(-1, self.channel_bin).sum(axis=-1)

        freq_map = np.empty(fc.shape[0], dtype=ss.index_map['freq'].dtype)
        freq_map['centre'] = fc
        freq_map['width'] = fw

        # Create new container for rebinned stream
        sb = containers.empty_like(ss, freq=freq_map)

        # Get all frequencies onto same node
        sb.redistribute(['time', 'ra'])

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
    :class:`~containers.SiderealStream` can contain more feeds and frequencies
    than are contained in the BeamTransfers, the converse is not true. That is,
    all the frequencies and feeds that are in the BeamTransfers must be found in
    the timestream object.

    Parameters
    ----------
    weight : string ('natural', 'uniform', or 'inverse_variance')
        How to weight the redundant baselines when stacking:
            'natural' - each baseline weighted by its redundancy (default)
            'uniform' - each baseline given equal weight
            'inverse_variance' - each baseline weighted by the weight attribute
    """

    weight = config.Property(proptype=str, default='natural')

    def setup(self, tel):
        """Set the Telescope instance to use.

        Parameters
        ----------
        tel : TransitTelescope
        """

        if self.weight not in ['natural', 'uniform', 'inverse_variance']:
            KeyError("Do not recognize weight = %s" % self.weight)

        self.telescope = io.get_telescope(tel)

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

        # Define two functions that are used below
        def find_key(key_list, key):
            try:
                return map(tuple, list(key_list)).index(tuple(key))
            except TypeError:
                return list(key_list).index(key)
            except ValueError:
                return None

        def pack_product_array(arr):

            nfeed = arr.shape[0]
            nprod = (nfeed * (nfeed+1)) // 2

            ret = np.zeros(nprod, dtype=arr.dtype)
            iout = 0

            for i in xrange(nfeed):
                ret[iout:(iout+nfeed-i)] = arr[i,i:]
                iout += (nfeed-i)

            return ret

        # Determine current conjugation and product map.
        match_sn = True
        if 'stack' in ss.index_map:
            match_sn = ss.index_map['stack'].size == ss.index_map['prod'].size
            ss_conj = ss.index_map['stack']['conjugate']
            ss_prod = ss.index_map['prod'][ss.index_map['stack']['prod']]
        else:
            ss_conj = np.zeros(ss.vis.shape[1], dtype=np.bool)
            ss_prod = ss.index_map['prod']

        # For each input in the file, find the corresponding index in the telescope instance
        ss_keys = ss.index_map['input'][:]
        try:
            bt_keys = self.telescope.input_index
        except AttributeError:
            bt_keys = np.array(np.arange(self.telescope.nfeed), dtype=[('chan_id', 'u2')])
            match_sn = False

        field_to_match = 'correlator_input' if match_sn else 'chan_id'
        input_ind = [find_key(bt_keys[field_to_match], sk) for sk in ss_keys[field_to_match]]

        # Figure out the reverse mapping (i.e., for each input in the telescope instance,
        # find the corresponding index in file)
        rev_input_ind = [find_key(ss_keys[field_to_match], bk) for bk in bt_keys[field_to_match]]

        if any([rv is None for rv in rev_input_ind]):
            raise ValueError("All feeds in Telescope instance must exist in Timestream instance.")

        # Figure out mapping between the frequencies
        freq_ind = [find_key(ss.freq[:], bf) for bf in self.telescope.frequencies]

        if any([fi is None for fi in freq_ind]):
            raise ValueError("All frequencies in Telescope instance must exist in Timestream instance.")

        bt_freq = ss.index_map['freq'][freq_ind]

        # Construct the equivalent prod and stack index_map for the telescope instance
        bt_prod = np.array([(fi, fj) for fi in range(self.telescope.nfeed)
                                     for fj in range(fi, self.telescope.nfeed)],
                                     dtype=[('input_a', '<u2'), ('input_b', '<u2')])

        bt_stack = np.array([(tools.cmap(upp[0], upp[1], self.telescope.nfeed), 0)
                             if upp[0] <= upp[1] else
                             (tools.cmap(upp[1], upp[0], self.telescope.nfeed), 1)
                             for upp in self.telescope.uniquepairs],
                             dtype=[('prod', '<u4'), ('conjugate', 'u1')])

        # Construct the equivalent stack reverse_map for the telescope instance.  Note
        # that we identify invalid products here using an index that is the size of the stack axis.
        feedmask = pack_product_array(self.telescope.feedmask)
        bt_rev = np.array(zip(
                 np.where(feedmask, pack_product_array(self.telescope.feedmap), self.telescope.npairs),
                 np.where(feedmask, pack_product_array(self.telescope.feedconj), 0)),
                 dtype=[('stack', '<u4'), ('conjugate', 'u1')])

        # Create output container
        if isinstance(ss, containers.SiderealStream):
            OutputContainer = containers.SiderealStream
        else:
            OutputContainer = containers.TimeStream

        sp = OutputContainer(freq=bt_freq, input=bt_keys, prod=bt_prod,
                             stack=bt_stack, reverse_map_stack=bt_rev,
                             axes_from=ss, attrs_from=ss, distributed=True, comm=ss.comm)

        # Ensure all frequencies and products are on each node
        ss.redistribute(['ra', 'time'])
        sp.redistribute(['ra', 'time'])

        # Initialize datasets in output container
        sp.vis[:] = 0.0
        sp.weight[:] = 0.0
        sp.input_flags[:] = ss.input_flags[rev_input_ind, :]
        if 'gain' in ss.datasets:
            sp.add_dataset('gain')
            sp.gain[:] = ss.gain[freq_ind][:, rev_input_ind, :]

        # Infer number of products that went into each stack
        if self.weight != 'inverse_variance':

            nprod_in_stack = tools.calculate_redundancy(ss.input_flags[:],
                                                        ss.index_map['prod'][:],
                                                        ss.reverse_map['stack']['stack'][:],
                                                        ss.vis.shape[1])

            if self.weight == 'uniform':
                nprod_in_stack = (nprod_in_stack > 0).astype(np.float32)

        # Create counter to increment during the stacking.
        # This will be used to normalize at the end.
        counter = np.zeros_like(sp.weight[:])

        # Iterate over products (stacked) in the sidereal stream
        for ss_pi, ((ii, ij), conj) in enumerate(zip(ss_prod, ss_conj)):

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

            # Generate weight
            if self.weight == 'inverse_variance':
                wss = ss.weight[freq_ind, ss_pi]

            else:
                wss = (ss.weight[freq_ind, ss_pi] > 0.0).astype(np.float32)
                wss *= nprod_in_stack[np.newaxis, ss_pi]

            # Accumulate visibilities, conjugating if required
            if feedconj == conj:
                sp.vis[:, sp_pi] += wss * ss.vis[freq_ind, ss_pi]
            else:
                sp.vis[:, sp_pi] += wss * ss.vis[freq_ind, ss_pi].conj()

            # Accumulate variances in quadrature.  Save in the weight dataset.
            sp.weight[:, sp_pi] += wss**2 * tools.invert_no_zero(ss.weight[freq_ind, ss_pi])

            # Increment counter
            counter[:, sp_pi] += wss

        # Divide through by counter to get properly weighted visibility average
        sp.vis[:] *= tools.invert_no_zero(counter)
        sp.weight[:] = counter**2 * tools.invert_no_zero(sp.weight[:])

        # Switch back to frequency distribution
        ss.redistribute('freq')
        sp.redistribute('freq')

        return sp


class SelectFreq(task.SingleTask):
    """Select a subset of frequencies from a container.

    Attributes
    ----------
    freq_physical : list
        List of physical frequencies in MHz.
        Given first priority.
    channel_range : list
        Range of frequency channel indices, either
        [start, stop, step], [start, stop], or [stop]
        is acceptable.  Given second priority.
    channel_index : list
        List of frequency channel indices.
        Given third priority.
    freq_physical_range : list
        Range of physical frequencies to include given as (low_freq, high_freq).
        Given fourth priority.
    """

    freq_physical = config.Property(proptype=list, default=[])
    freq_physical_range = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

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

        # Set up frequency selection.
        freq_map = data.index_map['freq']

        # Construct the frequency channel selection
        if self.freq_physical:
            newindex = sorted(set([np.argmin(np.abs(freq_map['centre'] - freq)) for freq in self.freq_physical]))

        elif self.channel_range and (len(self.channel_range) <= 3):
            newindex = slice(*self.channel_range)

        elif self.channel_index:
            newindex = self.channel_index

        elif self.freq_physical_range:
            low, high = sorted(self.freq_physical_range)
            newindex = np.where((freq_map['centre'] >= low) & (freq_map['centre'] < high))[0]

        else:
            ValueError("Must specify either freq_physical, channel_range, or channel_index.")

        freq_map = freq_map[newindex]

        # Destribute input container over ra or time.
        data.redistribute(['ra', 'time', 'pixel'])

        # Create new container with subset of frequencies.
        newdata = containers.empty_like(data, freq=freq_map)

        # Redistribute new container over ra or time.
        newdata.redistribute(['ra', 'time', 'pixel'])

        # Copy over datasets. If the dataset has a frequency axis,
        # then we only copy over the subset.
        if isinstance(data, containers.ContainerBase):

            for name, dset in data.datasets.iteritems():

                if name not in newdata.datasets:
                    newdata.add_dataset(name)

                if 'freq' in dset.attrs['axis']:
                    slc = [slice(None)] * len(dset.shape)
                    slc[list(dset.attrs['axis']).index('freq')] = newindex
                    newdata.datasets[name][:] = dset[slc]
                else:
                    newdata.datasets[name][:] = dset[:]

        else:
            newdata.vis[:] = data.vis[newindex, :, :]
            newdata.weight[:] = data.weight[newindex, :, :]
            newdata.gain[:] = data.gain[newindex, :, :]

        # Switch back to frequency distribution
        data.redistribute('freq')
        newdata.redistribute('freq')

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

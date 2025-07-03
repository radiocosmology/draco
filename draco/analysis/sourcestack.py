"""Source Stack Analysis Tasks."""

import numpy as np
from caput import config, pipeline
from cora.util import units
from mpi4py import MPI

from ..analysis import beamform
from ..core import containers, io, task
from ..util.random import RandomTask
from ..util.tools import invert_no_zero

# Constants
NU21 = units.nu21
C = units.c


class SourceStack(task.SingleTask):
    """Stack the product of `draco.analysis.BeamForm` accross sources.

    For this to work BeamForm must have been run with `collapse_ha = True` (default).

    Attributes
    ----------
    freqside : int
        Number of frequency bins to keep on each side of source bin
        when stacking. Default: 50.
    single_source_bin_index : int, optional
        Only stack on sources in frequency bin with this index.
        Useful for isolating stacking signal from a narrow frequency range.
        Default: None.
    """

    # Number of frequencies to keep on each side of source RA
    freqside = config.Property(proptype=int, default=50)

    # Only consider sources within frequency channel with this index
    single_source_bin_index = config.Property(proptype=int, default=None)

    def process(self, formed_beam):
        """Receives a formed beam object and stack across sources.

        Parameters
        ----------
        formed_beam : `containers.FormedBeam` object
            Formed beams to stack over sources.

        Returns
        -------
        stack : `containers.FrequencyStack` object
            The stack of sources.
        """
        # Get communicator
        comm = formed_beam.comm

        # Ensure formed_beam is distributed in sources
        formed_beam.redistribute("object_id")

        # Local shape and offset
        loff = formed_beam.beam.local_offset[0]
        lshape = formed_beam.beam.local_shape[0]

        # Frequency axis
        freq = formed_beam.freq
        nfreq = len(freq)

        # Polarisation axis
        pol = formed_beam.pol
        npol = len(pol)

        # Frequency of sources in MHz
        source_freq = NU21 / (formed_beam["redshift"]["z"][loff : loff + lshape] + 1.0)

        # Size of source stack array
        self.nstack = 2 * self.freqside + 1

        # Construct frequency offset axis (for stack container)
        self.stack_axis = np.copy(
            formed_beam.frequency[
                int(nfreq / 2) - self.freqside : int(nfreq / 2) + self.freqside + 1
            ]
        )
        self.stack_axis["centre"] = (
            self.stack_axis["centre"] - self.stack_axis["centre"][self.freqside]
        )

        # Get f_mask and source_indices
        freqdiff = freq[np.newaxis, :] - source_freq[:, np.newaxis]

        # Stack axis bin edges to digitize each source at, in either increasing
        # or decreasing order depending on order of frequencies
        if self.stack_axis["centre"][0] > self.stack_axis["centre"][-1]:
            stackbins = self.stack_axis["centre"] + 0.5 * self.stack_axis["width"]
            stackbins = np.append(
                stackbins,
                self.stack_axis["centre"][-1] - 0.5 * self.stack_axis["width"][-1],
            )
        else:
            stackbins = self.stack_axis["centre"] - 0.5 * self.stack_axis["width"]
            stackbins = np.append(
                stackbins,
                self.stack_axis["centre"][-1] + 0.5 * self.stack_axis["width"][-1],
            )

        # Index of each frequency in stack axis, for each source
        source_indices = np.digitize(freqdiff, stackbins) - 1

        # Indices to be processed in full frequency axis for each source
        f_mask = (source_indices >= 0) & (source_indices < self.nstack)

        # Only sources in the frequency range of the data.
        source_mask = (np.sum(f_mask, axis=1) > 0).astype(bool)

        # If desired, also restrict to sources within a specific channel.
        # This works because the frequency axis is not distributed between
        # ranks.
        if self.single_source_bin_index is not None:
            fs = formed_beam.index_map["freq"][self.single_source_bin_index]
            restricted_chan_mask = np.abs(source_freq - fs["centre"]) < (
                0.5 * fs["width"]
            )
            source_mask *= restricted_chan_mask

        # Container to hold the stack
        if npol > 1:
            stack = containers.FrequencyStackByPol(
                freq=self.stack_axis, pol=pol, attrs_from=formed_beam
            )
        else:
            stack = containers.FrequencyStack(
                freq=self.stack_axis, attrs_from=formed_beam
            )

        # Loop over polarisations
        for pp, pstr in enumerate(pol):
            fb = formed_beam.beam[:, pp].view(np.ndarray)
            fw = formed_beam.weight[:, pp].view(np.ndarray)

            # Source stack array.
            source_stack = np.zeros(self.nstack, dtype=np.float64)
            source_weight = np.zeros(self.nstack, dtype=np.float64)

            count = 0  # Source counter
            # For each source in the range of this process
            for lq in range(lshape):
                if not source_mask[lq]:
                    # Source not in the data redshift range
                    continue

                count += 1

                # Indices and slice for frequencies included in the stack.
                f_indices = np.arange(nfreq, dtype=np.int32)[f_mask[lq]]
                f_slice = np.s_[f_indices[0] : f_indices[-1] + 1]

                source_stack += np.bincount(
                    source_indices[lq, f_slice],
                    weights=fw[lq, f_slice] * fb[lq, f_slice],
                    minlength=self.nstack,
                )

                source_weight += np.bincount(
                    source_indices[lq, f_slice],
                    weights=fw[lq, f_slice],
                    minlength=self.nstack,
                )

            # Gather source stack for all ranks. Each contains the sum
            # over a different subset of sources.

            source_stack_full = np.zeros(
                comm.size * self.nstack, dtype=source_stack.dtype
            )
            source_weight_full = np.zeros(
                comm.size * self.nstack, dtype=source_weight.dtype
            )
            # Gather all ranks
            comm.Allgather(source_stack, source_stack_full)
            comm.Allgather(source_weight, source_weight_full)

            # Determine the index for the output container
            oslc = (pp, slice(None)) if npol > 1 else slice(None)

            # Sum across ranks
            stack.weight[oslc] = np.sum(
                source_weight_full.reshape(comm.size, self.nstack), axis=0
            )
            stack.stack[oslc] = np.sum(
                source_stack_full.reshape(comm.size, self.nstack), axis=0
            ) * invert_no_zero(stack.weight[oslc])

            # Gather all ranks of count. Report number of sources stacked
            full_count = comm.reduce(count, op=MPI.SUM, root=0)
            if comm.rank == 0:
                self.log.info(f"Number of sources stacked for pol {pstr}: {full_count}")

        return stack


class RandomSubset(task.SingleTask, RandomTask):
    """Take a large mock catalog and draw `number` catalogs of a given `size`.

    Attributes
    ----------
    number : int
        Number of catalogs to construct.
    size : int
        Number of objects in each catalog.
    """

    number = config.Property(proptype=int)
    size = config.Property(proptype=int)

    def __init__(self):
        super().__init__()
        self.catalog_ind = 0

    def setup(self, catalog):
        """Set the full mock catalog.

        Parameters
        ----------
        catalog : containers.SourceCatalog or containers.FormedBeam
            The mock catalog to draw from.
        """
        # If the catalog is distributed, then we need to make sure that it
        # is distributed over an axis other than the object_id axis.
        if catalog.distributed:
            axis_size = {
                key: len(val)
                for key, val in catalog.index_map.items()
                if key != "object_id"
            }

            if len(axis_size) > 0:
                self.distributed_axis = max(axis_size, key=axis_size.get)

                self.log.info(
                    f"Distributing over the {self.distributed_axis} axis "
                    "to take random subsets of objects."
                )
                catalog.redistribute(self.distributed_axis)

            else:
                raise ValueError(
                    "The catalog that was provided is distributed "
                    "over the object_id axis. Unable to take a "
                    "random subset over object_id."
                )
        else:
            self.distributed_axis = None

        if "tag" in catalog.attrs:
            self.base_tag = f"{catalog.attrs['tag']}_mock_{{:05d}}"
        else:
            self.base_tag = "mock_{{:05d}}"

        self.catalog = catalog

    def process(self):
        """Draw a new random catalog.

        Returns
        -------
        new_catalog : containers.SourceCatalog or containers.FormedBeam
            A catalog of the same type as the input catalog, with a random set of
            objects.
        """
        if self.catalog_ind >= self.number:
            raise pipeline.PipelineStopIteration

        objects = self.catalog.index_map["object_id"]
        num_cat = len(objects)

        # NOTE: We need to be very careful here, the RNG is initialised at first access
        # and this is a collective operation. So we need to ensure all ranks do it even
        # though only rank=0 is going to use the RNG in this task
        rng = self.rng

        # Generate a random selection of objects on rank=0 and broadcast to all other
        # ranks
        if self.comm.rank == 0:
            ind = np.sort(rng.choice(num_cat, self.size, replace=False))
        else:
            ind = np.zeros(self.size, dtype=np.int64)
        self.comm.Bcast(ind, root=0)

        # Create new container
        new_catalog = self.catalog.__class__(
            object_id=objects[ind],
            attrs_from=self.catalog,
            axes_from=self.catalog,
            comm=self.catalog.comm,
        )

        for name in self.catalog.datasets.keys():
            if name not in new_catalog.datasets:
                new_catalog.add_dataset(name)

        if self.distributed_axis is not None:
            new_catalog.redistribute(self.distributed_axis)

        new_catalog.attrs["tag"] = self.base_tag.format(self.catalog_ind)

        # Loop over all datasets and if they have an object_id axis, select the
        # relevant objects along that axis
        for name, dset in self.catalog.datasets.items():
            if dset.attrs["axis"][0] == "object_id":
                new_catalog.datasets[name][:] = dset[:][ind]
            else:
                new_catalog.datasets[name][:] = dset[:]

        self.catalog_ind += 1

        return new_catalog


class GroupSourceStacks(task.SingleTask):
    """Accumulate many frequency stacks into a single container.

    Attributes
    ----------
    ngroup : int
        The number of frequency stacks to accumulate into a
        single container.
    """

    ngroup = config.Property(proptype=int, default=100)

    def setup(self):
        """Create a list to be populated by the process method."""
        self.stack = []
        self.nmock = 0
        self.counter = 0

        self._container_lookup = {
            containers.FrequencyStack: containers.MockFrequencyStack,
            containers.FrequencyStackByPol: containers.MockFrequencyStackByPol,
            containers.MockFrequencyStack: containers.MockFrequencyStack,
            containers.MockFrequencyStackByPol: containers.MockFrequencyStackByPol,
        }

    def process(self, stack):
        """Add a FrequencyStack to the list.

        As soon as list contains `ngroup` items, they will be collapsed
        into a single container and output by the task.

        Parameters
        ----------
        stack : containers.FrequencyStack, containers.FrequencyStackByPol,
                containers.MockFrequencyStack, containers.MockFrequencyStackByPol

        Returns
        -------
        out : containers.MockFrequencyStack, containers.MockFrequencyStackByPol
            The previous `ngroup` FrequencyStacks accumulated into a single container.
        """
        self.stack.append(stack)
        if "mock" in stack.index_map:
            self.nmock += stack.index_map["mock"].size
        else:
            self.nmock += 1

        self.log.info(
            f"Collected frequency stack.  Current size is {len(self.stack):d}."
        )

        if (len(self.stack) % self.ngroup) == 0:
            return self._reset()

        return None

    def process_finish(self):
        """Return whatever FrequencyStacks are currently in the list.

        Returns
        -------
        out : containers.MockFrequencyStack, containers.MockFrequencyStackByPol
            The remaining frequency stacks accumulated into a single container.
        """
        if len(self.stack) > 0:
            return self._reset()

        return None

    def _reset(self):
        """Combine all frequency stacks currently in the list into new container.

        Then, empty the list, reset the stack counter, and increment the group counter.
        """
        self.log.info(
            f"We have accumulated {self.nmock:d} mock realizations.  "
            f"Saving to file. [group {self.counter:03d}]"
        )

        mock = np.arange(self.nmock, dtype=np.int64)

        # Create the output container
        OutputContainer = self._container_lookup[self.stack[0].__class__]

        out = OutputContainer(
            mock=mock, axes_from=self.stack[0], attrs_from=self.stack[0]
        )

        counter_str = f"{self.counter:03d}"

        # Update tag using the hierarchy that a group contains multiple mocks,
        # and a supergroup contains multiple groups.
        if "tag" in out.attrs:
            tag = out.attrs["tag"].split("_")
            if "group" in tag:
                ig = max(ii for ii, tt in enumerate(tag) if tt == "group")
                tag[ig] = "supergroup"
                tag[ig + 1] = counter_str

            elif "mock" in tag:
                im = max(ii for ii, tt in enumerate(tag) if tt == "mock")
                tag[im] = "group"
                tag[im + 1] = counter_str

            else:
                tag.append(f"group_{counter_str}")

            out.attrs["tag"] = "_".join(tag)
        else:
            out.attrs["tag"] = f"group_{counter_str}"

        for name in self.stack[0].datasets.keys():
            if name not in out.datasets:
                out.add_dataset(name)

        # Loop over mock stacks and save to output container
        for name, odset in out.datasets.items():
            mock_count = 0

            for ss, stack in enumerate(self.stack):
                dset = stack.datasets[name]
                if dset.attrs["axis"][0] == "mock":
                    data = dset[:]
                else:
                    data = dset[np.newaxis, ...]

                for mdata in data:
                    odset[mock_count] = mdata[:]
                    mock_count += 1

        # Reset the class attributes
        self.stack = []
        self.nmock = 0
        self.counter += 1

        return out


class BinEBOSSCatalog(task.SingleTask):

    resampler = config.enum(["ngp", "cic", "tsc", "pcs"], default="tsc")
    weighted = config.Property(proptype=bool, default=False)
    shuffle = config.Property(proptype=bool, default=False)
    seed = config.Property(proptype=int, default=0)

    def _ngp(s):
        return np.where(np.abs(s) < 0.5, 1, 0)

    def _cic(s):
        return np.where(np.abs(s) < 1.0, 1 - np.abs(s), 0)

    def _tsc(s):
        return np.where(
            np.abs(s) < 0.5,
            0.75 - s**2,
            np.where(np.abs(s) < 1.5, 0.5 * (1.5 - np.abs(s)) ** 2, 0),
        )

    def _pcs(s):
        return np.where(
            np.abs(s) < 1.0,
            (1 / 6.0) * (4 - 6 * s**2 + 3 * np.abs(s) ** 3),
            np.where(np.abs(s) < 2.0, (1 / 6.0) * (2 - np.abs(s)) ** 3, 0),
        )

    def _resolve_resampler(self):
        return self.__class__.__dict__[f"_{self.resampler}"]

    def setup(self, telescope):
        self.telescope = io.get_telescope(telescope)
        self.resampler_func = self._resolve_resampler()

    def process(self, catalog, ringmap):
        rm_out = containers.RingMap(
            attrs_from=ringmap,
            axes_from=ringmap,
        )

        rm_out.redistribute("freq")

        rm_freq = rm_out.index_map["freq"][:]
        rm_freq_c = rm_freq["centre"]

        # Get local portion of the frequency axis
        nfreq = rm_out.map.local_shape[2]
        freq_offset = rm_out.map.local_offset[2]

        local_freq_slice = slice(freq_offset, freq_offset + nfreq)
        local_freq = rm_freq_c[local_freq_slice]

        rm_ra = rm_out.index_map["ra"][:]
        rm_el = rm_out.index_map["el"][:]

        if "lsd" in ringmap.attrs:

            lsd = (
                ringmap.attrs["lsd"][0]
                if isinstance(ringmap.attrs["lsd"], np.ndarray)
                else ringmap.attrs["lsd"]
            )
            epoch = self.telescope.lsd_to_unix(lsd)
            cat_ra, cat_dec = beamform.icrs_to_cirs(
                catalog["position"]["ra"], catalog["position"]["dec"], epoch
            )

        else:
            cat_ra, cat_dec = catalog["position"]["ra"], catalog["position"]["dec"]

        cat_el = np.sin(np.radians(cat_dec - 49.32))

        cat_z = catalog["redshift"]["z"]

        # Functionality for generating a shuffled catalog
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(cat_z)

        cat_freq = NU21 / (1 + cat_z)

        if self.weighted:

            w_fkp = catalog["weight"]["fkp"]
            w_cp = catalog["weight"]["cp"]
            w_sys = catalog["weight"]["sys"]
            w_noz = catalog["weight"]["noz"]

            w = w_cp * w_noz * w_sys * w_fkp

        else:
            w = np.ones_like(cat_ra)

        cat_within_el = (cat_el < rm_el.max()) & (cat_el > rm_el.min())
        cat_within_ra = (cat_ra < rm_ra.max()) & (cat_ra > rm_ra.min())
        cat_within_local_freq = (cat_freq < local_freq.max()) & (
            cat_freq > local_freq.min()
        )

        cat_within_bounds = cat_within_el & cat_within_ra & cat_within_local_freq

        cat_ra_within = cat_ra[cat_within_bounds]
        cat_el_within = cat_el[cat_within_bounds]
        cat_freq_within = cat_freq[cat_within_bounds]

        dra = np.median(np.abs(np.diff(rm_ra)))
        dza = np.median(np.abs(np.diff(rm_el)))
        dnu = np.median(np.abs(np.diff(rm_freq_c)))

        ra_min = rm_ra.min()
        za_min = rm_el.min()

        for ii in range(np.sum(cat_within_bounds)):

            raw = cat_ra_within[ii]
            elw = cat_el_within[ii]
            nuw = cat_freq_within[ii]

            ra_ind = np.rint((raw - ra_min) / dra).astype(np.int64)
            za_ind = np.rint((elw - za_min) / dza).astype(np.int64)

            ra_sl = slice(max(ra_ind - 6, 0), min(ra_ind + 6, rm_ra.size))
            el_sl = slice(max(za_ind - 4, 0), min(za_ind + 4, rm_el.size))

            s_ra = (raw - rm_ra[ra_sl]) / dra
            s_el = (elw - rm_el[el_sl]) / dza
            s_nu = (nuw - local_freq) / dnu

            srag, selg, snug = np.meshgrid(s_ra, s_el, s_nu)

            windowed_obj = (
                w[ii]
                * (
                    self.resampler_func(srag)
                    * self.resampler_func(selg)
                    * self.resampler_func(snug)
                ).T
            )

            if 0 in windowed_obj.shape:
                continue

            self.log.info(
                f"Slices are {ra_sl}, {el_sl} and window shape is {windowed_obj.shape}."
            )

            rm_out.map.local_data[:, :, :, ra_sl, el_sl] += windowed_obj[
                np.newaxis, np.newaxis, ...
            ]

        return rm_out

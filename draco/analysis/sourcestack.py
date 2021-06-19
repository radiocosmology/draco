"""Source Stack Analysis Tasks"""

import numpy as np
from mpi4py import MPI

from caput import config, pipeline
from cora.util import units

from ..util.tools import invert_no_zero
from ..util.random import RandomTask
from ..core import task, containers

# Constants
NU21 = units.nu21
C = units.c


class SourceStack(task.SingleTask):
    """Stack the product of `draco.analysis.BeamForm` accross sources.

    For this to work BeamForm must have been run with `collapse_ha = True` (default).

    Attributes
    ----------
    freqside : int
        Number of frequency bins to keep on each side of quasar
        when stacking.
    """

    # Number of frequencies to keep on each side of quasar RA
    # Pick only frequencies around the quasar (50 on each side)
    freqside = config.Property(proptype=int, default=50)

    def process(self, formed_beam):
        """Receives a formed beam object and stack across sources.

        Parameters
        ----------
        formed_beam : `containers.FormedBeam` object
            Formed beams to stack over sources.

        Returns
        -------
        qstack : `containers.FrequencyStack` object
            The stack of sources.
        """
        # Get communicator
        comm = formed_beam.comm

        # Ensure formed_beam is distributed in sources
        formed_beam.redistribute("object_id")

        # local shape and offset
        loff = formed_beam.beam.local_offset[0]
        lshape = formed_beam.beam.local_shape[0]

        # Frequency axis
        freq = formed_beam.freq
        nfreq = len(freq)

        # Polarisation axis
        pol = formed_beam.pol
        npol = len(pol)

        # Frequency of quasars
        qso_freq = NU21 / (formed_beam["redshift"]["z"] + 1.0)  # MHz.
        # Size of quasar stack array
        self.nstack = 2 * self.freqside + 1

        # Construct frequency offset axis (for qstack container)
        self.stack_axis = np.copy(
            formed_beam.frequency[
                int(nfreq / 2) - self.freqside : int(nfreq / 2) + self.freqside + 1
            ]
        )
        self.stack_axis["centre"] = (
            self.stack_axis["centre"] - self.stack_axis["centre"][self.freqside]
        )

        # Get f_mask and qs_indices
        freqdiff = freq[np.newaxis, :] - qso_freq[:, np.newaxis]
        # Stack axis bin edges to digitize each quasar at.
        stackbins = self.stack_axis["centre"] + 0.5 * self.stack_axis["width"]
        stackbins = np.append(
            stackbins,
            self.stack_axis["centre"][-1] - 0.5 * self.stack_axis["width"][-1],
        )
        # Index of each frequency in stack axis, for each quasar
        qs_indices = np.digitize(freqdiff, stackbins) - 1
        # Indices to be processed in full frequency axis for each quasar
        f_mask = (qs_indices >= 0) & (qs_indices < self.nstack)
        # Only quasars in the frequency range of the data.
        qso_mask = (np.sum(f_mask, axis=1) > 0).astype(bool)

        # Reduce mask and indices to this process range
        # to reduce striding through this data
        qso_mask = qso_mask[loff : loff + lshape]
        qs_indices = qs_indices[loff : loff + lshape]
        f_mask = f_mask[loff : loff + lshape]

        # Container to hold the stack
        if npol > 1:
            qstack = containers.FrequencyStackByPol(
                freq=self.stack_axis, pol=pol, attrs_from=formed_beam
            )
        else:
            qstack = containers.FrequencyStack(
                freq=self.stack_axis, attrs_from=formed_beam
            )

        # Loop over polarisations
        for pp, pstr in enumerate(pol):

            # Quasar stack array.
            quasar_stack = np.zeros(self.nstack, dtype=np.float)
            quasar_weight = np.zeros(self.nstack, dtype=np.float)

            qcount = 0  # Quasar counter
            # For each quasar in the range of this process
            for lq, gq in formed_beam.beam[:].enumerate(axis=0):
                if not qso_mask[lq]:
                    # Quasar not in the data redshift range
                    continue
                qcount += 1
                # Indices and slice for frequencies included in the stack.
                f_indices = np.arange(nfreq, dtype=np.int32)[f_mask[lq]]
                f_slice = np.s_[f_indices[0] : f_indices[-1] + 1]

                quasar_stack += np.bincount(
                    qs_indices[lq][f_slice],
                    weights=(
                        formed_beam.beam[gq, pp][f_slice]
                        * formed_beam.weight[gq, pp][f_slice]
                    ),
                    minlength=self.nstack,
                )

                quasar_weight += np.bincount(
                    qs_indices[lq][f_slice],
                    weights=formed_beam.weight[gq, pp][f_slice],
                    minlength=self.nstack,
                )

            # Gather quasar stack for all ranks. Each contains the sum
            # over a different subset of quasars.
            quasar_stack_full = np.zeros(
                comm.size * self.nstack, dtype=quasar_stack.dtype
            )
            quasar_weight_full = np.zeros(
                comm.size * self.nstack, dtype=quasar_weight.dtype
            )
            # Gather all ranks
            comm.Allgather(quasar_stack, quasar_stack_full)
            comm.Allgather(quasar_weight, quasar_weight_full)

            # Determine the index for the output container
            oslc = (pp, slice(None)) if npol > 1 else slice(None)

            # Sum across ranks
            qstack.weight[oslc] = np.sum(
                quasar_weight_full.reshape(comm.size, self.nstack), axis=0
            )
            qstack.stack[oslc] = np.sum(
                quasar_stack_full.reshape(comm.size, self.nstack), axis=0
            ) * invert_no_zero(qstack.weight[oslc])

            # Gather all ranks of qcount. Report number of quasars stacked
            full_qcount = comm.reduce(qcount, op=MPI.SUM, root=0)
            if comm.rank == 0:
                self.log.info(
                    f"Number of quasars stacked for pol {pstr}: {full_qcount}"
                )

        return qstack


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
        catalog : containers.SourceCatalog
            The mock catalog to draw from.
        """
        self.catalog = catalog
        self.base_tag = f'{catalog.attrs.get("tag", "mock")}_{{}}'

    def process(self):
        """Draw a new random catalog.

        Returns
        -------
        new_catalog : containers.SourceCatalog subclass
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
            ind = rng.choice(num_cat, self.size, replace=False)
        else:
            ind = np.zeros(self.size, dtype=np.int64)
        self.comm.Bcast(ind, root=0)

        new_catalog = self.catalog.__class__(
            object_id=objects[ind], attrs_from=self.catalog
        )
        new_catalog.attrs["tag"] = self.base_tag.format(self.catalog_ind)

        # Loop over all datasets and if they have an object_id axis, select the
        # relevant objects along that axis
        for name, dset in new_catalog.datasets.items():
            if dset.attrs["axis"][0] == "object_id":
                dset[:] = self.catalog.datasets[name][ind]

        self.catalog_ind += 1

        return new_catalog

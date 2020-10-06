# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import h5py
import numpy as np
import healpy as hp
from mpi4py import MPI
import random

from caput import mpiarray, config, pipeline
from cora.util import units
from ch_util import tools, ephemeris, rfi
from drift.telescope import cylbeam

from ..util.tools import invert_no_zero
from ..util._fast_tools import beamform
from ..core import task, containers, io

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
    # Remove mean of each slice before stacking?
    remove_mean = config.Property(proptype=bool, default=False)
    weight = config.enum(['uniform', 'inverse_variance'],
                         default='inverse_variance')
    weight_threshold = config.Property(proptype=float, default=None) #100000
    beam_threshold = config.Property(proptype=float, default=None) #0.02
    freq_lims = config.Property(proptype=list, default=None)
    z_lims = config.Property(proptype=list, default=None)
    random_sample = config.Property(proptype=int, default=None)

    def setup(self):

        if self.freq_lims is not None:
            # Overwrite z_lims
            self.z_lims = [NU21 / freq - 1. for freq in self.freq_lims]

    def process(self, formed_beam):
        """ Receives a formed beam object and stack across sources.

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
        nqso = formed_beam.beam.global_shape[0]

        # Frequency axis
        freq = formed_beam.freq
        #static_mask = rfi.frequency_mask(freq)
        nfreq = len(freq)

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

        if self.z_lims is not None:
            # Redshift limits. Order is not important.
            z_mask = (formed_beam["redshift"]["z"][:] >= np.amin(self.z_lims))
            z_mask = z_mask & (formed_beam["redshift"]["z"][:] <= np.amax(self.z_lims))
            qso_mask = qso_mask & z_mask

        if self.random_sample is not None:
            # Draw random QSOs to stack. This is the lazy and slow way. Just modify qso_mask.
            if self.random_sample > nqso:
                raise RuntimeError("Too many quasars to sample: {0} out of {1}".format(
                                                                self.random_sample, nqso))
            if comm.rank == 0:
                samples = np.array(random.sample(range(nqso), self.random_sample)).astype(int)
            else:
                samples = np.zeros(self.random_sample, dtype=int)
            # Ensure all ranks have the same sample
            comm.Bcast(samples, root=0)
            sample_mask = np.zeros(nqso, dtype=bool)
            sample_mask[samples] = True
            # Does not garanty self.random_samples valid QSOs, but close enough.
            qso_mask = qso_mask & sample_mask

        # Reduce mask and indices to this process range
        # to reduce striding through this data
        qso_mask = qso_mask[loff : loff + lshape]
        qs_indices = qs_indices[loff : loff + lshape]
        f_mask = f_mask[loff : loff + lshape]

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

            this_weight_mask = np.ones_like(formed_beam.weight[gq, 0][f_slice], dtype=bool)
            if self.weight_threshold is not None:
                this_weight_mask = (formed_beam.weight[gq, 0][f_slice] > self.weight_threshold)
            if self.beam_threshold is not None:
                this_weight_mask *= (abs(formed_beam.beam[gq, 0][f_slice]) < self.beam_threshold)
#            this_weight_mask *= ~static_mask[f_slice]
            this_slice = np.where(this_weight_mask, formed_beam.beam[gq, 0][f_slice], 0.)
            if self.weight=='uniform':
                this_slice_weight = np.where(this_weight_mask, 1., 0.)
            else:
                this_slice_weight = np.where(this_weight_mask, formed_beam.weight[gq, 0][f_slice], 0.)
            this_slice *= this_slice_weight
            if self.remove_mean:
                if not np.isfinite(this_slice[this_weight_mask]).all():
                    raise RuntimeError("Found infinity in slice.")
                this_slice[this_weight_mask] -= np.mean(this_slice[this_weight_mask])

#            if qcount==10:
#                print('Hau', quasar_stack.shape, qs_indices[lq][f_slice].shape, this_slice.shape, self.nstack)
            quasar_stack += np.bincount(
                qs_indices[lq][f_slice],
                weights=this_slice,
                minlength=self.nstack)

            quasar_weight += np.bincount(
                qs_indices[lq][f_slice],
                weights=this_slice_weight,
                minlength=self.nstack)

            #mpiutil.barrier()  # TODO: I am not sure I need a barrier here.
        # Gather quasar stack for all ranks. Each contains the sum
        # over a different subset of quasars.
        quasar_stack_full = np.zeros(comm.size * self.nstack, dtype=quasar_stack.dtype)
        quasar_weight_full = np.zeros(
            comm.size * self.nstack, dtype=quasar_weight.dtype
        )
        # Gather all ranks
        comm.Allgather(quasar_stack, quasar_stack_full)
        comm.Allgather(quasar_weight, quasar_weight_full)

        # Container to hold the stack
        qstack = containers.FrequencyStack(freq=self.stack_axis)
        # Sum across ranks
        qstack.weight[:] = np.sum(
            quasar_weight_full.reshape(comm.size, self.nstack), axis=0
        )
        qstack.stack[:] = np.sum(
            quasar_stack_full.reshape(comm.size, self.nstack), axis=0
        ) * invert_no_zero(qstack.weight[:])

        # Gather all ranks of qcount. Report number of quasars stacked
        full_qcount = comm.reduce(qcount, op=MPI.SUM, root=0)
        full_qcount = comm.bcast(full_qcount, root=0)
        if comm.rank == 0:
            self.log.info("Number of quasars stacked: {0}".format(full_qcount))
        qstack.attrs['n_stacked'] = full_qcount

        return qstack


#static_mask = np.array([
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, True, True, True, True, True, True, True, True, True, True, True, True, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, False, False, False, False, False, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,
#    True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, False,
#    False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False,
#    False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False,
#    False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, False, False, False, False, False, False, False, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
#    True, True, True, True, False, False, False, False, False, False, False, False, False, True, True, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,
#    True, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False,
#    False, False, False, False, False, True, True, False, True, True, True, True, True, False, False, False,
#    True, True, True, True, True, False, True, True, True, False, False, False, True, True, True, True,
#    True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, False,
#    False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False,
#    True, True, False, False, False, False, False, False, False, True, True, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])

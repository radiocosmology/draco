"""Construct sidereal stream including template-based beam perturbations.
"""

import numpy as np

from caput import config
from caput.time import STELLAR_S

from drift.telescope.external_beam import CylinderPerturbedTemplates

from ..core import task, containers, io
from ..util import tools, random


class MakePerturbedBeamStream(task.SingleTask):
    """Construct sidereal stream with beam perturbations with specified amplitudes.

    Attributes
    ----------
    pert_vals : list
        List of floats corresponding to beam perturbation amplitudes. A sidereal stream
        will be constructed assuming that the true primary beam A is
            A_base + \sum_i a_i A_pert ,
        where a_i are the elements of pert_vals and A_pert are the beam perturbation
        templates. pert_vals must therefore have the same length as number of
        perturbation templates in the specified telescope class.
    """

    pert_vals = config.Property(proptype=list)

    def setup(self, bt_pert, bt_unpert):
        """Set the telescope instance.

        Parameters
        -------------
        bt_pert : beamtransfer.BeamTransfer or manager.ProductManager
            Specifes telescope class that includes perturbations.
        bt_unpert : beamtransfer.BeamTransfer or manager.ProductManager
            Specifes telescope class that only include base beam model.
        """
        self.tel_pert = io.get_telescope(bt_pert)
        self.tel_unpert = io.get_telescope(bt_unpert)

        if self.tel_pert.__class__ is not CylinderPerturbedTemplates:
            raise InputError("Must specify perturbed telescope class!")

        if len(self.pert_vals) != self.tel_pert.n_pert:
            raise InputError(
                "Length of pert_vals must equal number of beam perturbations!"
            )

    def process(self, sstream_full):
        """Convert full sidereal stream into perturbed stream.

        Parameters
        ----------
        sstream_full : :class:`containers.SiderealStream`
            Sidereal stream that separately includes base values and perturbation
            values.

        Returns
        -------
        sstream_out : :class:`containers.SiderealStream`
            Sidereal stream with sum of base and perturbation values.
        """

        # Redistribute input sstream over freq, and get local sections of vis and
        # weights
        sstream_full.redistribute("freq")
        vis_full = sstream_full.vis[:]
        weight_full = sstream_full.weight[:]

        # Make container for output sstream, redistribute over freq, and get local
        # sections of vis and weights
        sstream_out = containers.SiderealStream(
            freq=sstream_full.index_map["freq"],
            ra=sstream_full.index_map["ra"],
            input=self.tel_unpert.input_index,
            prod=self.tel_unpert.index_map_prod,
            stack=self.tel_unpert.index_map_stack,
            reverse_map_stack=self.tel_unpert.reverse_map_stack,
            distributed=True,
            comm=sstream_full.comm,
        )
        sstream_out.redistribute("freq")
        vis_out = sstream_out.vis[:]
        weight_out = sstream_out.weight[:]

        # Get beamclass of each input in the full sstream, and also in unpert sstream
        bc_a = self.tel_pert.beamclass[sstream_full.prodstack["input_a"]]
        bc_b = self.tel_pert.beamclass[sstream_full.prodstack["input_b"]]
        bc_a_u = self.tel_unpert.beamclass[sstream_out.prodstack["input_a"]]
        bc_b_u = self.tel_unpert.beamclass[sstream_out.prodstack["input_b"]]

        # Transfer unperturbed visibilities and corresponding weights into output
        # sstream
        mask = (bc_a <= 1) & (bc_b <= 1)
        if not (np.allclose(bc_a[mask], bc_a_u) and np.allclose(bc_b[mask], bc_b_u)):
            raise RuntimeError("Slicing error in perturbed sstream!")
        vis_out[:] = vis_full[:, mask]
        weight_out[:] = weight_full[:, mask]

        # Transfer visibilities with one perturbed input and one unperturbed input
        for p in range(1, self.tel_pert.n_pert + 1):

            # a=unpert, b=pert (order pert_vals)
            mask = (bc_a <= 1) & ((bc_b == 2 * p) | (bc_b == 2 * p + 1))
            if not (
                np.allclose(bc_a[mask], bc_a_u) and np.allclose(bc_b[mask] % 2, bc_b_u)
            ):
                raise RuntimeError("Slicing error in perturbed sstream!")
            vis_out[:] += self.pert_vals[p - 1] * vis_full[:, mask]

            # a=pert, b=unpert (order pert_vals)
            mask = ((bc_a == 2 * p) | (bc_a == 2 * p + 1)) & (bc_b <= 1)
            if not (
                np.allclose(bc_b[mask], bc_b_u) and np.allclose(bc_a[mask] % 2, bc_a_u)
            ):
                raise RuntimeError("Slicing error in perturbed sstream!")
            vis_out[:] += self.pert_vals[p - 1] * vis_full[:, mask]

            # a,b=pert (order pert_vals^2)
            for pp in range(p, self.tel_pert.n_pert + 1):
                mask = ((bc_a == 2 * p) | (bc_a == 2 * p + 1)) & (
                    (bc_b == 2 * pp) | (bc_b == 2 * pp + 1)
                )
                if not (
                    np.allclose(bc_a[mask] % 2, bc_a_u)
                    and np.allclose(bc_b[mask] % 2, bc_b_u)
                ):
                    raise RuntimeError("Slicing error in perturbed sstream!")
                vis_out[:] += (
                    self.pert_vals[p - 1] * self.pert_vals[pp - 1] * vis_full[:, mask]
                )

        return sstream_out

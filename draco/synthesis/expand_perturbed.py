
"""Tasks for expanding sidereal stream data with perturbed beams.

A typical pattern would be to turn a sidereal stream from the :class:`SimulateSidereal` task,
then generate one or more non-zero perturbation values with the :class:`GeneratePerturbation`
or :class:`GenerateSinglePerturbation` tasks then apply the perturbations and expand
nominally redundant products to first order with :class:`ExpandPerturbedProducts`
or to second order with :class:`ExpandPerturbedProducts2ndOrder`. Optionally, individual
components of the expanded sidereal stream can be output via :class:`OutputPertStructure`.
Then, the expanded stream can be passed on to make timestreams as in unperturbed sims.

Tasks
=====

.. autosummary::
    :toctree:

    GeneratePerturbation
    GenerateSinglePerturbation
    ExpandPerturbedProducts
    ExpandPerturbedProducts2ndOrder
    OutputPertStructure

"""

import numpy as np

from cora.util import hputil
from caput import mpiutil, pipeline, config, mpiarray

from ..core import containers, task



class GeneratePerturbation(task.SingleTask):
    """ Generate small, random number perturbatons to input to the
        ExpandPerturbedProducts task.
    """

    # Define multiplier value. This is usually set in the .yaml file used to run the pipeline.
    # The idea is it sets the overall size of the beam perturbations, allowing
    # us to ensure smaller values than the usual size of a numpy random number.
    mult = config.Property(proptype = float, default = 0.01)
    def setup(self,telescope):
        """Get a reference to the telescope class.

        Parameters
        -------------
        tel : : class: `drift.core.TransitTelescope`
            Telescope object.
        """
        self.telescope = telescope

    def process(self, sstream, map_):
        # Need to write container and add saving to container to this.
        # Then will be ready to roll out in first form.
        # Drop multiplier if that's not working
        """ Generate perturbations for each feed.
        """
        tel = self.telescope

        # Determine total number of inputs from sstream
        ninput = len(sstream.input)
        # Determine the number of inputs per perturbation.
        ninputperpert = ninput / tel.npert

        # There's a slight chance this is going to freak out for more than 1
        # perturbation. If it does, we'll modify it.
        pertlistlen = ninputperpert
        #print "Pert List Len", pertlistlen

        # Define frequency map for upcoming container.
        freqmap = map_.index_map['freq'][:]

        # Calculate random numbers to be perturbation values. Set the general size using input mult.
        #perturbations = np.random.standard_normal(pertlistlen) * self.mult
        # Do this only for communicator rank 0 (the first one) so that you don't
        # have four independent sets of perturbations running around.
        comm = sstream.comm
        if comm.rank == 0:
            perturbations = np.random.standard_normal(pertlistlen) *0.01
        else:
            perturbations = None

        perturbations = comm.bcast(perturbations, root=0)
        # Calculate the desired number of products
        desired_prod = np.array([ (fi, fj) for fi in range(ninputperpert) for fj in range(fi, ninputperpert)])

        # Define a BeamPerturbation container to hold the beam perturbations
        perturbation_list = containers.BeamPerturbation(freq = freqmap, input = pertlistlen,
                            distributed = True, prod = desired_prod, comm = map_.comm)
        # Add beam perturbation values to container.
        perturbation_list.pert[:] = perturbations

        return perturbation_list


class GenerateSinglePerturbation(task.SingleTask):
    """ Generate small, random number perturbation for just 1 input to send to the
        ExpandPerturbedProducts task.

        This is a bit limited in scope, but useful for getting an idea of where
        the perturbation is showing up in later analyses.
    """

    # Define multiplier value. This is usually set in the .yaml file used to run the pipeline.
    # The idea is it sets the overall size of the beam perturbations, allowing
    # us to ensure smaller values than the usual size of a numpy random number.
    pert_val = config.Property(proptype = float, default = 0.01)
    pert_index = config.Property(proptype = float, default = 0)
    def setup(self,telescope):
        """Get a reference to the telescope class.

        Parameters
        -------------
        tel : : class: `drift.core.TransitTelescope`
            Telescope object.
        """
        self.telescope = telescope

    def process(self, sstream, map_):
        # Need to write container and add saving to container to this.
        # Then will be ready to roll out in first form.
        # Drop multiplier if that's not working
        """ Generate perturbations for each feed.
        """
        tel = self.telescope

        # Determine total number of inputs from sstream
        ninput = len(sstream.input)
        # Determine the number of inputs per perturbation.
        ninputperpert = ninput / tel.npert

        # There's a slight chance this is going to freak out for more than 1
        # perturbation. If it does, we'll modify it.
        pertlistlen = ninputperpert
        #print "Pert List Len", pertlistlen

        # Define frequency map for upcoming container.
        freqmap = map_.index_map['freq'][:]

        # Calculate random numbers to be perturbation values. Set the general size using input mult.
        #perturbations = np.random.standard_normal(pertlistlen) * self.mult
        # Do this only for communicator rank 0 (the first one) so that you don't
        # have four independent sets of perturbations running around.
        comm = sstream.comm
        if comm.rank == 0:
            # Set the perturbation matrix to be all zeros
            perturbations = np.zeros(pertlistlen)
            # Set just one entry (which you selected) to be the perturbation value you put in.
            perturbations[pert_index] = self.pert_val
        else:
            perturbations = None

        perturbations = comm.bcast(perturbations, root=0)
        # Calculate the desired number of products
        desired_prod = np.array([ (fi, fj) for fi in range(ninputperpert) for fj in range(fi, ninputperpert)])

        # Define a BeamPerturbation container to hold the beam perturbations
        perturbation_list = containers.BeamPerturbation(freq = freqmap, input = pertlistlen,
                            distributed = True, prod = desired_prod, comm = map_.comm)
        # Add beam perturbation values to container.
        perturbation_list.pert[:] = perturbations

        return perturbation_list

class ExpandPerturbedProducts(task.SingleTask):
    """Un-wrap collated products to full triangle.
    Combine unperturbed and perturbed beam components into one set of output products.
    """

    def setup(self, telescope):
        """Get a reference to the telescope class.

        Parameters
        ----------
        tel : :class:`drift.core.TransitTelescope`
            Telescope object.
        """
        self.telescope = telescope

    def process(self, sstream, map_,perturbations):
    #def process(self,sstream,map_,perturbations=0)
        """Transform a sidereal stream to having a full product matrix.

        Parameters
        ----------
        sstream : :class:`containers.SiderealStream`
            Sidereal stream to unwrap.

        Returns
        -------
        new_sstream : :class:`containers.SiderealStream`
            Unwrapped sidereal stream with unperturbed and perturbed components of products combined.
        """
        tel = self.telescope

        # Load and redistribute perturbations from input list.
        # Generally, this list will come from GeneratePertubation, but it may
        # also be any other BeamPerturbation container generated any other way.
        perturbations.redistribute('freq')
        pert_arr = perturbations.pert[:]
        # Select only set 0 - this avoids the whole MPI Communicator thing.
        feed_perts = pert_arr[0,:]

        # Redistribute sstream over freq
        sstream.redistribute('freq')

        # Define total ninput based on sstream
        ninput = len(sstream.input)

        # Determine ninput per perturbation based on ninput/n perturbations.
        ninputperpert = len(sstream.input) / tel.npert

        # Define array with the size of all of the products from sstream.
        #all_prod = np.array([ (fi, fj) for fi in range(ninput) for fj in range(fi, ninput)])

        # Define array with the size of the desired number of total products from sstream.
        desired_prod = np.array([ (fi, fj) for fi in range(ninputperpert) for fj in range(fi, ninputperpert)])

        # Set l max, m max, number of frequencies,  number of polarizations, and sky resolution in the same manner as sstream.
        # We aren't changing  any of them, but we need have them defined here to make new_stream.
        lmax = tel.lmax
        mmax = tel.mmax
        nfreq = tel.nfreq
        npol = tel.num_pol_sky

        # Set the minimum resolution required for the sky.
        ntime = 2 * mmax + 1

        freqmap = map_.index_map['freq'][:]
        row_map = map_.map[:]

        if (tel.frequencies != freqmap['centre']).all():
            raise RuntimeError('Frequencies in map do not match those in Beam Transfers.')

        # Define new sidereal stream container for the new dimensions. We now have ninputperpert inputs and desired_prod products.
        new_stream = containers.SiderealStream(freq=freqmap, ra=ntime, input=ninputperpert,
                                    prod=desired_prod, distributed=True, comm=map_.comm)

        new_stream.redistribute('freq')
        new_stream.vis[:] = 0.0
        new_stream.weight[:] = 0.0

        # Iterate over all feed pairs and work out which is the correct index in
        # the sidereal stack.
        for pi, (fi, fj) in enumerate(desired_prod):

            # Define product index for the unperturbed component of the given product
            unique_ind = self.telescope.feedmap[fi, fj]
            # Also conjugate just in case
            conj = self.telescope.feedconj[fi, fj]

            # unique_ind is less than zero it has masked out
            if unique_ind < 0:
                continue
            # if either fi or fj are in fact perturbations, skip ahead
            elif self.telescope.beamclass[fi] > 1:
                continue
            elif self.telescope.beamclass[fj] > 1:
                continue

            # Select visibility corresponding to unique_ind product.
            prod_stream_noconj = sstream.vis[:, unique_ind]
            # Conjugate if necessary.
            prod_stream  = prod_stream_noconj.conj() if conj else prod_stream_noconj

            # Loop over each perturbation, starting from 1 as 0 would be the unperturbed component.
            for pert_index in range(1,self.telescope.npert):
                # Define the feed index in sstream for the perturbed component corresponding to each feed.
                fi_pert = fi + ninputperpert * pert_index
                fj_pert = fj + ninputperpert * pert_index

                # Define the product index for the fi & fj perturbed component.
                # By default we won't put this component in new_stream as we're working to first order.
                # However, its' good to track what it is.
                pertfifj_ind = self.telescope.feedmap[fi_pert, fj_pert]
                pertfifj_conj = self.telescope.feedconj[fi_pert,fj_pert]

                # Define product index in sstream for the fi perturbed component.
                pertfi_ind = self.telescope.feedmap[fi_pert, fj]
                # Select the visibility corresponding to the fi perturbed component
                # Multiply it by the perturbation value corresponding to the fi in this perturbation
                pertfi_stream = sstream.vis[:, pertfi_ind] * feed_perts[fi_pert - ninputperpert * pert_index]

                # Define product index for fj perturbed component
                pertfj_ind = self.telescope.feedmap[fi, fj_pert]
                # Select visibility corresponding to fj perturbed component and multiply by perturbation value
                pertfj_stream = sstream.vis[:, pertfj_ind] * feed_perts[fj - ninputperpert * pert_index]

                # Conjugates if necessary
                pertfi_conj = self.telescope.feedconj[fi_pert, fj]
                pertfj_conj = self.telescope.feedconj[fi, fj_pert]

                # Add fi perturbation component to unperturbed sstream
                prod_stream = prod_stream + (pertfi_stream.conj() if pertfi_conj else pertfi_stream)

                # Add fj perturbation component to unperturbed sstream + fi perturbation sstream
                prod_stream = prod_stream +  (pertfj_stream.conj() if pertfj_conj else pertfj_stream)


            # Put prod_stream into new container new_stream.
            new_stream.vis[:, pi] = prod_stream
            new_stream.weight[:, pi] = 1.0

        return new_stream


class ExpandPerturbedProducts2ndOrder(task.SingleTask):
    """Un-wrap collated products to full triangle.
        Combine perturbed and Unperturbed products, keeping the term perturbed
        in both feed i and j (which is therefore 2nd order).
    """

    def setup(self, telescope):
        """Get a reference to the telescope class.

        Parameters
        ----------
        tel : :class:`drift.core.TransitTelescope`
            Telescope object.
        """
        self.telescope = telescope
    def process(self, sstream, map_,perturbations):
    #def process(self,sstream,map_,perturbations=0)
        """Transform a sidereal stream to having a full product matrix.

        Parameters
        ----------
        sstream : :class:`containers.SiderealStream`
            Sidereal stream to unwrap.

        Returns
        -------
        new_sstream : :class:`containers.SiderealStream`
            Unwrapped sidereal stream with unperturbed and perturbed components of products combined.
        """
        tel = self.telescope

        # Load and redistribute perturbations from input list.
        # Generally, this list will come from GeneratePertubation, but it may
        # also be any other BeamPerturbation container generated any other way.
        perturbations.redistribute('freq')
        pert_arr = perturbations.pert[:]
        # Select only set 0 - this avoids the whole MPI Communicator thing.
        feed_perts = pert_arr[0,:]

        # Redistribute sstream over freq
        sstream.redistribute('freq')

        # Define total ninput based on sstream
        ninput = len(sstream.input)

        # Determine ninput per perturbation based on ninput/n perturbations.
        ninputperpert = len(sstream.input) / tel.npert

        # Define array with the size of all of the products from sstream.
        #all_prod = np.array([ (fi, fj) for fi in range(ninput) for fj in range(fi, ninput)])

        # Define array with the size of the desired number of total products from sstream.
        desired_prod = np.array([ (fi, fj) for fi in range(ninputperpert) for fj in range(fi, ninputperpert)])

        # Set l max, m max, number of frequencies,  number of polarizations, and sky resolution in the same manner as sstream.
        # We aren't changing  any of them, but we need have them defined here to make new_stream.
        lmax = tel.lmax
        mmax = tel.mmax
        nfreq = tel.nfreq
        npol = tel.num_pol_sky

        # Set the minimum resolution required for the sky.
        ntime = 2 * mmax + 1

        freqmap = map_.index_map['freq'][:]
        row_map = map_.map[:]

        if (tel.frequencies != freqmap['centre']).all():
            raise RuntimeError('Frequencies in map do not match those in Beam Transfers.')

        # Define new sidereal stream container for the new dimensions. We now have ninputperpert inputs and desired_prod products.
        new_stream = containers.SiderealStream(freq=freqmap, ra=ntime, input=ninputperpert,
                                    prod=desired_prod, distributed=True, comm=map_.comm)

        new_stream.redistribute('freq')
        new_stream.vis[:] = 0.0
        new_stream.weight[:] = 0.0

        # Iterate over all feed pairs and work out which is the correct index in
        # the sidereal stack.

        for pi, (fi, fj) in enumerate(desired_prod):
            # Define product index for the unperturbed component of the given product
            unique_ind = self.telescope.feedmap[fi, fj]
            # Also conjugate just in case
            conj = self.telescope.feedconj[fi, fj]

            # unique_ind is less than zero it has masked out
            if unique_ind < 0:
                continue
            # if either fi or fj are in fact perturbations, skip ahead
            elif self.telescope.beamclass[fi] > 1:
                continue
            elif self.telescope.beamclass[fj] > 1:
                continue

            # Select visibility corresponding to unique_ind product.
            prod_stream_noconj = sstream.vis[:, unique_ind]
            # Conjugate if necessary.
            prod_stream  = prod_stream_noconj.conj() if conj else prod_stream_noconj

            for pert_index in range(1, self.telescope.npert):
                # Define the feed index in sstream for the perturbed component corresponding to each feed.
                fi_pert = fi + ninputperpert * pert_index
                fj_pert = fj + ninputperpert * pert_index

                # Define the product index for the fi & fj perturbed component.
                pertfifj_ind = self.telescope.feedmap[fi_pert, fj_pert]

                # Select the visibility corresponding to this product.
                # Multiply it by both the fi and fj perturbation values.
                pertfifj_stream = sstream.vis[:,pertfifj_ind] * feed_perts[fi_pert - ninputperpert * pert_index] * feed_perts[fj_pert - ninputperpert * pert_index]

                # Define the product index for the only fi perturbed component
                pertfi_ind = self.telescope.feedmap[fi_pert, fj]
                # Select the visibility corresponding to this product.
                # Multiply it by both the fi and fj perturbation values.
                pertfi_stream = sstream.vis[:,pertfi_ind]*feed_perts[fi_pert - ninputperpert * pert_index]

                # Define the product index for the only fj perturbed component
                pertfj_ind = self.telescope.feedmap[fi, fj_pert]
                # Select the visibility corresponding to this product.
                # Multiply it by both the fi and fj perturbation values.
                pertfj_stream = sstream.vis[:,pertfj_ind]*feed_perts[fj - ninputperpert * pert_index]

                # Conjugate indices if necessary.
                pertfi_conj = self.telescope.feedconj[fi_pert,fj]
                pertfj_conj = self.telescope.feedconj[fi, fj_pert]
                pertfifj_conj = self.telescope.feedconj[fi_pert,fj_pert]

                # Add the fi perturbed component to the unperturbed stream.
                prod_stream = prod_stream + (pertfi_stream.conj() if pertfi_conj else pertfi_stream)
                # Add the fj perturbec domponent to the unperturbed + fi perturbed stream.
                prod_stream = prod_stream +  (pertfj_stream.conj() if pertfj_conj else pertfj_stream)
                # Add the both perturbed component to the unperturbed + fi perturbed + fj perturbed stream.
                prod_stream = prod_stream + (pertfifj_stream.conj() if pertfifj_conj else pertfifj_stream )

            # Put the new component into the new sidereal stream.
            new_stream.vis[:, pi] = prod_stream
            new_stream.weight[:, pi] = 1.0

        return new_stream



class OutputPertStructure(task.SingleTask):
    """ Output the individual unperturbed, fi perturbed, and fj perturbed
        components of a sidereal stream.
        DOES NOT APPLY PERTURBATIONS because this is really designed for analyses
        which are trying to solve for perturbations.
    """
    pert_feed = config.Property(proptype=float, default=0)

    def setup(self, telescope):
        """Get a reference to the telescope class.

        Parameters
        ----------
        tel : :class:`drift.core.TransitTelescope`
            Telescope object.
        """
        self.telescope = telescope


    def process(self, sstream, map_,):
        """ Seperate a sidereal stream into unperturbed, perturbed in fi,
        perturbed in fj, and second order (perturbed in fi and fj).

        Parameters
        ----------
        sstream : :class:`containers.SiderealStream`
            Sidereal stream to unwrap.

        Returns
        -------
        f_stream : :class:`containers.SiderealStream`
            Individual unpert or pert sidereal stream.
        """
        tel = self.telescope
        # Redistribute sstream.
        sstream.redistribute('freq')

        # Determine n input from sstream.
        ninput = len(sstream.input)
        # Determine ninput per perturbation.
        ninputperpert = ninput / tel.npert


        all_prod = np.array([ (fi, fj) for fi in range(ninput) for fj in range(fi, ninput)])
        #print len(all_prod)

        # Define array with the size of the desired number of total products from sstream.
        desired_prod = np.array([ (fi, fj) for fi in range(ninputperpert) for fj in range(fi, ninputperpert)])


        # Copy down from sstream time & freq info - need to properly assemble new container.
        lmax = tel.lmax
        mmax = tel.mmax
        nfreq = tel.nfreq
        npol = tel.num_pol_sky

        # Set the minimum resolution required for the sky.
        ntime = 2 * mmax + 1

        freqmap = map_.index_map['freq'][:]
        row_map = map_.map[:]

        if (tel.frequencies != freqmap['centre']).all():
            raise RuntimeError('Frequencies in map do not match those in Beam Transfers.')

        # Define a new sidereal stream container to hold one or more of the perturbation structure components.
        f_stream = containers.SiderealStream(freq=freqmap, ra=ntime, input=ninputperpert,
                                    prod=desired_prod, distributed=True, comm=map_.comm)

        f_stream.redistribute('freq')
        f_stream.vis[:] = 0.0
        f_stream.weight[:] = 0.0


        # Iterate over all feed pairs and work out which is the correct index in
        # the sidereal stack.
        for pi, (fi, fj) in enumerate(desired_prod):

            unique_ind = self.telescope.feedmap[fi, fj]
            conj = self.telescope.feedconj[fi, fj]

            # unique_ind is less than zero it has masked out
            if unique_ind < 0:
                continue
            # if either fi or fj are in fact perturbations, skip ahead
            elif self.telescope.beamclass[fi] > 1:
                continue
            elif self.telescope.beamclass[fj] > 1:
                continue

            prod_stream_noconj = sstream.vis[:, unique_ind]
            prod_stream  = prod_stream_noconj.conj() if conj else prod_stream_noconj

            for pert_index in range(1,self.telescope.npert):
                # Define the feed index in sstream for the perturbed component corresponding to each feed.
                fi_pert = fi + ninputperpert * pert_index
                fj_pert = fj + ninputperpert * pert_index

                # Determine product index for fi-fj perturbed portion.
                pertfifj_ind = self.telescope.feedmap[fi_pert, fj_pert]

                # Select the visibility corresponding to this product.
                pertfifj_stream = sstream.vis[:,pertfifj_ind]

                # Determine product index for fi perturbed portion.
                pertfi_ind = self.telescope.feedmap[fi_pert, fj]
                # Select the visibility corresponding to this product.
                pertfi_stream = sstream.vis[:,pertfi_ind]

                # Determine product index for fj perturbed portion.
                pertfj_ind = self.telescope.feedmap[fi, fj_pert]
                # Select the visibility corresponding to this product.
                pertfj_stream = sstream.vis[:,pertfj_ind]

                # Conjugates if necessary.
                pertfi_conj = self.telescope.feedconj[fi_pert,fj]
                pertfj_conj = self.telescope.feedconj[fi, fj_pert]
                pertfifj_conj = self.telescope.feedconj[fi_pert,fj_pert]

                # Select fi stream/conj if necessary.
                pert_stream_fi = pertfi_stream.conj() if pertfi_conj else pertfi_stream
                # Select fj stream/conj if necessary.
                pert_stream_fj = pertfj_stream.conj() if pertfj_conj else pertfj_stream
                # Select fifj stream/conj if necessary.
                pert_stream_fifj = pertfifj_stream.conj() if pertfifj_conj else pertfifj_stream

            # Select unperturbed term.
            if self.pert_feed == 0:
                f_stream.vis[:,pi] = prod_stream
                f_stream.weight[:,pi] = 1.0
            # Select fi perturbed term
            elif self.pert_feed ==1:
                f_stream.vis[:, pi] = pert_stream_fi
                f_stream.weight[:, pi] = 1.0
            # Select fj perturbed term
            elif self.pert_feed == 2:
                f_stream.vis[:, pi] = pert_stream_fj
                f_stream.weight[:, pi] = 1.0
            # Select second order fi perturbed, fj perturbed term.
            elif self.pert_feed == 3:
                f_stream.vis[:, pi] = pert_stream_fifj
                f_stream.weight[:, pi] = 1.0


        return f_stream

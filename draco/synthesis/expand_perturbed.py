"""Tasks for expanding sidereal stream data with perturbed beams.

A typical pattern would be to turn a sidereal stream from the
:class:`SimulateSidereal` task, then generate one or more non-zero perturbation
values, apply the perturbations and expand nominally redundant products to either
first or second order in perturbation value with :class:`ExpandPerturbedProducts`.

Tasks
=====

.. autosummary::
    :toctree:

    ExpandPerturbedProducts

"""

import numpy as np

from caput import config

from ..core import containers, task, io


class ExpandPerturbedProducts(task.SingleTask):
    """Un-wrap collated products to full triangle. Apply perturbation values
    from a BeamPerturbation container to the first order perturbation
    components of the input SiderealStream. Combine unperturbed and perturbed
    beam components into one set of output products.
    """
    solution_order = config.Property(proptype=float, default=1)
    pert_val = config.Property(proptype=float, default=0.01)
    ewidth_only = config.Property(proptype=bool, default=True)

    def setup(self, bt):
        """Set up an instance of a telescope.

        Parameters
        -------------
        bt : beamtransfer.BeamTransfer or manager.ProductManager
            Sets up an observer holding the geographic location of the telscope.
        """
        self.telescope = io.get_telescope(bt)
        ninput = self.telescope.nfeed / self.telescope.npert
        if not ninput.is_integer():
            raise Exception('nfeed/npert is not an integer!')
        ninput = int(ninput)
        
        self._generate_pertubations(ninput)

    def process(self, sstream):
        """Transform a sidereal stream to having a full product matrix. Multiply
        first order beam perturbation sidereal stream components by per-feed
        perturbation values. Combine zeroth and first order beam perturbation
        components of full product matrix.

        Parameters
        ----------
        sstream : :class:`containers.SiderealStream`
            Sidereal stream to unwrap.
        perturbations: :class: `containers.BeamPerturbation`
            Beam perturbation values to be applied.

        Returns
        -------
        new_sstream : :class:`containers.SiderealStream`
            Unwrapped sidereal stream with unperturbed and perturbed components
            of products combined.
        """
        tel = self.telescope

        pertubations = self._pertubations
        
        # If only perturbing the primary beam E-plane width, the beam perturbation
        # entries are actually derivatives with respect to fwhm_e itself, so if we
        # want pert_val to be the fractional variation in this width, we need
        # to multiply the Gaussian perturbation values by fwhm_e
        if self.ewidth_only:
            pertubations *= self.telescope.fwhm_e

        # Redistribute sstream over freq
        sstream.redistribute('freq')

        # Determine ninput per perturbation based on ninput/n perturbations.
        ninput_pert = len(sstream.input)
        ninput = int(ninput_pert / tel.npert)
        
        # If I perturb also frequencies....
        # linp, sinp, einp = mpiutil.split_local(ninput)

        # Pre-define prod map and conjugation and feedmap outside of loop
        prod = np.array([(fi, fj) for fi in range(ninput) for fj in range(fi, ninput)], dtype=[('input_a', int), ('input_b', int)])

        # Make array of feed indices for new SiderealStream
        new_stream_input = np.array(np.arange(ninput), dtype=sstream.input.dtype)

        # Define new SiderealStream. Need to specify stack=None for it to
        # correspond to a non-stacked stream
        new_stream = containers.SiderealStream(input=new_stream_input, prod=prod, axes_from=sstream, stack=None, comm=self.comm)
        new_stream.redistribute('freq')
        new_stream.vis[:] = 0.0
        new_stream.weight[:] = 0.0
        
        # Dereference the global slices
        nss = new_stream.vis[:]
        nssw = new_stream.weight[:]
        ss = sstream.vis[:]
        ssw = sstream.weight[:]
        
        # Iterate over all feed pairs and work out which is the correct index in
        # the sidereal stack.
        for pi, (fi, fj) in enumerate(prod):
            unique_ind = tel.feedmap[fi, fj]
            conj = tel.feedconj[fi, fj]

            # unique_ind is less than zero it has masked out
            if unique_ind < 0:
                continue
            # if either fi or fj are perturbations, skip ahead
            elif tel.beamclass[fi] > 1:
                continue
            elif tel.beamclass[fj] > 1:
                continue

            # Select visibilities corresponding to unique_ind product.
            # Make sure to copy them, since we will be modifying ssp in-place.
            ssp = ss[:, unique_ind].local_array.copy()
            # Conjugate if necessary.
            ssp = ssp.conj() if conj else ssp

            # Loop over each perturbation, starting from 1 as 0
            # would be the unperturbed component.
            for ii in range(1, tel.npert):
                # Define the feed index in sstream for the perturbed component
                # corresponding to each feed.
                fii = fi + ninput * ii
                fjj = fj + ninput * ii

                # Define product index pfii/pfjj in sstream for the fii/fjj perturbed component.
                p_fii = tel.feedmap[fii, fj]
                p_fjj = tel.feedmap[fi, fjj]

                # Select the visibility corresponding to the fii/fjj perturbed component.
                # Multiply it by the corresponding perturbation value
                ss_fii = ss[:, p_fii] * pertubations[fi]
                ss_fjj = ss[:, p_fjj] * pertubations[fj]

                # Is this a conjugated product
                conj_fii = tel.feedconj[fii, fj]
                conj_fjj = tel.feedconj[fi, fjj]
                
                # Add perturbation components to unperturbed sstream
                ssp += ss_fii.conj() if conj_fii else ss_fii
                ssp += ss_fjj.conj() if conj_fjj else ss_fjj
        
                if self.solution_order == 1:
                    continue
                else:
                    # Define the product index for the fi & fj perturbed component.
                    p_fiifjj = tel.feedmap[fii, fjj]
                    conj_fiifjj = tel.feedconj[fii, fjj]

                    # Select the visibility corresponding to this product.
                    # Multiply it by both the fi and fj perturbation values.
                    ss_fiifjj = ss[:, f_fiifjj] * pertubations[fi] * pertubations[fj]
                    ssp += ss_fiifjj.conj() if conj_fiifjj else ss_fiifjj
                    
            # Put prod_stream into new container new_stream.
            nss[:, pi] = ssp
            nssw[:, pi] = 1.0
            
        return new_stream

    def _generate_pertubations(self, ninput):
        """Initialise beam pertubations for input channels"""
        if self.comm.rank == 0:
            # Choose random pertubations, each frequency and input?
            pertubations = np.random.normal(0, self.pert_val, size=ninput)

        else:
            pertubations = None

        # Broadcast slices to all ranks
        self._pertubations = self.comm.bcast(pertubations, root=0)

import h5py
import numpy as np
from caput import mpiarray, config, mpiutil
from ..core import task, containers#, io
from cora.util import units
#from ..util import tools
from ch_util import tools, andata, ephemeris

from caput import pipeline

# Constants
NU21 = units.nu21
C = units.c


class QuasarStack(task.SingleTask):
    """Blah.

    Attributes
    ----------
    qcat_path : str
            Full path to quasar catalog to stack on.



    """

    # Full path to quasar catalog to stack on
    qcat_path = config.Property(proptype=str)

    # Number of frequencies to keep on each side of quasar RA
    # Pick only frequencies around the quasar (50 on each side)
    freqside = config.Property(proptype=int, default=50)

    def setup(self):
        """Load quasar catalog and initialize the stack array.
        """

        # Load base quasar catalog from file (Not distributed)
        self._qcat = containers.SpectroscopicCatalog.from_file(
                                                    self.qcat_path)
        self.nqso = len(self._qcat['position'])

        # Size of quasar stack array
        self.nstack = 2 * self.freqside + 1

        # Quasar stack array.
        self.quasar_stack = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype='complex64'), axis=0)
        # Keep track of number of quasars added to each frequency bin 
        # in the quasar stack array
        self.quasar_stack_wheight = mpiarray.MPIArray.wrap(
                np.zeros(self.nstack, dtype='complex64'), axis=0)


    def process(self, data):
#    def process(self):
        """Smooth the weights with a median filter.

        Parameters
        ----------
        data : :class:`andata.CorrData` or :class:`containers.TimeStream` object
            Data containing the weights to be smoothed

        Returns
        -------
        data : Same object as data
            Data object containing the same data as the input, but with the
            weights substituted by the smoothed ones.
        """
        # Ensure data is distributed in vis-stack axis
        # TODO: This is not working. I checked and it actually calls
        # memh5.BasicCont.redistribute and, inside that function,
        # the effects take place on the datasets!!! Very crazy!
        data.redistribute(1)
        print 'Hu', type(data), data['vis'], data['vis'].local_shape, data['vis'].global_shape, data['vis'].local_offset, data['vis'].distributed_axis

        nfreq = len(data.index_map['freq'])
        nvis = len(data.index_map['stack'])
#        nvis = 20  # TODO: delete

        # Find where each Quasar falls in the RA axis
        # Assume equal spacing in RA axis.
        ra_width = np.mean(data.index_map['ra'][1:]
                         -data.index_map['ra'][:-1])
        # Normaly I would set the RAs as the center of the bins. 
        # But these start at 0 and end at 360 - ra_width. 
        # So they are the left edge of the bin here...
        ra_bins = np.insert(arr=data.index_map['ra'][:] + ra_width,
                                                      values=0.,obj=0)
        # Bin Quasars in RA axis. Need -1 due to the way np.digitize works.
        qso_ra_indices = np.digitize(self._qcat['position']['ra'],ra_bins) - 1
        if not ((qso_ra_indices>=0) 
              & (qso_ra_indices<len(data.index_map['ra']))).all():
            # TODO: raise an error?
            pass

        # Compute all baseline vectors.
        # Baseline vectors in meters. Mpiarray is created distributed in the
        # 0th axis by default. Argument is global shape.
        bvec_m = mpiarray.MPIArray((nvis,2),dtype=np.float64)  
        nvis_local = bvec_m.shape[0]  # Local number of visibilities
        xx_indices, yy_indices = [], []  # Indices (local) of co-pol products

        ## TODO: delete. To show slicing works.
        #print mpiutil.rank, bvec_m.shape, bvec_m.global_shape, type(bvec_m)
        #aa = np.ones(7,dtype=float)
        #bb = aa[:,np.newaxis,np.newaxis] * bvec_m[np.newaxis,:,:]
        #print bb.shape#, bb.global_shape, type(bb)
        #print bb[:,[1,2,3],0].shape, type(bb[:,[1,2,3],0])
        

        for lvi, gvi in bvec_m.enumerate(axis=0):

            gpi = data.index_map['stack'][gvi][0]  # Global product index
            conj = data.index_map['stack'][gvi][1]  # Product conjugation
            # Inputs that go into this product
            ipt0 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][gpi][0]]
            ipt1 = data.index_map['input']['chan_id'][
                                data.index_map['prod'][gpi][1]]

            # Get position and polarization of each input
            pos0, pol0 = self._pos_pol(ipt0)
            pos1, pol1 = self._pos_pol(ipt1)

            if ((pol0==0) and (pol1==0)):
                xx_indices.append(lvi)
            if ((pol0==1) and (pol1==1)):
                yy_indices.append(lvi)

            # Beseline vector in meters
            # TODO: I am actually computing the baseline vector
            # for all products here, even cross-pol. If it is slow
            # I should change this.
            bvec_m[lvi] = self._baseline(pos0,pos1,conj=conj)

        #for qq in range(self.nqso):
        for qq in [8,10]:  # This are quasars in this reduced frequency range.
            dec = self._qcat['position']['dec'][qq]
            qso_z = self._qcat['redshift']['z'][qq]
            ra_index = qso_ra_indices[qq]

            # Frequency of Quasar
            qso_f = NU21/(qso_z + 1.)  # MHz.
            # Index of closest frequency
            qso_findex = np.argmin(abs(
                    data.index_map['freq']['centre'] - qso_f))
        
            # Pick only frequencies around the quasar (50 on each side)
            # Indices to be processed in full frequency axis
            lowindex = np.amax((0, qso_findex - self.freqside))
            upindex = np.amin((nfreq, qso_findex + self.freqside + 1))
            f_slice = np.s_[lowindex:upindex]
            # Corresponding indices in quasar stack array
            lowindex = lowindex - qso_findex + self.freqside
            upindex = upindex - qso_findex + self.freqside
            qs_slice = np.s_[lowindex:upindex]
        
            # Pick a polarization.
            # TODO: How do I add polarizations later? In quadrature?
            pol_indices = xx_indices

            nu = data.index_map['freq']['centre'][f_slice]
            # Baseline vectors in wavelengths. Shape (nstack, nvis_local, 2)
            bvec = bvec_m[np.newaxis,:,:] * nu[:,np.newaxis,np.newaxis] * 1E6 / C
            
            # Complex corrections to be multiplied by the visibilities to make them real.
            correc = tools.fringestop_phase(ha=0., lat=np.deg2rad(ephemeris.CHIMELATITUDE),
                                                dec=np.deg2rad(dec),
                                                u=bvec[:,pol_indices,0],
                                                v=bvec[:,pol_indices,1])

            # This is done in a slightly weird order: adding visibility subsets
            # for different quasars in each rank first and then co-adding accross visibilities
            # and finally taking the real part: Real( Sum_j Sum_i [ qso_i_vissubset_j ] )

            # Fringestop and sum.
            # TODO: this corresponds to Uniform weighting, not Natural.
            # Need to figure out the multiplicity of each visibility stack.
            print 'Hi', correc.shape, bvec.shape, data['vis'].shape

            self.quasar_stack[qs_slice] += np.sum(
                data['vis'][f_slice, pol_indices, ra_index] * correc, axis=1)
            # Increment wheight for the appropriate quasar stack indices.
            quasar_stack_wheight[qs_slice] += 1.

        # TODO take real part after summing over stack visibilities accross ranks.
        # Have tp call Gather here.




        raise pipeline.PipelineStopIteration() 


    # TODO: the next two functions are temporary hacks. The information
    # should either be obtained from a TransitTelescope object in the 
    # pipeline or these functions, if still useful, should be moved to
    # some appropriate place like ch_util.tools.

    # TODO: This is a temporary hack.
    def _pos_pol(self, chan_id, nfeeds_percyl = 64, ncylinders = 2):
        """ This is a temporary hack. The pipeline will have a
        drift.core.telescope.TransitTelescope object with all of this 
        information in it. I have to look up how to use it.

        Parameters
        ----------
        nfeeds_percyl : int 
            Number of feeds per cylinder
        ncylinders : int
            Number of cylinders
        """

        cylpol = chan_id//nfeeds_percyl
        cyl = cylpol//ncylinders
        pol = cylpol%ncylinders
        pos = chan_id%nfeeds_percyl

        return (cyl, pos), pol

    # TODO: This is a temporary hack.
    def _baseline(self, pos0, pos1, nu = None, conj=1):
        """ Computes the vector sepparation between two positions 
        given in cylinder index and feed position index.
        The vector goes from pos1 to pos0. This gives the right visibility
        phase for CHIME: phi_0_1 = 2pi baseline_0_1 * \hat{n}.
        
        +X is due East and +Y is due North.
        
        Parameters
        ----------
        pos0, pos1 : tuple or array-like (cyl, pos)
            cylinder number (W to E) and position in f.l. N to S.
        nu : float
            Frequency in MHz
        conj : int or bool
            If 1 (True) Multiply the final vector by -1 
            to account for conjugation of the product.
            
        Returns
        -------
        Baseline vector in meters (if nu is None) 
        or wavelengths (if nu is not None).
       """
        cylinder_sepparation = 22.  # In meters
        feed_sepparation = 0.3048  # In meters

        # -1 is to convert NS feed number (which runs South) 
        # into Y coordinates (which point North)
        baseline_vec = np.array([cylinder_sepparation*float(pos0[0]-pos1[0]),
                                 feed_sepparation*float(pos0[1]-pos1[1])*(-1.)])
    
        if nu is not None:
            nu = float(nu)*1E6
            baseline_vec *= nu / C
    
        if conj:
            return (-1.)*baseline_vec
        else:
            return baseline_vec


import json

import numpy as np
from scipy import interpolate
from scipy.constants import c as speed_of_light
import h5py

from caput import config, pipeline, memh5
from caput import mpiarray, mpiutil

from ch_util import tools
from ch_util import ephemeris
from ch_util import ni_utils
from ch_util import cal_utils
from ch_util import fluxcat
from ch_util import finder
from ch_util import rfi

from draco.core import task
from draco.util import _fast_tools
from draco.core import containers

class LoadMyGains(task.SingleTask):
    """LOAD GAINS AND PASS IT INTO GAIN CONTAINER USING DRACO. OUTPUT GAINS TO BE USED BY DRACO/APPLYGAINS
        T.CHEN 14/12/2020
    """
    _path = config.Property(proptype=str)

    def process(self, stream):
        
        #Load gains from file
        gf = h5py.File(self._path, 'r')
        g = gf['gain'][...]
        gf.close()

        #Figure out the frequencies and time axes from your sidereal stream?
        #freq, time = stream.freq, stream.time
        freq, input = stream.freq, stream.input

        #Create container
        gain = containers.StaticGainData(
                freq=freq, input = input, distributed=True
                )

        #Redistribute
        gain.redistribute("freq")
        lo = gain.gain.local_offset[0]
        ls = gain.gain.local_shape[0]

        #Assign the frequency range appropriate to each rank
        gain.gain[:] = g[lo:lo+ls]

        #Add something here if you have a model for your gains
        #gain.weight[:] =

        return gain

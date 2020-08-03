cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as cnp

cdef extern from "truncate.hpp":
    inline float bit_truncate_float(float val, float err) nogil

def bit_truncate(float val, float err):
    return bit_truncate_float(val, err)

@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_weights(float[:] val, float[:] wgt, float fallback):
    cdef int n = val.shape[0]
    if val.ndim != 1:
        raise ValueError("Input array must be 1-d.")
    if wgt.shape[0] != n:
        raise ValueError(
            "Weight and value arrays must have same "
            "shape ({:d} != {:d})".format(wgt.shape, n)
        )

    cdef int i = 0

    for i in prange(n, nogil=True):
        if wgt[i] != 0:
            val[i] = bit_truncate_float(val[i], 1. / wgt[i]**0.5)
        else:
            val[i] = bit_truncate_float(val[i], fallback * val[i])

    return np.asarray(val)

@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_fixed(float[:] val, float prec):
    cdef int n = val.shape[0]
    cdef int i = 0

    for i in range(n):
        val[i] = bit_truncate_float(val[i], prec * val[i])

    return np.asarray(val)
"""Routines for Gaussian Process Regression."""

import numpy as np
import scipy.linalg as la


def simple_regression(data, weight, K, Kstar):
    """Interpolate an input array assuming constant noise.

    Iterate the first axis.
    """
    # Get the square of each kernel for use each loop
    K2 = K**2
    Kstar2 = Kstar**2

    # Make the output arrays
    nsamp = Kstar.shape[0]
    out = np.zeros((data.shape[0], nsamp, *data.shape[2:]), dtype=data.dtype)
    wout = np.zeros((weight.shape[0], nsamp, *weight.shape[2:]), dtype=weight.dtype)
    eye = np.ones_like(weight[0, :, 0])

    # Iterate the first axis and interpolate the second
    for ii in range(data.shape[0]):
        wi = np.ascontiguousarray(weight[ii])
        xi = np.ascontiguousarray(data[ii])
        mi = ~np.all(wi == 0, axis=-1)

        if mi.sum() < 10:
            continue

        y = xi.view(np.float32)
        # Get the cholesky of the kernel to invert
        kl = la.cho_factor(K[mi][:, mi], check_finite=False, lower=False)
        kl2 = la.cho_factor(K2[mi][:, mi], check_finite=False, lower=False)

        # Solve the data
        alpha = la.cho_solve(kl, y[mi], check_finite=False)
        yp = (Kstar[:, mi] @ alpha).reshape(out.shape[1:] + (2,))
        out[ii] = yp[..., 0] + 1j * yp[..., 1]

        # Interpolate weights and solve for a mask
        beta = la.cho_solve(kl2, wi[mi], check_finite=False)
        wout[ii] = Kstar2[:, mi] @ beta

        # Get a mask and apply it to the data and weights
        err = la.cho_solve(kl2, eye[mi], check_finite=False)
        err = (Kstar2[:, mi] @ err) > 0.97

        out[ii] *= err[:, np.newaxis]
        wout[ii] *= err[:, np.newaxis]

    return out, wout

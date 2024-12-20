"""Tasks to do data interpolation/inpainting."""

import numpy as np
from caput import config

from ..core import task
from ..util import dpss


class InpaintDPSS(task.SingleTask):
    """Fill data gaps using DPSS inpainting.

    Discrete prolate spheroidal sequence (DPSS) inpainting involves
    projecting a partially-masked data series onto a basis which
    maximally concentrates spectral power within a defined window.
    This basis, called the nth order discrete prolate spheroidal
    sequence or Slepian sequence consists of the large eigenvectors
    of a covariance matrix defined as a sum of `sinc` functions,
    which represent top-hats in the spectral inverse of the data.

    Attributes
    ----------
    axis : str
        Name of the axis over which to inpaint. Only one-dimensional
        inpainting is currently supported. Must be either "freq" or
        "ra". Default is `freq`.
    centres : list
        List of top-hat window centres. If all windows are centred
        about zero, the covariance matrix will be real, which provides
        significant performance improvements.
    halfwidths : list
        List of window half-widths. Must be the same length as `centres`.
    snr_cov : float
        Wiener filter inverse signal covariance. Default is 1.0e-3.
    copy : bool
        If true, copy the container instead of inpainting in-place.
    """

    axis = config.enum(["freq", "ra"], default="freq")
    centres = config.Property(proptype=list)
    halfwidths = config.Property(proptype=list)
    snr_cov = config.Property(proptype=float, default=1.0e-3)
    copy = config.Property(proptype=bool, default=True)

    def process(self, data):
        """Inpaint visibility data.

        Parameters
        ----------
        data : containers.VisContainer
            Container with a visibility dataset

        Returns
        -------
        data : containers.VisContainer
            Input container with masked values filled
        """
        try:
            # Get the axis samples
            samples = getattr(data, self.axis)
        except AttributeError as exc:
            raise ValueError(f"Could not get axis `{self.axis}`.") from exc

        # Redistribute over an independent axis
        data.redistribute(["stack", "el"])

        # Make the output container
        out = data.copy() if self.copy else data
        # This should already have the same distributed
        # axis as `data`
        out.redistribute(["stack", "el"])

        vinp = self.inpaint(data.vis, data.weight, samples)
        # TODO: this should also return a weight estimate
        out.vis[:].local_array[:] = vinp

        return out

    def inpaint(self, vis, weight, samples):
        """Inpaint visibilities using a wiener filter."""
        # Construct the covariance matrix and get dpss modes
        cov = dpss.make_covariance(samples, self.halfwidths, self.centres)
        A = dpss.get_sequence(cov)

        # Move the interpolation axis to the front
        # and flatten the other axes
        vaxind = list(vis.attrs["axis"]).index(self.axis)
        waxind = list(weight.attrs["axis"]).index(self.axis)

        vobs = _move_front(vis[:].local_array, vaxind, vis.local_shape)
        wobs = _move_front(weight[:].local_array, waxind, weight.local_shape)

        # Project visibilities into the dpss basis
        vproj = dpss.project(vobs, wobs, A)
        # Solve for basis coefficients
        b = dpss.solve(vproj, wobs, A)
        # Get the inpainted visibilities
        vinp = dpss.inpaint(A, b, vobs, wobs > 0)

        # Reshape and move the interpolation axis back
        # TODO: we also need to return some weights
        return _inv_move_front(vinp, vaxind, vis.local_shape)


# TODO: these are copied from `draco.analysis.delay`. They
# should be moved somewhere more general
def _move_front(arr: np.ndarray, axis: int, shape: tuple) -> np.ndarray:
    # Move the specified axis to the front and flatten to give a 2D array
    new_arr = np.moveaxis(arr, axis, 0)
    return new_arr.reshape(shape[axis], -1)


def _inv_move_front(arr: np.ndarray, axis: int, shape: tuple) -> np.ndarray:
    # Move the first axis back to it's original position and return the original shape,
    # i.e. reverse the above operation
    rshape = (shape[axis],) + shape[:axis] + shape[(axis + 1) :]
    new_arr = arr.reshape(rshape)
    new_arr = np.moveaxis(new_arr, 0, axis)
    return new_arr.reshape(shape)

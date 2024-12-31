"""Tasks to do data interpolation/inpainting."""

import numpy as np
from caput import config
from cora.util import units

from ..core import io, task
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
        # Set the local selection over the distributed axis
        self._local_sel = self._get_sel(data)

        vinp, _ = self.inpaint(data.vis, data.weight, samples)
        # TODO: need a weight estimate

        # Make the output container
        out = data.copy() if self.copy else data
        out.redistribute(["stack", "el"])

        out.vis[:].local_array[:] = vinp

        return out

    def inpaint(self, vis, weight, samples):
        """Inpaint visibilities using a wiener filter.

        Use a single sequence for the entire dataset.
        """
        # Move the iteration and interpolation axes
        # to the front and flatten the other axes
        vax = list(vis.attrs["axis"])
        wax = list(vis.attrs["axis"])

        vaxind = []
        waxind = []

        for axis in ("stack", "el", self.axis):
            if axis in vax:
                vaxind.append(vax.index(axis))
                waxind.append(wax.index(axis))

        vobs = _move_front(vis[:].local_array, vaxind, vis.local_shape)
        wobs = _move_front(weight[:].local_array, waxind, weight.local_shape)

        # Pre-allocate the full output array
        vinp = np.zeros_like(vobs)
        winp = np.zeros_like(wobs)

        # Construct the covariance matrix and get dpss modes
        modes, amap = self._get_basis(samples)

        # Iterate over the variable axis
        for ii in range(vobs.shape[0]):
            # Get the correct basis for each slice
            A = modes[amap[ii]]
            # Write directly into the preallocated output array
            self.inpaint_single(vobs[ii], wobs[ii], A, out=vinp[ii])

        # Reshape and move the interpolation axis back
        vinp = _inv_move_front(vinp, vaxind, vis.local_shape)
        winp = _inv_move_front(winp, waxind, weight.local_shape)

        return vinp, winp

    @staticmethod
    def inpaint_single(vobs, wobs, A, out):
        """Inpaint a data slice."""
        # Project visibilities into the dpss basis
        vproj = dpss.project(vobs, wobs, A)
        # Solve for basis coefficients
        b = dpss.solve(vproj, wobs, A)
        # Get the inpainted visibilities
        return dpss.inpaint(A, b, vobs, wobs > 0, out=out)

    def _get_sel(self, data):
        """Extract selection along local axis."""
        return data.vis[:].local_bounds

    def _get_basis(self, samples):
        """Make the DPSS basis.

        Returns a list of bases and a map.
        """
        # Construct the covariance matrix and get dpss modes
        cov = dpss.make_covariance(samples, self.halfwidths, self.centres)
        modes = dpss.get_sequence(cov)
        # All iterations map to the same basis
        amap = [0] * (self._local_sel.stop - self._local_sel.start)

        return [modes], amap


class InpaintDPSSDelay(InpaintDPSS):
    """Inpaint with baseline-dependent delay cut."""

    axis = config.enum(["freq"], default="freq")

    def setup(self, telescope):
        """Load an observer object."""
        self.telescope = io.get_telescope(telescope)

    def _get_sel(self, data):
        """Get the set of local baselines."""
        prod = data.prodstack

        return self.telescope.feedmap[(prod["input_a"], prod["input_b"])]

    def _get_basis(self, samples):
        """Make the DPSS basis."""
        # Note that this produces covariances for
        # _all_ baselines. An additional slice has to
        # be checked at each iteration or something
        blen = abs(self.telescope.baselines[:, 1])
        # Use only the local baselines
        blen = blen[self._local_sel]

        # Get the delay cut for each baseline
        delay_cut = np.maximum(blen / units.c * 1.0e6, self.halfwidths[0])
        delay_cut = np.round(delay_cut, decimals=3)

        # Compute covariances for each unique baseline and
        # map to each individual baseline
        delay_cut, amap = np.unique(delay_cut, return_inverse=True)

        modes = []

        for ii, cut in enumerate(delay_cut):
            self.log.debug(f"Making unique covariance {ii}/{len(delay_cut)}.")
            cov = dpss.make_covariance(samples, cut, 0.0)
            modes.append(dpss.get_sequence(cov))

        return modes, amap


# TODO: These should be moved somewhere more general
def _move_front(arr: np.ndarray, axis: int | list, shape: tuple) -> np.ndarray:
    """Move specified axes to the front and flatten remaining axes."""
    if np.isscalar(axis):
        axis = [axis]

    new_shape = [shape[i] for i in axis]
    # Move the N specified axes to the first N positions
    inds = list(range(len(axis)))
    # Move the specified axes to the front and flatten
    # the remaining axes
    arr = np.moveaxis(arr, axis, inds)

    return arr.reshape(*new_shape, -1)


def _inv_move_front(arr: np.ndarray, axis: int | list, shape: tuple) -> np.ndarray:
    """Move axes back to their original position and expand."""
    new_shape = [shape[i] for i in axis]
    new_shape += [sh for sh in shape if sh not in new_shape]
    inds = list(range(len(axis)))

    # Undo the flattening process
    arr = arr.reshape(new_shape)
    # Move axes back to their original positions
    arr = np.moveaxis(arr, inds, axis)

    return arr.reshape(shape)

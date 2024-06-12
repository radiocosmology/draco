"""Helper functions for the delay PS estimation via ML/MAP optimisation."""

from typing import Protocol

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from ..util import tools


class OptFunc(Protocol):
    """A protocol for a function and its gradients to optimise.

    The function should be defined such that it can be minimised.
    """

    def value(self, x: np.ndarray) -> float:
        """Compute the value of the function.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        The value of the function being optimized.
        """
        raise NotImplementedError()

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the function with respect to the optimization parameters.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        The gradient of the function being optimized.
        """
        raise NotImplementedError()

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute the hessian (matrix of second derivatives) of the function with respect to the optimization parameters.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        The hessian of the function being optimized.
        """
        raise NotImplementedError()


class LogLikePS(OptFunc):
    """Compute the likelihood, gradient, and hessian for delay PS estimation.

    This class efficiently computes the *negative* likelihood, as well as its gradient
    and hessian. It will precompute and cache relevant quantities such that all of these
    can be calculated per iteration without recomputing. It is designed to be used with
    `scipy.optimize.minimize`.

    The parameters used for this are the log of the delay power spectrum samples.

    Parameters
    ----------
    X
        The covariance matrix of the data.
    MF
        The masked Fourier matrix which maps from delay space to frequency space and
        applies zero to any masked channels.
    N
        The noise covariance matrix.
    nsamp
        The number of samples used to calculate the covariance.
    fsel
        An optional array selection to apply to limit the frequencies used in the
        estimation. If not supplied, any frequencies entirely masked within `MF` are
        skipped.
    exact_hessian
        If set, use the exact Hessian for the calculation. Otherwise use the Fisher
        matrix in the same way as the original NRML methods.
    """

    def __init__(
        self,
        X: np.ndarray,
        MF: np.ndarray,
        N: np.ndarray,
        nsamp: int,
        fsel: np.ndarray | slice | list | None = None,
        exact_hessian: bool = True,
    ) -> None:
        if fsel is None:
            fsel = (MF != 0).any(axis=1)

        self.X = X[fsel][:, fsel]
        self.MF = MF[fsel]
        self.N = N[fsel][:, fsel]

        self.nsamp = nsamp
        self.exact_hessian = exact_hessian

    # Store the location we are calculating for.
    _s_a: np.ndarray | None = None

    def _precompute(self, x: np.ndarray) -> bool:
        """Pre-compute useful matrices for the given value.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        True if a pre-computation was done, otherwise False.
        """
        if np.all(x == self._s_a):
            return False

        self._s_a = x

        S = np.exp(x)
        dS = S

        self._C = (self.MF * S[np.newaxis, :]) @ self.MF.T.conj() + self.N
        self._XC = self.X - self._C

        self._Ch = la.cholesky(self._C, lower=False)

        self._U = dS[np.newaxis, :] ** 0.5 * self.MF
        self._Ut = la.cho_solve((self._Ch, False), self._U)
        self._XC_Ut = self._XC @ self._Ut

        self._W = self._U
        self._Wt = self._Ut
        self._XC_Wt = self._XC_Ut

        return True

    def value(self, x: np.ndarray) -> float:
        """Calculate the negative log-likelihood.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Value of negative log-likelihood for the given set of params.
        """
        self._precompute(x)

        CiX = la.cho_solve((self._Ch, False), self.X)

        lndet = 2 * np.log(np.diagonal(self._Ch)).sum().real

        ll = lndet + np.diagonal(CiX).sum().real

        return self.nsamp * ll

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the negative log-likelihood.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Gradient of negative log-likelihood for the given set of params.
        """
        self._precompute(x)

        g = -(self._Ut.conj() * self._XC_Ut).sum(axis=0).real

        return self.nsamp * g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Hessian of the negative log-likelihood.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Hessian of negative log-likelihood for the given set of params.
        """
        self._precompute(x)

        Ua_Utb = self._U.T.conj() @ self._Ut
        Fab = Ua_Utb * Ua_Utb.T.conj()
        H = Fab.real

        if self.exact_hessian:
            Uta_dX_Utb = self._Ut.T.conj() @ self._XC_Ut
            H += (2 * Uta_dX_Utb * Ua_Utb.T).real
            t = -(self._Wt.conj() * self._XC_Wt).sum(axis=0).real
            H += np.diag(t.real)

        return self.nsamp * H


class SmoothnessRegulariser(OptFunc):
    """A smoothness prior on the values at given locations.

    This calculates the average in a window of `width` points, and then applies a
    Gaussian with precision `alpha` and this average as the mean for each point. For a
    `width` of 3 this is effectively a constraint on the second derivative.

    Parameters
    ----------
    N
        The number of points in the function we are inferring. The sample locations are
        implicitly assumed to be at `np.arange(N)`.
    alpha
        The smoothness precision.
    width
        The width over which the smoothness is calculated in samples.
    periodic
        Assume that the function is periodic and wrap around.
    """

    def __init__(
        self, N: int, alpha: float, *, width: int = 5, periodic: bool = True
    ) -> None:

        # Calculate the matrix for the moving average
        W = np.zeros((N, N))
        for i in range(N):

            ll, ul = i - (width - 1) // 2, i + (width + 1) // 2
            if not periodic:
                ll, ul = max(0, ll), min(ul, N)
            v = np.arange(ll, ul)

            W[i][v] = 1.0 / len(v)

        IW = np.identity(N) - W
        self.Ci = alpha * (IW.T @ IW)

    def value(self, x: np.ndarray) -> float:
        """Calculate the value of the smoothness regularizer.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Value of the smoothness regularizer for the given set of params.
        """
        return 0.5 * float(x @ self.Ci @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the smoothness regularizer.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Gradient of the smoothness regularizer for the given set of params.
        """
        return self.Ci @ x

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hessian of the smoothness regularizer.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Hessian of the smoothness regularizer for the given set of params.
        """
        return self.Ci


class GaussianProcessPrior(OptFunc):
    """A Gaussian process prior on the inputs.

    The kernel has a standard deviation of `width`, and the variance is `alpha**2`.

    Parameters
    ----------
    N
        The number of points in the function we are inferring. The sample locations are
        implicitly assumed to be at `np.arange(N)`.
    alpha
        The scale of the distribution.
    width
        The width of the kernel used in the covariance.
    kernel
        The name of the kernel to use. Either 'gaussian' or 'rational'.
    a
        Parameter for the rational quadratic kernel.
    periodic
        Assume that the function is periodic and wrap around.
    reg
        Add a small diagonal entry of size `alpha**2 * reg` for numerical stability.
    """

    def __init__(
        self,
        N: int,
        alpha: float,
        width: int = 5,
        *,
        kernel: str = "gaussian",
        a: float = 1.0,
        periodic: bool = True,
        reg: float = 1e-6,
    ) -> None:

        ii = np.arange(N)
        d = ii[:, np.newaxis] - ii[np.newaxis, :]

        if periodic:
            d = ((d + N / 2) % N) - N / 2

        d2 = (d / width) ** 2

        if kernel == "gaussian":
            C = np.exp(-0.5 * d2)
        elif kernel == "rational":
            C = (1 + d2 / (2 * a)) ** -a
        else:
            raise ValueError(f"Unknown kernel type '{kernel}'")

        self.Ci: np.ndarray = la.inv(C + np.identity(N) * reg) / alpha**2

    def value(self, x: np.ndarray) -> float:
        """Calculate the value of the gaussian process prior.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Value of the gaussian process prior for the given set of params.
        """
        return 0.5 * float(x @ self.Ci @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the gaussian process prior.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Gradient of the gaussian process prior for the given set of params.
        """
        return self.Ci @ x

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hessian of the gaussian process prior.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Hessian of the gaussian process prior for the given set of params.
        """
        return self.Ci


class AddFunctions(OptFunc):
    """Optimise the sum of several functions.

    The values, gradients and hessians of each function are added together.

    Parameters
    ----------
    functions
        A list of functions to optimise. The individual functions must all
        take the same set of parameters.
    """

    def __init__(self, functions: list[OptFunc]) -> None:
        if len(functions) <= 0:
            raise ValueError("At least one function must be supplied.")
        self.functions = functions

    def value(self, x: np.ndarray) -> float:
        """Calculate the sum of the individual function values.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Sum of individual function values for the given set of params.
        """
        return sum(f.value(x) for f in self.functions)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate the sum of the individual function gradients.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Sum of individual function gradients for the given set of params.
        """
        g = self.functions[0].gradient(x)
        for f in self.functions[1:]:
            g += f.gradient(x)
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the sum of the individual function hessians.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Sum of individual function hessians for the given set of params.
        """
        h = self.functions[0].hessian(x)
        for f in self.functions[1:]:
            h += f.hessian(x)
        return h


def delay_power_spectrum_maxpost(
    data,
    N,
    Ni,
    initial_S: np.ndarray | None = None,
    window: str = "nuttall",
    fsel: np.ndarray | None = None,
    maxiter: int = 30,
    tol: float = 1e-3,
):
    """Estimate the delay power spectrum with a maximum-likelihood estimator.

    This routine uses `scipy.optimize.minimize` to find the maximum likelihood power
    spectrum.

    Parameters
    ----------
    data : np.ndarray[:, freq]
        Data to estimate the delay spectrum of.
    N : int
        The length of the output delay spectrum. There are assumed to be `N/2 + 1`
        total frequency channels if assuming a real delay spectrum, or `N` channels
        for a complex delay spectrum.
    Ni : np.ndarray[freq]
        Inverse noise variance.
    initial_S : np.ndarray[delay]
        The initial delay power spectrum guess.
    window : one of {'nuttall', 'blackman_nuttall', 'blackman_harris', None}, optional
        Apply an apodisation function. Default: 'nuttall'.
    fsel : np.ndarray[freq], optional
        Indices of channels that we have data at. By default assume all channels.
    maxiter : int, optional
        Maximum number of iterations to run of the solver.
    tol : float, optional
        The convergence tolerance for the optimization that is passed to scipy.optimize.minimize.

    Returns
    -------
    spec : list
        List of spectrum samples.
    success : bool
        True if the solver successfully converged, False otherwise.
    """
    from .delay import fourier_matrix

    nsamp, Nf = data.shape

    if fsel is None:
        fsel = np.arange(Nf)
    elif len(fsel) != Nf:
        raise ValueError(
            "Length of frequency selection must match frequencies passed. "
            f"{len(fsel)} != {data.shape[-1]}"
        )

    # Construct the Fourier matrix
    F = fourier_matrix(N, fsel)

    # Compute the covariance matrix of the data
    # Window the frequency data if requested
    if window is not None:
        # Construct the window function
        x = fsel * 1.0 / N
        w = tools.window_generalised(x, window=window)

        # Apply to the projection matrix and the data
        F *= w[:, np.newaxis]
        data = data * w[np.newaxis, :]

    X = data.T @ data.conj()
    X /= nsamp

    # Construct the noise matrix from the diagonal of its inverse
    Nm = np.diag(tools.invert_no_zero(Ni))

    # Mask out any completely missing frequencies
    F[Ni == 0] = 0.0

    # Use the pseudo-inverse to give a starting point for the optimiser
    if initial_S is None:
        lsi = np.log((data @ la.pinv(F.T, rtol=1e-3)).var(axis=0))
    else:
        lsi = np.log(initial_S)

    optfunc = AddFunctions(
        [
            LogLikePS(X, F, Nm, nsamp, exact_hessian=True),
            GaussianProcessPrior(
                N, alpha=5, width=3.0, kernel="gaussian", a=5.0, reg=1e-8
            ),
        ]
    )

    samples = []

    # This callback is for getting the intermediate samples such that we can access
    # convergence of the solution
    def _get_intermediate(xk):
        samples.append(np.exp(xk))

    try:
        res = minimize(
            optfunc.value,
            x0=lsi,
            jac=optfunc.gradient,
            hess=optfunc.hessian,
            method="Newton-CG",
            options={"maxiter": maxiter, "xtol": tol},
            callback=_get_intermediate,
        )
        success = res.success

    # LinAlgError gets thrown for certain baselines in _precompute during a Cholesky decomposition
    # of the covariance matrix (used in likelihood computation) when the covariance matrix isn't
    # positive definite. This appears to happen after one of the minimization parameters blows up,
    # likely from a numerical instability somewhere in the gradient/hessian computation. This has
    # only been observed to affect a small number of (almost entirely masked) baselines that have
    # very low weights.
    # ValueError also occasionally gets thrown from (almost certainly) the same numerical
    # instability: one of the minimization parameters blows up and causes an overflow when taking
    # np.exp() of the minimization parameter to get the delay spectrum. The ValueError gets thrown
    # again in the Cholesky decomposition when numpy runs a check_finite on the covariance (which
    # at that point contains infs/nans).
    except (la.LinAlgError, ValueError):
        success = False

    # In rare cases a LinAlgError can be thrown before the _get_intermediate callback is called in `minimize`.
    # In this scenario, an empty `samples` is returned and this causes errors in the calling function.
    # Add the initial guess to the samples list in this case and ensure success is set to False.
    if len(samples) == 0:
        samples.append(np.exp(lsi))
        success = False

    # NOTE: the final sample in samples is already the final result
    return samples, success

"""Provides a model of the integrand, with the capability to perform Bayesian quadrature."""

from typing import Tuple, Union, List

import numpy as np
from GPy.kern import Kern, RBF
from multimethod import multimethod
from numpy import ndarray, newaxis

from ._decorators import flexible_array_dimensions
from .gps import WarpedGP, WsabiLGP
from ._maths_helpers import jacobian_of_f_squared_times_g, hessian_of_f_squared_times_g
from .priors import Gaussian, Prior


class IntegrandModel:
    """Represents the product of a warped Gaussian Process and a prior.

    Typically, this product is the function that we're interested in integrating."""

    def __init__(self, warped_gp: WarpedGP, prior: Prior):
        self.warped_gp = warped_gp
        self.prior = prior
        self.dimensions = warped_gp.dimensions

    @flexible_array_dimensions
    def posterior_mean_and_variance(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """Get the posterior mean and variance of the product of warped GP and prior at a point, or a set of points.

        Parameters
        ----------
        x
            The point(s) at which to evaluate the posterior mean and variance. A 2D array of shape
            (num_points, num_dimensions), or a 1D array of shape (num_dimensions).

        Returns
        -------
        mean : ndarray
            A 1D array of shape (num_points) if the input was 2D, or a 0D array if the input was 1D. The :math:`i`-th
            element is the posterior mean at the :math:`i`-th point of `x`.
        variance : ndarray
            A 1D array of shape (num_points) if the input was 2D, or a 0D array if the input was 1D. The :math:`i`-th
            element is the posterior variance at the :math:`i`-th point of `x`.
        """
        warped_gp_mean, warped_gp_variance = self.warped_gp.posterior_mean_and_variance(x)
        prior = self.prior(x)

        mean = warped_gp_mean * prior
        variance = warped_gp_variance * prior ** 2

        return mean, variance

    @flexible_array_dimensions
    def posterior_variance_jacobian(self, x: ndarray) -> ndarray:
        """Get the Jacobian of the posterior variance of the product of warped GP and prior at a point or set of points.

        Parameters
        ----------
        x
            The point(s) at which to evaluate the Jacobian. A 2D array of shape (num_points, num_dimensions), or a 1D
            array of shape (num_dimensions).

        Returns
        -------
        jacobian : ndarray
            A 2D array of shape (num_points, num_dimensions) if the input was 2D, or a 1D array of shape
            (num_dimensions) if the input was 1D. The :math:`(i, j)`-th element is the :math:`j`-th component of the
            Jacobian of the posterior variance at the :math:`i`-th point of `x`.

        Notes
        -----
        Writing :math:`\\pi(x)` for the prior, and :math:`V(x)` for the posterior variance, the posterior variance of
        the product is :math:`\\pi(x)^2 V(x)`.
        """
        _, gp_variance = self.warped_gp.posterior_mean_and_variance(x)
        gp_variance_jacobian = self.warped_gp.posterior_variance_jacobian(x)

        prior = self.prior(x)
        prior_jacobian, _ = self.prior.gradient(x)

        return jacobian_of_f_squared_times_g(
            f=prior, f_jacobian=prior_jacobian,
            g=gp_variance, g_jacobian=gp_variance_jacobian)

    @flexible_array_dimensions
    def posterior_variance_hessian(self, x: ndarray) -> ndarray:
        """Get the Hessian of the posterior variance of the product of warped GP and prior at a point, or set of points.

        Parameters
        ----------
        x
            The point(s) at which to evaluate the Hessian. A 2D array of shape (num_points, num_dimensions), or a 1D
            array of shape (num_dimensions).

        Returns
        -------
        hessian : ndarray
            A 3D array of shape (num_points, num_dimensions, num_dimensions) if the input was 2D, or a 2D array of shape
            (num_dimensions, num_dimensions) if the input was 1D. The :math:`(i, j, k)`-th element is the
            :math:`(j, k)`-th mixed partial derivative of the posterior variance at the :math:`i`-th point of `x`.

        Notes
        -----
        Writing :math:`\\pi(x)` for the prior, and :math:`V(x)` for the posterior variance, the posterior variance of
        the product is :math:`\\pi(x)^2 V(x)`.
        """
        _, gp_variance = self.warped_gp.posterior_mean_and_variance(x)
        gp_variance_jacobian = self.warped_gp.posterior_variance_jacobian(x)
        gp_variance_hessian = self.warped_gp.posterior_variance_hessian(x)

        prior = self.prior(x)
        prior_jacobian, prior_hessian = self.prior.gradient(x)

        return hessian_of_f_squared_times_g(
            f=prior, f_jacobian=prior_jacobian, f_hessian=prior_hessian,
            g=gp_variance, g_jacobian=gp_variance_jacobian, g_hessian=gp_variance_hessian)

    def update(self, x: ndarray, y: ndarray) -> None:
        """Add new data to the model.

        Parameters
        ----------
        x
            A 2D array of shape (num_points, num_dimensions), or a 1D array of shape (num_dimensions).
        y
            A 1D array of shape (num_points). If X is 1D, this may also be a 0D array or float.

        Raises
        ------
        ValueError
            If the number of points in `x` does not equal the number of points in `y`.
        """
        self.warped_gp.update(x, y)

    def remove(self, x: Union[ndarray, List[ndarray]], y: Union[ndarray, List[ndarray]]) -> None:
        """Remove data from the model.

        Parameters
        ----------
        x
            A 2D array of shape (num_points, num_dimensions), or a 1D array of shape (num_dimensions), or a list of such
            arrays.
        y
            A 1D array of shape (num_points), or a list of such arrays. If `x` is 1D, this may also be a 0D array or
            float. Must be of the same type as `x`.

        Raises
        ------
        ValueError
            If the number of points in `x` does not equal the number of points in `y`.
            If `x` is an array and `y` is a list, or vice versa.
        """
        self.warped_gp.remove(x, y)

    def integral_mean(self) -> float:
        """Compute the mean of the integral of the function under this model."""
        return _compute_mean(self.prior, self.warped_gp, self.warped_gp.kernel)


@multimethod
def _compute_mean(prior: Prior, gp: WarpedGP, kernel: Kern) -> float:
    """Compute the mean of the integral for the given prior, warped GP, and kernel.

    This method will delegate to other methods of the same name defined in this module, based on the type of the
    arguments. If no implementation is found for the provided types, this default implementation will raise an error."""
    raise NotImplementedError("Integration is not supported for this combination of prior, warping and kernel.\n\n"
                              "Prior was of type {}.\n"
                              "Warped GP was of type {}.\n"
                              "Kernel was of type {}."
                              .format(type(prior), type(gp), type(kernel)))


@multimethod
def _compute_mean(prior: Gaussian, gp: WsabiLGP, kernel: RBF) -> float:
    """Compute the mean of the integral for a WSABI-L GP with a squared exponential kernel against a Gaussian prior."""
    underlying_gp = gp.underlying_gp

    dimensions = gp.dimensions

    alpha = gp.alpha
    kernel_lengthscale = kernel.lengthscale.values[0]
    kernel_variance = kernel.variance.values[0]

    X_D = underlying_gp.X

    mu = prior.mean
    sigma = prior.covariance
    sigma_inv = prior.precision

    nu = (X_D[:, newaxis, :] + X_D[newaxis, :, :]) / 2
    A = underlying_gp.K_inv_Y

    L = np.exp(-(np.linalg.norm(X_D[:, newaxis, :] - X_D[newaxis, :, :], axis=2) ** 2)/(4 * kernel_lengthscale**2))
    L = kernel_variance ** 2 * L
    L = np.linalg.det(2 * np.pi * sigma) ** (-1/2) * L

    C = sigma_inv + 2 * np.eye(dimensions) / kernel_lengthscale ** 2

    C_inv = np.linalg.inv(C)
    gamma_part = 2 * nu / kernel_lengthscale ** 2 + (sigma_inv @ mu)[newaxis, newaxis, :]

    gamma = np.einsum('kl,ijl->ijk', C_inv, gamma_part)

    k_1 = 2 * np.einsum('ijk,ijk->ij', nu, nu) / kernel_lengthscale ** 2
    k_2 = mu.T @ sigma_inv @ mu
    k_3 = np.einsum('ijk,kl,ijl->ij', gamma, C, gamma)

    k = k_1 + k_2 - k_3

    K = np.exp(-k/2)

    integral_mean = alpha + (np.linalg.det(2 * np.pi * np.linalg.inv(C)) ** 0.5)/2 * (A.T @ (K * L) @ A)

    return integral_mean.item()

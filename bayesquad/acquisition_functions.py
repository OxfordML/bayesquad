"""Acquisition functions and related functions."""
import numpy as np

from .plotting import plottable
from .quadrature import IntegrandModel


def model_variance(integrand_model: IntegrandModel):

    @plottable("Model variance", default_plotting_parameters={'calculate_jacobian': False})
    def f(x, *, calculate_jacobian=True):
        """Evaluate the variance, and the Jacobian of the variance, for the given `IntegrandModel` at a point, or a set
        of points.

        Given an array of shape (num_points, num_dimensions), returns an array of shape (num_points) containing the
        function values and an array of shape (num_points, num_dimensions) containing the function Jacobians.

        Given an array of shape (num_dimensions), returns a 0D array containing the function value and an array of shape
        (num_dimensions) containing the function Jacobian.

        If the Jacobian is not required (e.g. for plotting), the relevant calculations can be disabled by setting
        `calculate_jacobian=False`.
        """
        _, variance = integrand_model.posterior_mean_and_variance(x)

        if calculate_jacobian:
            variance_jacobian = integrand_model.posterior_variance_jacobian(x)
        else:
            variance_jacobian = None

        return variance, variance_jacobian

    return f


def model_variance_norm_of_gradient_squared(integrand_model: IntegrandModel):

    @plottable("Gradient squared", default_plotting_parameters={'calculate_jacobian': False})
    def f(x, *, calculate_jacobian=True):
        """Evaluate the squared norm of the gradient of the variance, and the Jacobian of this quantity, for the given
        `IntegrandModel` at a point, or a set of points.

        Given an array of shape (num_points, num_dimensions), returns an array of shape (num_points) containing the
        function values and an array of shape (num_points, num_dimensions) containing the function Jacobians.

        Given an array of shape (num_dimensions), returns a 0D array containing the function value and an array of shape
        (num_dimensions) containing the function Jacobian.

        If the Jacobian is not required (e.g. for plotting), the relevant calculations can be disabled by setting
        `calculate_jacobian=False`.
        """
        variance_jacobian = integrand_model.posterior_variance_jacobian(x)

        # Inner product of the Jacobian with itself, for each point.
        gradient_squared = np.einsum('...i,...i->...', variance_jacobian, variance_jacobian, optimize=True)

        if calculate_jacobian:
            variance_hessian = integrand_model.posterior_variance_hessian(x)

            # Matrix product of Hessian and Jacobian, for each point.
            gradient_squared_jacobian = 2 * np.einsum('...ij,...j->...i',
                                                      variance_hessian,
                                                      variance_jacobian,
                                                      optimize=True)
        else:
            gradient_squared_jacobian = None

        return gradient_squared, gradient_squared_jacobian

    return f

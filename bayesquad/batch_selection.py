"""Methods for selecting a batch of points to evaluate for Bayesian quadrature."""

from math import sqrt
from typing import List

import numpy as np
import numpy.ma as ma
from numpy import ndarray

from .optimisation import multi_start_maximise_log, multi_start_maximise
from .plotting import returns_plottable
from .quadrature import IntegrandModel

LOCAL_PENALISATION = "Local Penalisation"


def select_batch(integrand_model: IntegrandModel,
                 batch_size: int,
                 batch_method: str = LOCAL_PENALISATION) -> List[ndarray]:
    """Select a batch of points at which to evaluate the integrand.

    Parameters
    ----------
    integrand_model
        The model with which we wish to perform Bayesian quadrature.
    batch_size
        The number of points to return in the new batch.
    batch_method
        The method by which to compute the new batch. Currently supported methods are:
            - "Local Penalisation"

    Returns
    -------
    list[ndarray]
        A list of arrays. Each array is a point of the new batch.
    """
    if batch_method == LOCAL_PENALISATION:
        return select_local_penalisation_batch(integrand_model, batch_size)
    else:
        raise NotImplementedError("{} is not a supported batch method.".format(batch_method))


def select_local_penalisation_batch(integrand_model: IntegrandModel, batch_size: int) -> List[ndarray]:
    """Select a batch of points based on a local penalisation method.

    Parameters
    ----------
    integrand_model
        The model with which we wish to perform Bayesian quadrature.
    batch_size
        The number of points to return in the new batch.

    Returns
    -------
    list[ndarray]
        A list of arrays. Each array is a point of the new batch.

    Notes
    -----
    In this method, we sequentially select batch points by maximising an acquisition function (currently, this is fixed
    to be the posterior variance of the integrand). In order to avoid repeatedly selecting the same point (or very
    nearby points) repeatedly, we update the acquisition function after selecting each point to penalise that point and
    a region around it. In the method used here, we first find the maximal gradient of the acquisition function, and
    then take the penalised function to be the minimum of the original function and a cone with half of this maximal
    gradient around the selected point. We choose half the gradient since the true function must have zero gradient
    at the selected point and at the nearest maximum, so the average gradient here will not be as large as the maximum.
    """
    batch = []
    penaliser_gradients = []

    acquisition_function = _model_variance(integrand_model)
    num_initial_points = 10 * integrand_model.dimensions

    while len(batch) < batch_size:
        softmin_penalised_log_acquisition_function = \
            _get_soft_penalised_log_acquisition_function(acquisition_function, batch, penaliser_gradients)

        initial_points = [integrand_model.prior.sample() for _ in range(num_initial_points)]
        batch_point, value = multi_start_maximise(softmin_penalised_log_acquisition_function,
                                                  initial_points)
        batch.append(batch_point)

        if len(batch) < batch_size:
            num_local_initial_points = integrand_model.dimensions * 10
            local_initial_points = _get_local_initial_points(batch_point, num_local_initial_points)

            _, max_gradient_squared = multi_start_maximise_log(_variance_gradient_squared_and_jacobian(integrand_model),
                                                               local_initial_points,
                                                               gtol=1e-1)
            max_gradient = sqrt(max_gradient_squared)

            penaliser_gradients.append(max_gradient / 2)

    return batch


def _get_local_initial_points(central_point, num_points):
    """Get a set of points close to a given point."""
    perturbations = [0.1 * np.random.randn(*central_point.shape) for _ in range(num_points)]
    return [central_point + perturbation for perturbation in perturbations]


@returns_plottable("Model variance")
def _model_variance(integrand_model: IntegrandModel):
    def f(x, *, calculate_jacobian=True):
        """Evaluate the variance, and the jacobian of the variance, for the given `IntegrandModel` at a point, or a set
        of points.

        Given an array of shape (num_points, num_dimensions), returns an array of shape (num_points) containing the
        function values and an array of shape (num_points, num_dimensions) containing the function jacobians.

        Given an array of shape (num_dimensions), returns a 0D array containing the function value and an array of shape
        (num_dimensions) containing the function jacobian.

        If the jacobian is not required (e.g. for plotting), the relevant calculations can be disabled by setting
        `calculate_jacobian=False`.
        """
        _, variance = integrand_model.posterior_mean_and_variance(x)

        if calculate_jacobian:
            variance_jacobian = integrand_model.posterior_variance_jacobian(x)
        else:
            variance_jacobian = None

        return variance, variance_jacobian

    return f


@returns_plottable("Grad squared")
def _variance_gradient_squared_and_jacobian(integrand_model: IntegrandModel):
    def f(x):
        variance_jacobian = integrand_model.posterior_variance_jacobian(x)
        variance_hessian = integrand_model.posterior_variance_hessian(x)

        # Inner product of the jacobian with itself, for each point.
        gradient_squared = np.einsum('...i,...i->...', variance_jacobian, variance_jacobian, optimize=True)

        # Matrix product of hessian and jacobian, for each point.
        gradient_squared_jacobian = 2 * np.einsum('...ij,...j->...i',
                                                  variance_hessian,
                                                  variance_jacobian,
                                                  optimize=True)

        return gradient_squared, gradient_squared_jacobian

    return f


def _get_penalised_acquisition_function(acquisition_function, penaliser_centres, penaliser_gradients):
    """Create a function which will return the minimum of the given acquisition function and the given penalisers at
    any point, or set of points.

    The penalisers take the form of a cone around a central point.
    """
    penalisers = [_cone(centre, gradient) for centre, gradient in zip(penaliser_centres, penaliser_gradients)]

    def penalised_acquisition_function(x):
        function_evaluations = [acquisition_function(x)] + [f(x) for f in penalisers]
        function_values, function_jacobians = [np.array(ret) for ret in zip(*function_evaluations)]

        # This is necessary to ensure that function_values has the same dimensions as function_jacobians, so that we can
        # index into both arrays in a consistent manner.
        function_values = np.expand_dims(function_values, -1)

        min_indices = np.argmin(function_values, axis=0)

        values = np.choose(min_indices, function_values)
        jacobians = np.choose(min_indices, function_jacobians)

        return values.squeeze(), jacobians.squeeze()

    return penalised_acquisition_function


@returns_plottable("Soft penalised log acquisition function")
def _get_soft_penalised_log_acquisition_function(acquisition_function, penaliser_centres,
                                                 penaliser_gradients):
    """Create a function which will return the log of a soft minimum of the given acquisition function and the given
    penalisers at any point, or set of points.

    The soft minimisation is performed by taking the p-norm of all function values for a negative value of p. This
    gives a differentiable function which is approximately equal to the min of the given functions.

    If the jacobian is not required (e.g. for plotting), the relevant calculations can be disabled by setting
    `calculate_jacobian=False`.
    """
    penalisers = [_cone(centre, gradient) for centre, gradient in zip(penaliser_centres, penaliser_gradients)]
    p = 6

    def penalised_acquisition_function(x, *, calculate_jacobian=True):
        function_evaluations = \
            [acquisition_function(x, calculate_jacobian=calculate_jacobian)] + [f(x) for f in penalisers]
        function_values, function_jacobians = [np.array(ret) for ret in zip(*function_evaluations)]

        # This is necessary to ensure that function_values has the same dimensions as function_jacobians, so that we can
        # index into both arrays in a consistent manner.
        function_values = np.expand_dims(function_values, -1)

        # We want to avoid dividing by zero and taking the log of zero, so we mask out all zeroes.
        function_values = ma.masked_equal(function_values, 0)

        min_function_values = np.min(function_values, axis=0)
        min_function_values = ma.masked_array(min_function_values, mask=np.any(function_values.mask, axis=0))

        # Any values more than roughly an order of magnitude from the minimum value will be irrelevant to the final
        # result, but might cause overflows, so we clip them here.
        scaled_function_values = (function_values / min_function_values).clip(max=1e2)

        scaled_inverse_power_sum = (1 / (scaled_function_values ** p)).sum(axis=0)
        values = -ma.log(scaled_inverse_power_sum) / p + ma.log(min_function_values)
        values = ma.filled(values, -1e3)

        if calculate_jacobian:
            scaled_function_jacobians = (function_jacobians / min_function_values).clip(max=1e2, min=-1e2)
            jacobian_numerator = (1 / (scaled_function_values ** (p + 1)) * scaled_function_jacobians).sum(axis=0)
            jacobians = jacobian_numerator / scaled_inverse_power_sum
            jacobians = ma.filled(jacobians, np.random.randn())
        else:
            jacobians = None

        return values.squeeze(), jacobians

    return penalised_acquisition_function


def _cone(centre, gradient):
    def f(x):
        """Evaluate a cone around the given centre with the given gradient, i.e. a function whose value increases
        linearly with distance from the centre.

        Given an array of shape (num_points, num_dimensions), returns an array of shape (num_points) containing the
        function values and an array of shape (num_points, num_dimensions) containing the function jacobians.

        Given an array of shape (num_dimensions), returns a 0D array containing the function value and an array of shape
        (num_dimensions) containing the function jacobian.
        """
        distance = np.linalg.norm(x - centre, axis=-1)

        value = distance * gradient

        distance = np.expand_dims(distance, -1)
        distance = ma.masked_equal(distance, 0)  # Avoid division by zero

        jacobian = (x - centre) * gradient / distance

        # The jacobian isn't defined at the centre of the cone but we return a value to keep the optimiser happy.
        jacobian = ma.filled(jacobian, x)

        return value, jacobian

    return f

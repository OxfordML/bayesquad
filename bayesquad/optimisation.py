"""Utility functions wrapping a scipy optimizer."""

from functools import wraps
from typing import Tuple, Callable, List

import numpy as np
import scipy.optimize
from numpy import ndarray

DEFAULT_GTOL = 1e-2

DEFAULT_MINIMIZER_KWARGS = {'method': 'BFGS',
                            'jac': True,
                            'options': {'gtol': DEFAULT_GTOL}}


def multi_start_maximise(objective_function: Callable,
                         initial_points: List[ndarray], **kwargs) -> Tuple[ndarray, float]:
    """Run multi-start maximisation of the given objective function.

    Warnings
    --------
    This is a hack to take advantage of fast vectorised computation and avoid expensive python loops. There may be some
    issues with this method!

    The objective function provided here must be a vectorised function. We take advantage of the fast computation of
    vectorised functions to view a multi-start optimisation as a single pass of a higher-dimensional optimisation,
    rather than several passes of a low-dimensional optimisation (which would require an expensive python loop). We
    simply concatenate all the points where the function is to be evaluated into a single high-dimensional vector, give
    the function value as the sum of all the individual function values, and give the jacobian as the concatenation of
    all the individual jacobians. In this way we can essentially perform many optimisations in parallel. Note that
    (among other issues) there is an issue here with the stopping condition: we can only consider all optimisations
    together, so even if most have come very close to an optimum, the process will continue as long as one is far away.
    However, this does seem to perform well in practice.

    Parameters
    ----------
    objective_function
        Function to be maximised. Must return both the function value and the jacobian. Must also accept a 2D array of
        points, returning a 1D array and a 2D array for the function values and jacobians respectively.
    initial_points
        A list of arrays, each of shape (num_dimensions).
    **kwargs
        Keyword arguments will be included in the 'options' dict passed to the underlying scipy optimiser.

    Returns
    -------
    ndarray
        The location of the found maximum.
    float
        The value of the objective function at the found maximum.
    """
    minimizer_kwargs = DEFAULT_MINIMIZER_KWARGS.copy()
    minimizer_kwargs['options'] = {**minimizer_kwargs['options'], **kwargs}  # This merges the two dicts.

    initial_point = np.concatenate(initial_points)
    num_initial_points = len(initial_points)
    num_dims = len(initial_points[0])

    gtol = minimizer_kwargs['options']['gtol']

    optimisation_terminated = np.full(num_initial_points, False)
    located_optima = np.zeros((num_initial_points, num_dims))
    located_optimal_values = np.full(num_initial_points, np.nan)

    def function_to_minimise(x, *inner_args, **inner_kwargs):
        nonlocal optimisation_terminated

        x = np.reshape(x, (num_initial_points, num_dims))

        value, jacobian = objective_function(x, *inner_args, **inner_kwargs)

        jacobian_norms = np.linalg.norm(jacobian, axis=1, ord=np.inf)
        new_optimum_found = np.logical_and(jacobian_norms < gtol,
                                           np.logical_not(optimisation_terminated))

        new_optima_indices = _indices_where(new_optimum_found)

        located_optima[new_optima_indices] = x[new_optima_indices]
        located_optimal_values[new_optima_indices] = value[new_optima_indices]

        optimisation_terminated = np.logical_or(optimisation_terminated,
                                                (jacobian_norms < gtol))
        terminated_indices = _indices_where(optimisation_terminated)

        value[terminated_indices] = located_optimal_values[terminated_indices]

        dummy_jacobian = np.zeros(num_dims)
        jacobian[terminated_indices] = dummy_jacobian

        combined_value, combined_jacobian = -value.sum(), -jacobian.ravel()

        if not np.isfinite(combined_value) or not np.all(np.isfinite(combined_jacobian)):
            raise FloatingPointError("Objective function for multi-start optimisation returned NaN or infinity.")

        return combined_value, combined_jacobian

    maximum = scipy.optimize.minimize(function_to_minimise, initial_point, **minimizer_kwargs)
    maxima = maximum.x.reshape(num_initial_points, num_dims)

    failed_indices = _indices_where(np.isnan(located_optimal_values))
    located_optima[failed_indices] = maxima[failed_indices]

    values, _ = objective_function(located_optima)
    max_index = np.argmax(values)

    optimal_x = maxima[max_index, :]
    optimal_y = values[max_index]

    return optimal_x, optimal_y


def multi_start_maximise_slow(objective_function: Callable,
                         initial_points: List[ndarray], **kwargs) -> Tuple[ndarray, float]:
    """Run multi-start maximisation of the given objective function.

    This function performs the multi-start maximisation naively by running a Python loop over the initial points.

    Parameters
    ----------
    objective_function
        Function to be maximised. Must return both the function value and the jacobian. Must also accept a 2D array of
        points, returning a 1D array and a 2D array for the function values and jacobians respectively.
    initial_points
        A list of arrays, each of shape (num_dimensions).
    **kwargs
        Keyword arguments will be included in the 'options' dict passed to the underlying scipy optimiser.

    Returns
    -------
    ndarray
        The location of the found maximum.
    float
        The value of the objective function at the found maximum.
    """
    minimizer_kwargs = DEFAULT_MINIMIZER_KWARGS.copy()
    minimizer_kwargs['options'] = {**minimizer_kwargs['options'], **kwargs}  # This merges the two dicts.

    def function_to_minimise(x, *inner_args, **inner_kwargs):
        value, jacobian = objective_function(x, *inner_args, **inner_kwargs)

        return -value, -jacobian

    all_maxima = [scipy.optimize.minimize(function_to_minimise, initial_point, **minimizer_kwargs)
                  for initial_point in initial_points]
    maxima_x, maxima_y = zip(*[(opt.x, -opt.fun) for opt in all_maxima])

    max_index = np.argmax(maxima_y)

    optimal_x = maxima_x[max_index]
    optimal_y = maxima_y[max_index]

    return optimal_x, optimal_y


def multi_start_maximise_log(objective_function: Callable,
                             initial_points: List[ndarray], **kwargs) -> Tuple[ndarray, float]:
    """Maximise the given objective function in log space. This may be significantly easier for functions with a high
    dynamic range.

    See Also
    --------
    :func:`~multi_start_maximise` : `multi_start_maximise_log` is a thin wrapper around this function. See this function
    for further details on parameters and return values.
    """
    @wraps(objective_function)
    def log_objective_function(x, *inner_args, **inner_kwargs):
        import numpy.ma as ma

        value, jacobian = objective_function(x, *inner_args, **inner_kwargs)
        masked_value = ma.masked_equal(value, 0)

        log_value = ma.log(masked_value)

        # We need expand_dims here because value is lower-dimensional than jacobian, but they must have the same
        # dimensionality for numpy broadcasting to work here.
        log_jacobian = jacobian / ma.expand_dims(masked_value, -1)

        log_value = ma.filled(log_value, -1e100)
        log_jacobian = ma.filled(log_jacobian, np.random.randn())

        return log_value, log_jacobian

    optimal_x, optimal_value = multi_start_maximise(log_objective_function, initial_points, **kwargs)

    return optimal_x, np.exp(optimal_value)


def _indices_where(array: ndarray) -> Tuple:
    """Returns the indices where the elements of `array` are True."""
    return np.nonzero(array)

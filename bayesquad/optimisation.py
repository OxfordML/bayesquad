"""Utility functions wrapping a scipy optimizer."""

from functools import wraps
from typing import Tuple, Callable, List

import numpy as np
import scipy.optimize
from numpy import ndarray

DEFAULT_MINIMIZER_KWARGS = {'method': 'BFGS',
                            'jac': True,
                            'options': {'gtol': 1e-2}}


def multi_start_maximise_experimental(objective_function: Callable,
                                      initial_points: List[ndarray], **kwargs) -> Tuple[ndarray, float]:
    """Run multi-start maximisation of the given objective function.

    Warnings
    --------
    This is a hack to take advantage of fast vectorised computation and avoid expensive python loops. There are some
    issues with this method!

    The objective function provided here must be a vectorised function. We take advantage of the fast computation of
    vectorised functions to view a multi-start optimisation as a single pass of a higher-dimensional optimisation,
    rather than several passes of a low-dimensional optimisation (which would require an expensive python loop). We
    simply concatenate all the points where the function is to be evaluated into a single high-dimensional vector, give
    the function value as the sum of all the individual function values, and give the jacobian as the concatenation of
    all the individual jacobians. In this way we can essentially perform many optimisations in parallel. Note that
    (among other issues) there is an issue here with the stopping condition: we can only consider all optimisations
    together, so even if most have come very close to an optimum, the process will continue as long as one is far away.
    However, this hack does seem to perform well in practice.

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

    # The magnitude of the jacobian will grow with the number of initial points once we concatenate them all, so we need
    # to up the gradient tolerance.
    gtol = minimizer_kwargs['options']['gtol']
    minimizer_kwargs['options']['gtol'] = gtol * np.sqrt(num_initial_points)

    def function_to_minimise(x, *inner_args, **inner_kwargs):
        x = np.reshape(x, (num_initial_points, num_dims))

        value, jacobian = objective_function(x, *inner_args, **inner_kwargs)

        return -value.sum(), -jacobian.ravel()

    maximum = scipy.optimize.minimize(function_to_minimise, initial_point, **minimizer_kwargs)
    maxima = maximum.x.reshape(num_initial_points, num_dims)

    values, _ = objective_function(maxima)
    max_index = np.argmax(values)

    optimal_x = maxima[max_index, :]
    optimal_y = values[max_index]

    return optimal_x, optimal_y


def multi_start_maximise(objective_function: Callable,
                         initial_points: List[ndarray], **kwargs) -> Tuple[ndarray, float]:
    """Run multi-start maximisation of the given objective function.

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
    :func:`~multi_start_maximise_experimental` : `multi_start_maximise_log` is a thin wrapper around this function. See
    this function for further details on parameters and return values.
    """
    @wraps(objective_function)
    def log_objective_function(x, *inner_args, **inner_kwargs):
        value, jacobian = objective_function(x, *inner_args, **inner_kwargs)

        log_value = np.log(value)

        # We need expand_dims here because value is lower-dimensional than jacobian, but they must have the same
        # dimensionality for numpy broadcasting to work here.
        log_jacobian = jacobian / np.expand_dims(value, -1)

        return log_value, log_jacobian

    optimal_x, optimal_value = multi_start_maximise(log_objective_function, initial_points, **kwargs)

    return optimal_x, np.exp(optimal_value)

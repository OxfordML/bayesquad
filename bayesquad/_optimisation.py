"""Utility functions wrapping a scipy optimizer."""

from typing import Tuple, Callable

import numpy as np
import scipy.optimize
from numpy import ndarray

DEFAULT_GTOL = 1e-2

DEFAULT_MINIMIZER_KWARGS = {'method': 'BFGS',
                            'jac': True,
                            'options': {'gtol': DEFAULT_GTOL}}


def multi_start_maximise(objective_function: Callable,
                         initial_points: ndarray, **kwargs) -> Tuple[ndarray, float]:
    """Run multi-start maximisation of the given objective function.

    Warnings
    --------
    This is a hack to take advantage of fast vectorised computation and avoid expensive python loops. There may be some
    issues with this method!

    The objective function provided here must be a vectorised function. We take advantage of the fast computation of
    vectorised functions to view a multi-start optimisation as a single pass of a higher-dimensional optimisation,
    rather than several passes of a low-dimensional optimisation (which would require an expensive python loop). We
    simply concatenate all the points where the function is to be evaluated into a single high-dimensional vector, give
    the function value as the sum of all the individual function values, and give the Jacobian as the concatenation of
    all the individual Jacobians. In this way we can essentially perform many optimisations in parallel. Note that
    there is an issue here with the stopping condition: we can only consider all optimisations together, so even if most
    have come very close to an optimum, the process will continue as long as one is far away. However, this does seem to
    perform well in practice.

    Parameters
    ----------
    objective_function
        Function to be maximised. Must return both the function value and the Jacobian. Must also accept a 2D array of
        points, returning a 1D array and a 2D array for the function values and Jacobians respectively.
    initial_points
        Points at which to begin the optimisation, as a 2D array of shape (num_points, num_dimensions).
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

    num_points, num_dims = np.shape(initial_points)

    def function_to_minimise(x, *inner_args, **inner_kwargs):
        x = np.reshape(x, (num_points, num_dims))

        value, jacobian = objective_function(x, *inner_args, **inner_kwargs)
        combined_value, combined_jacobian = -value.sum(), -jacobian.ravel()

        if not np.isfinite(combined_value) or not np.all(np.isfinite(combined_jacobian)):
            raise FloatingPointError("Objective function for multi-start optimisation returned NaN or infinity.")

        return combined_value, combined_jacobian

    maximum = scipy.optimize.minimize(function_to_minimise, initial_points, **minimizer_kwargs)
    maxima = maximum.x.reshape(num_points, num_dims)

    values, _ = objective_function(maxima)
    max_index = np.argmax(values)

    optimal_x = maxima[max_index, :]
    optimal_y = values[max_index]

    return optimal_x, optimal_y


def _indices_where(array: ndarray) -> Tuple:
    """Returns the indices where the elements of `array` are True."""
    return np.nonzero(array)

"""Decorators to modify function behaviour."""
from functools import wraps

import numpy as np
from numpy import ndarray


def flexible_array_dimensions(func):
    """Modifies a function so that it can accept either 1D or 2D arrays, and return arrays of consistent dimension.

    This decorator allows a vectorised function to be evaluated at a single point, passed as a 1D array. It is intended
    to be applied to a function whose array arguments have first dimension ranging across data points, and whose return
    values are also arrays with first dimension ranging across data points. After this decorator has been applied, the
    function may be evaluated at a single point by passing a 1D array, and the trivial first axis will be removed from
    the returned arrays. Within a method which has this decorator applied, we may assume that all array arguments are
    2D, with shape (num_points, num_dimensions).

    NB this means that a 1D array of length n will be interpreted as a single n-dimensional point, rather than n
    1-dimensional points.
    """
    @wraps(func)
    def transformed_function(*args, **kwargs):
        new_args = tuple(np.atleast_2d(arg) if isinstance(arg, ndarray)
                         else arg
                         for arg in args)

        values = func(*new_args, **kwargs)

        if all([arg is new_arg for arg, new_arg in zip(args, new_args)]):
            return values

        if isinstance(values, tuple):
            return tuple(np.squeeze(value, axis=0) if isinstance(value, ndarray) and np.size(value, 0) == 1
                         else value
                         for value in values)
        elif isinstance(values, ndarray) and np.size(values, 0) == 1:
            return np.squeeze(values, axis=0)
        else:
            return values

    return transformed_function

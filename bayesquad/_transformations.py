"""Transform functions"""
from functools import wraps
from typing import Callable

import numpy as np
import numpy.ma as ma


def log_of_function(original_function: Callable) -> Callable:
    """Given a function f and its Jacobian, return the log(f) and the Jacobian of log(f).

    f may evaluate to 0 in some places (e.g. due to numerical issues), so we set the log to be -1e10 where f is 0.
    Since it's not possible to evaluate the log Jacobian in this case, we set the Jacobian to be the Jacobian of the
    original function.

    Parameters
    ----------
    original_function
        A function returning a tuple of arrays (f(x), Jac(f(x))). The second element of this tuple may be `None`.

    Returns
    -------
    log_function : Callable
        A function returning a tuple of arrays (log(f(x)), Jac(log(f(x)))). If `original_function` returns `None` for
        Jac(f(x)), then Jac(log(f(x))) will also be `None`.
    """
    @wraps(original_function)
    def log_function(x, *args, **kwargs):
        value, jacobian = original_function(x, *args, **kwargs)
        masked_value = ma.masked_equal(value, 0)

        log_value = ma.log(masked_value)
        log_value = ma.filled(log_value, -1e10)

        if jacobian is not None:
            # We need expand_dims here because value is lower-dimensional than jacobian, but they must have the same
            # dimensionality for numpy broadcasting to work here.
            masked_value = ma.filled(masked_value, 1)
            log_jacobian = jacobian / np.expand_dims(masked_value, -1)
        else:
            log_jacobian = None

        return log_value, log_jacobian

    return log_function

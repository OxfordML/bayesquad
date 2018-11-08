"""Miscellaneous utility functions"""

import numpy as np
from numpy import ndarray


def validate_dimensions(x: ndarray, expected_dimensionality: int):
    """Checks that `x` represents data of dimensionality `expected_dimensions`.

    Raises
    ------
    ValueError
        If `x` is not a 2D array, or if the second dimension of `x` does not have size `expected_dimensions`.
    """
    array_dimensions = np.ndim(x)

    if array_dimensions != 2:
        raise ValueError("Expected a 2-dimensional array, but got a {}-dimensional array.".format(array_dimensions))

    actual_dimensionality = np.size(x, 1)

    if actual_dimensionality != expected_dimensionality:
        raise ValueError("Expected data in {} dimensions, but got data in {} dimensions."
                         .format(expected_dimensionality, actual_dimensionality))

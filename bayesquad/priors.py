"""Classes representing probability distributions, intended to be integrated against a likelihood."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy.stats
from numpy import ndarray, newaxis

from ._util import validate_dimensions


class Prior(ABC):
    """A prior, providing methods for sampling, and for pointwise evaluation of the pdf and its derivatives."""

    @abstractmethod
    def gradient(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """Compute the jacobian and hessian of the prior's pdf at the given set of points.

        Parameters
        ----------
        x
            A 2D array of the points at which to evaluate the derivatives, with shape (num_points, num_dimensions).

        Returns
        -------
        jacobian
            A 2D array of shape (num_points, num_dimensions), containing the value of the jacobian at each point.
        hessian
            A 3D array of shape (num_points, num_dimensions, num_dimensions), whose (i, j, k)-th element is the
            (j, k)-th mixed partial derivative of the pdf at the i-th point of `x`.
        """

    @abstractmethod
    def sample(self) -> ndarray:
        """Sample a point from the prior.

        Returns
        -------
        ndarray
            A sample from the prior, as a 1D array of shape (num_dimensions).
        """

    @abstractmethod
    def __call__(self, x: ndarray) -> ndarray:
        """Evaluate the prior's pdf at the given set of points.

        Parameters
        ----------
        x
            An array of shape (num_points, num_dimensions).

        Returns
        -------
        ndarray
            A 1D array of shape (num_points).
        """


class Gaussian(Prior):
    """A multivariate Gaussian prior.

    Parameters
    ----------
    mean
        A 1D array of shape (num_dimensions).
    covariance
        A 2D array of shape (num_dimensions, num_dimensions).

    Attributes
    ----------
    mean : ndarray
        A 1D array of shape (num_dimensions).
    covariance : ndarray
        A 2D array of shape (num_dimensions, num_dimensions).
    precision : ndarray
        The inverse of the covariance matrix.
    """
    def __init__(self, mean: ndarray, covariance: ndarray):
        self.mean = mean
        self.covariance = covariance
        self.precision = np.linalg.inv(covariance)

        self._dimensions = np.size(mean)
        self._multivariate_normal = scipy.stats.multivariate_normal(mean=mean, cov=covariance)

    def sample(self) -> ndarray:
        """See :func:`~Prior.sample`"""
        return np.atleast_1d(self._multivariate_normal.rvs())

    def gradient(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """See :func:`~Prior.gradient`"""
        validate_dimensions(x, self._dimensions)

        # The (i, j)-th element of this is (covariance^-1 (x_i - mean))_j, where x_i is the i-th point of x.
        cov_inv_x = np.einsum('jk,ik->ij', self.precision, x - self.mean, optimize=True)

        jacobian = -self(x)[:, newaxis] * cov_inv_x

        # The outer product of each row of cov_inv_x with itself.
        outer_products = cov_inv_x[:, newaxis, :] * cov_inv_x[:, :, newaxis]

        hessian = self(x)[:, newaxis, newaxis] * (outer_products - self.precision[newaxis, :, :])

        return jacobian, hessian

    def __call__(self, x: ndarray) -> ndarray:
        """See :func:`~Prior.__call__`"""
        validate_dimensions(x, self._dimensions)
        return np.atleast_1d(self._multivariate_normal.pdf(x))

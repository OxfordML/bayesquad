"""Functions for computing the gradients of Gaussian Process kernels."""

import numpy as np

from GPy.kern.src.kern import Kern
from GPy.kern.src.rbf import RBF
from GPy.kern.src.stationary import Stationary
from numpy import ndarray, newaxis


def jacobian(kernel: Kern, variable_points: ndarray, fixed_points: ndarray) -> ndarray:
    """Return the Jacobian of a kernel evaluated at all pairs from two sets of points.

    Given a kernel and two sets :math:`X, D` of points (`variable_points` and `fixed_points` respectively), this
    function will evaluate the Jacobian of the kernel at each pair of points :math:`(x, d)` with :math:`x \in X` and
    :math:`d \in D`. The derivative is taken with respect to the first argument, i.e. :math:`d` is regarded as a
    fixed quantity. Typically, :math:`D` will be the set of :math:`x` values in the data set of a Gaussian Process, and
    :math:`X` will be the set of :math:`x` values at which we wish to find the gradient of the posterior GP.

    Parameters
    ----------
    kernel
        The kernel to be differentiated. Currently supported kernels are:
            - :class:`GPy.kern.src.rbf.RBF`
    variable_points
        A 2D array of points, with shape (num_variable_points, num_dimensions).
    fixed_points
        A 2D array of points, with shape (num_fixed_points, num_dimensions).

    Returns
    -------
    ndarray
        A 3D array of shape (num_variable_points, num_fixed_points, num_dimensions), whose (i, j, k)-th element is the
        k-th component of the Jacobian of the kernel evaluated at the i-th point of :math:`X` and the j-th point of
        :math:`D`.

    Raises
    ------
    NotImplementedError
        If the provided kernel type is not supported. See the parameters list for a list of supported kernels.
    """
    if isinstance(kernel, RBF):
        lengthscale = kernel.lengthscale.values[0]
        k = kernel.K(variable_points, fixed_points)

        # The (i, j, k)-th element of this is the k-th component of X_i - D_j.
        differences = variable_points[:, newaxis, :] - fixed_points[newaxis, :, :]

        return -k[:, :, newaxis] * differences / (lengthscale ** 2)
    else:
        raise NotImplementedError


def hessian(kernel: Kern, variable_points: ndarray, fixed_points: ndarray) -> ndarray:
    """Return the Hessian of a kernel evaluated at all pairs from two sets of points.

    Given a kernel and two sets :math:`X, D` of points (`variable_points` and `fixed_points` respectively), this
    function will evaluate the Hessian of the kernel at each pair of points :math:`(x, d)` with :math:`x \in X` and
    :math:`d \in D`. The derivatives are taken with respect to the first argument, i.e. :math:`d` is regarded as a
    fixed quantity. Typically, :math:`D` will be the set of :math:`x` values in the data set of a Gaussian Process, and
    :math:`X` will be the set of :math:`x` values at which we wish to find the gradient of the posterior GP.

    Parameters
    ----------
    kernel
        The kernel to be differentiated. Currently supported kernels are:
            - :class:`GPy.kern.src.rbf.RBF`
    fixed_points
        A 2D array of points, with shape (num_variable_points, num_dimensions).
    variable_points
        A 2D array of points, with shape (num_fixed_points, num_dimensions).

    Returns
    -------
    ndarray
        A 4D array of shape (num_variable_points, num_fixed_points, num_dimensions, num_dimensions), whose
        (i, j, k, l)-th element is the (k, l)-th mixed partial derivative of the kernel evaluated at the i-th point of
        :math:`X` and the j-th point of :math:`D`.

    Raises
    ------
    NotImplementedError
        If the provided kernel type is not supported. See the parameters list for a list of supported kernels.
    """
    if isinstance(kernel, RBF):
        lengthscale = kernel.lengthscale.values[0]
        k = kernel.K(variable_points, fixed_points)

        _, num_dimensions = variable_points.shape

        # The (i, j, k)-th element of this is the k-th component of X_i - D_j (i.e. (X_i - D_j)_k).
        differences = variable_points[:, newaxis, :] - fixed_points[newaxis, :, :]

        # The (i, j, k, l)-th element of this is (X_i - D_j)_k * (X_i - D_j)_l. This can be viewed as a matrix of
        # matrices, whose (i, j)-th matrix is the outer product of (X_i - D_j) with itself.
        outer_products_of_differences = np.einsum('ijk,ijl->ijkl', differences, differences, optimize=True)

        transformed_outer_products = (outer_products_of_differences / lengthscale ** 2) - np.eye(num_dimensions)

        # Now multiply the (i, j)-th transformed outer product by K(X_i, D_j).
        product = np.einsum('ij,ijkl->ijkl', k, transformed_outer_products, optimize=True)

        return product / (lengthscale ** 2)
    else:
        raise NotImplementedError


def diagonal_hessian(kernel: Kern, x: ndarray) -> ndarray:
    """Return the Hessian of a kernel considered as a function of one variable by constraining both inputs to be equal.

    Given a kernel :math:`K` and a set of points :math:`X`, this function will evaluate the Hessian of :math:`K(x, x)`
    at each point :math:`x` of :math:`X`.

    Parameters
    ----------
    kernel
        The kernel to be differentiated. Currently supported kernels are:
            - All subclasses of :class:`GPy.kern.src.rbf.Stationary`
    x
        A 2D array of points, with shape (num_points, num_dimensions).

    Returns
    -------
    ndarray
        A 3D array of shape (num_points, num_dimensions, num_dimensions).
    """
    if isinstance(kernel, Stationary):
        num_points, num_dimensions = x.shape

        return np.zeros((num_points, num_dimensions, num_dimensions))
    else:
        raise NotImplementedError

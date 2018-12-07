"""A home for mathematical operations which are used multiple times in this package."""

from numpy import ndarray, newaxis


def jacobian_of_f_squared_times_g(*,
        f: ndarray, f_jacobian: ndarray,
        g: ndarray, g_jacobian: ndarray) -> ndarray:
    """Given two functions f and g, along with their Jacobians, returns the Jacobian of the function f^2 * g.

    Parameters
    ----------
    f
        A 1D array whose :math:`i`-th element is the value of the function :math:`f` at point :math:`x_i`.
    f_jacobian
        A 2D array whose :math:`(i,j)`-th element is the :math:`j`-th component of the Jacobian of :math:`f` at point
        :math:`x_i`.
    g
        A 1D array whose :math:`i`-th element is the value of the function :math:`g` at point :math:`x_i`.
    g_jacobian
        A 2D array whose :math:`(i,j)`-th element is the :math:`j`-th component of the Jacobian of :math:`g` at point
        :math:`x_i`.

    Returns
    -------
    jacobian : ndarray
        A 2D array of shape (num_points, num_dimensions). The :math:`(i, j)`-th element is the :math:`j`-th component of
        the Jacobian of :math:`f^2 g` at point :math:`x_i`.

    Notes
    -----
    The required derivative is as follows:

    .. math::

        \\frac{\\partial f^2 g}{\\partial x_j} = 2 f g \\frac{\\partial f}{\\partial x_j}
        + f^2 \\frac{\\partial g}{\\partial x_j}
    """
    assert f.ndim == g.ndim == 1, "Function data must be a 1-dimensional array"
    assert f_jacobian.ndim == g_jacobian.ndim == 2, "Function Jacobian data must be a 2-dimensional array"

    # The Jacobian has dimensions (num_points, num_dimensions). For NumPy to broadcast the calculations
    # appropriately, we need to augment our 1D variables with a new axis.
    f, g = f[:, newaxis], g[:, newaxis]

    jacobian = 2 * f * g * f_jacobian + g_jacobian * f ** 2

    return jacobian


def hessian_of_f_squared_times_g(*,
        f: ndarray, f_jacobian: ndarray, f_hessian: ndarray,
        g: ndarray, g_jacobian: ndarray, g_hessian: ndarray) -> ndarray:
    """Given two functions f and g, along with their Jacobian and Hessian, returns the Hessian of the function f^2 * g.

    Parameters
    ----------
    f
        A 1D array whose :math:`i`-th element is the value of the function :math:`f` at point :math:`x_i`.
    f_jacobian
        A 2D array whose :math:`(i,j)`-th element is the :math:`j`-th component of the Jacobian of :math:`f` at point
        :math:`x_i`.
    f_hessian
        A 3D array whose :math:`(i,j,k)`-th element is the :math:`(j,k)`-th mixed partial derivative of :math:`f` at
        point :math:`x_i`.
    g
        A 1D array whose :math:`i`-th element is the value of the function :math:`g` at point :math:`x_i`.
    g_jacobian
        A 2D array whose :math:`(i,j)`-th element is the :math:`j`-th component of the Jacobian of :math:`g` at point
        :math:`x_i`.
    g_hessian
        A 3D array whose :math:`(i,j,k)`-th element is the :math:`(j,k)`-th mixed partial derivative of :math:`g` at
        point :math:`x_i`.

    Returns
    -------
    hessian : ndarray
        A 3D array of shape (num_points, num_dimensions, num_dimensions). The :math:`(i, j, k)`-th element is the
        :math:`(j, k)`-th mixed partial derivative of :math:`f^2 g` at point :math:`x_i`.

    Notes
    -----
    The required derivatives are as follows:

    .. math::

        \\frac{\\partial f^2 g}{\\partial x_j} & = & 2 f g \\frac{\\partial f}{\\partial x_j}
        + f^2 \\frac{\\partial g}{\\partial x_j} \\\\
        \\frac{\\partial^2 f^2 g}{\\partial x_j \\partial x_k} & = &
        2 f \\left( g \\frac{\\partial^2 f}{\\partial x_j \\partial x_k}
        + \\frac{\\partial g}{\\partial x_j} \\frac{\\partial f}{\\partial x_k}
        + \\frac{\\partial f}{\\partial x_j} \\frac{\\partial g}{\\partial x_k} \\right) \\\\
        & & + 2 g \\frac{\\partial f}{\\partial x_j} \\frac{\\partial f}{\\partial x_k}
        + f^2 \\frac{\\partial^2 f}{\\partial x_j \\partial x_k}
    """
    assert f.ndim == g.ndim == 1, "Function data must be a 1-dimensional array"
    assert f_jacobian.ndim == g_jacobian.ndim == 2, "Function Jacobian data must be a 2-dimensional array"
    assert f_hessian.ndim == g_hessian.ndim == 3, "Function Hessian data must be a 3-dimensional array"

    # The Hessian has dimensions (num_points, num_dimensions, num_dimensions). For NumPy to broadcast the calculations
    # appropriately, we need to augment our 1D variables with new axes.
    f, g = f[:, newaxis, newaxis], g[:, newaxis, newaxis]

    # The (i,j,k)-th element of these arrays is the j-th component of the Jacobian at x_i (the k axis has size 1).
    f_jacobian_dxj, g_jacobian_dxj = f_jacobian[:, :, newaxis], g_jacobian[:, :, newaxis]

    # The (i,j,k)-th element of these arrays is the k-th component of the Jacobian at x_i (the j axis has size 1).
    f_jacobian_dxk, g_jacobian_dxk = f_jacobian[:, newaxis, :], g_jacobian[:, newaxis, :]

    hessian = \
        2 * f * (
                f_hessian * g +
                g_jacobian_dxj * f_jacobian_dxk +
                f_jacobian_dxj * g_jacobian_dxk
        ) + 2 * g * f_jacobian_dxj * f_jacobian_dxk \
        + g_hessian * f ** 2

    return hessian

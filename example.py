"""Plot bayesian quadrature on a simple 2D test function."""

from typing import Dict, Any

import GPy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

import bayesquad.plotting as plotting
from bayesquad.batch_selection import select_batch, LOCAL_PENALISATION
from bayesquad.gps import WsabiLGP
from bayesquad.priors import Gaussian
from bayesquad.quadrature import IntegrandModel


# Set up test function and WSABI-L model.

def true_function(x):
    x = np.atleast_2d(x)
    return np.atleast_2d((((np.sin(x) + 0.5 * np.cos(3 * x))**2)/((x/2)**2+0.3)).prod(axis=1))


initial_x = np.array([[0, 0]])
initial_y = np.sqrt(2 * true_function(initial_x))

k = GPy.kern.RBF(2, variance=2, lengthscale=2)
lik = GPy.likelihoods.Gaussian(variance=1e-10)

prior = Gaussian(mean=np.array([0, 0]), covariance=2*np.eye(2))

gpy_gp = GPy.core.GP(initial_x, initial_y, kernel=k, likelihood=lik)
warped_gp = WsabiLGP(gpy_gp)
model = IntegrandModel(warped_gp, prior)


def true_integrand(x):
    return true_function(x) * prior(x)


# Set up plotting.

LOWER_LIMIT = -4
UPPER_LIMIT = 4
PLOTTING_RESOLUTION = 200
COLOUR_MAP = 'summer'


def get_plotting_domain(lower_limit, upper_limit, resolution):
    x = np.linspace(lower_limit, upper_limit, resolution)
    y = np.linspace(lower_limit, upper_limit, resolution)
    x_grid, y_grid = np.meshgrid(x, y)
    return np.concatenate(np.dstack([x_grid, y_grid]))


figure = plt.figure(figsize=(18, 6))
images: Dict[Any, AxesImage] = {}
PLOTTING_DOMAIN = get_plotting_domain(LOWER_LIMIT, UPPER_LIMIT, PLOTTING_RESOLUTION)


def plot_data(data, subplot, title=""):
    data = data.reshape(PLOTTING_RESOLUTION, PLOTTING_RESOLUTION)

    if subplot in images:
        image = images[subplot]
        image.set_data(data)
        image.set_clim(vmin=data.min(), vmax=data.max())
    else:
        axis = figure.add_subplot(subplot)
        image = axis.imshow(data, cmap=plt.get_cmap(COLOUR_MAP), vmin=data.min(), vmax=data.max(),
                            extent=[LOWER_LIMIT, UPPER_LIMIT, LOWER_LIMIT, UPPER_LIMIT],
                            interpolation='nearest', origin='lower')
        images[subplot] = image

        axis.set_title(title)

    plt.pause(0.01)


def plot_true_function():
    z = true_integrand(PLOTTING_DOMAIN)
    plot_data(z, 133, "True Integrand")


def plot_integrand_posterior(integrand_model: IntegrandModel):
    z = integrand_model.posterior_mean_and_variance(PLOTTING_DOMAIN)[0]
    plot_data(z, 132, "Posterior Mean")


def plotting_callback(func):
    z = np.exp(func(PLOTTING_DOMAIN)[0])
    plot_data(z, 131, "Acquisition Function")


plotting.add_callback("Soft penalised log acquisition function", plotting_callback)
plot_true_function()

# Run algorithm.

BATCHES = 25
BATCH_SIZE = 4

for i in range(BATCHES):
    plot_integrand_posterior(model)
    batch = select_batch(model, BATCH_SIZE, LOCAL_PENALISATION)

    X = np.array(batch)
    Y = true_function(X)
    model.update(X, Y)

    gpy_gp.optimize()

    print("Integral: {}".format(model.integral_mean()))

plot_integrand_posterior(model)
plt.show()

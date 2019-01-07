"""Plot bayesian quadrature on a simple 1D test function."""
from typing import Dict, Any

import GPy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import newaxis

import bayesquad.plotting as plotting
from bayesquad.batch_selection import select_batch, LOCAL_PENALISATION
from bayesquad.gps import WsabiLGP
from bayesquad.priors import Gaussian
from bayesquad.quadrature import IntegrandModel


PLOTTING_DELAY = 0.25
BATCHES = 10
BATCH_SIZE = 3
BATCH_METHOD = LOCAL_PENALISATION


# Set up test function and WSABI-L model.

def true_function(x):
    return ((np.sin(x) + 0.5 * np.cos(3 * x))**2)/((x/2)**2+0.3)


initial_x = np.array([[-3]]).T
initial_y = np.sqrt(2 * true_function(initial_x))

k = GPy.kern.RBF(1, variance=2, lengthscale=2)
lik = GPy.likelihoods.Gaussian(variance=1e-10)

prior = Gaussian(mean=np.array([0]), covariance=np.atleast_2d(2))

gpy_gp = GPy.core.GP(initial_x, initial_y, kernel=k, likelihood=lik)
warped_gp = WsabiLGP(gpy_gp)
model = IntegrandModel(warped_gp, prior)


def true_integrand(x):
    return true_function(x) * prior(x)[:, newaxis]


# Set up plotting.

LOWER_LIMIT = -4
UPPER_LIMIT = 4
PLOTTING_RESOLUTION = 2000


def get_plotting_domain(lower_limit, upper_limit, resolution):
    x = np.linspace(lower_limit, upper_limit, resolution)
    return np.atleast_2d(x).T


figure = plt.figure(figsize=(18, 8))

axes: Dict[Any, Axes] = {
    "left": figure.add_subplot(121),
    "right": figure.add_subplot(122)
}

for subplot in axes:
    axes[subplot].set_ylim(-0.1, 1)

PLOTTING_DOMAIN = get_plotting_domain(LOWER_LIMIT, UPPER_LIMIT, PLOTTING_RESOLUTION)


def plot_data(data, subplot, title="", color=None):
    axis = axes[subplot]
    axis.set_title(title)

    return axis.plot(PLOTTING_DOMAIN, data, color=color)


plot_elements = {
    "posterior_mean": None,
    "uncertainty_window": None,
    "uncertainty_upper_bound": None,
    "uncertainty_lower_bound": None,
    "evaluated_points": None
}


def plot_true_function():
    z = true_integrand(PLOTTING_DOMAIN)
    plot_data(z, "right", "True Integrand")


def compute_and_plot_integrand_posterior(integrand_model: IntegrandModel):
    global posterior_mean

    z = integrand_model.posterior_mean_and_variance(PLOTTING_DOMAIN)[0].T
    posterior_mean = z.T
    plot_elements["posterior_mean"], = plot_data(z, "left", title="Posterior Mean", color="tab:red")

    integral_mean = integrand_model.integral_mean()
    axes["left"].text(
        x=0.5,
        y=0.95,
        s="Integral Estimate: {:.4f}".format(integral_mean),
        verticalalignment="top",
        size=12,
        bbox={
            "facecolor": "white",
            "edgecolor": "black"
        }
    )

    print("Integral Estimate: {}".format(integral_mean))


def plot_uncertainty_window(func):
    variance = np.exp(func(PLOTTING_DOMAIN)[0]).T
    standard_deviation = np.sqrt(variance)

    if plot_elements["uncertainty_window"]:
        plot_elements["uncertainty_window"].remove()
        plot_elements["uncertainty_upper_bound"].remove()
        plot_elements["uncertainty_lower_bound"].remove()

    upper_uncertainty = (posterior_mean + 2 * standard_deviation).squeeze()
    lower_uncertainty = (posterior_mean - 2 * standard_deviation).squeeze()
    domain = PLOTTING_DOMAIN.squeeze()

    plot_elements["uncertainty_window"] = \
        axes["left"].fill_between(domain, lower_uncertainty, upper_uncertainty, color=(.6, .7, 1))

    plot_elements["uncertainty_lower_bound"], = axes["left"].plot(domain, lower_uncertainty, color="tab:blue")
    plot_elements["uncertainty_upper_bound"], = axes["left"].plot(domain, upper_uncertainty, color="tab:blue")

    plt.pause(PLOTTING_DELAY)


plotting.add_callback("Soft penalised log acquisition function", plot_uncertainty_window)
plot_true_function()
plt.pause(PLOTTING_DELAY)


# Run algorithm.

for i in range(BATCHES):
    if plot_elements["posterior_mean"]:
        plot_elements["posterior_mean"].remove()

    compute_and_plot_integrand_posterior(model)
    batch = select_batch(model, BATCH_SIZE, BATCH_METHOD)

    X = np.array(batch)
    Y = true_function(X)
    model.update(X, Y)

    Y = true_integrand(X)

    if plot_elements["evaluated_points"]:
        plot_elements["evaluated_points"].remove()

    plot_elements["evaluated_points"], = axes["left"].plot(X, Y, "xr", markersize=10, markeredgewidth=2)

    plt.pause(PLOTTING_DELAY)

    axes["left"].plot(X, Y, "xg", markersize=5, markeredgewidth=1)

    gpy_gp.optimize()


plot_elements["posterior_mean"].remove()
plot_elements["evaluated_points"].remove()

compute_and_plot_integrand_posterior(model)

select_batch(model, 1, BATCH_METHOD)
plt.show()

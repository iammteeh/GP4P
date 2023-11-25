from jax import numpy as jnp
from jax import vmap
import numpy as np
import torch
from domain.feature_model.feature_modeling import get_feature_model

def analyze_posterior(model, X_test, y_test, features):
    words, opposites, literals, interactions = get_feature_model(X_test, features)
    print(f"words: {words}")
    print(f"interactions: {interactions}")

    word_posterior = model.posterior(words)
    mean_words, variance_words = word_posterior.mixture_mean.numpy(), word_posterior.mixture_variance.numpy()
    mean_words = jnp.mean(mean_words, axis=0)
    variance_words = jnp.median(variance_words, axis=0)

    opposite_posterior = model.posterior(opposites)
    mean_opposites, variance_opposites = opposite_posterior.mixture_mean.numpy(), opposite_posterior.mixture_variance.numpy()
    mean_opposites = jnp.mean(mean_opposites, axis=0)
    variance_opposites = jnp.median(variance_opposites, axis=0)

    literals_posterior = model.posterior(literals)
    mean_literals, variance_literals = literals_posterior.mixture_mean.numpy() , literals_posterior.mixture_variance.numpy()
    mean_literals = jnp.mean(mean_literals, axis=0)
    variance_literals = jnp.median(variance_literals, axis=0)

    interactions_posterior = model.posterior(interactions)
    mean_interactions, variance_interactions = interactions_posterior.mixture_mean.numpy(), interactions_posterior.mixture_variance.numpy()
    mean_interactions = jnp.mean(mean_interactions, axis=0)
    variance_interactions = jnp.median(variance_interactions, axis=0)

    return (mean_opposites, variance_opposites), (mean_interactions, variance_interactions)

def explore_data(data):
    # if data is from get_data()
    print(f"shape of X_train: {data[0].shape}")
    print(f"shape of X_test: {data[1].shape}")
    # if data is any df, numpy array, tensor
    print(f"shape of data: {data.shape}")
    # return all the data
    print(f"{data[:]}")
    # slice for shape (d,)
    print(f"slice of data: {data[0]}")
    print(f"{data[:-1]}") # all but last
    print(f"{data[-1:]}") # last
    # for shape (n, d)
    print(f"shape of data: {data.shape[0]} rows, {data.shape[1]} columns")
    # slice of data
    print(f"slice of data: {data[0:5, 0]}") # first 5 rows of first column
    # convert tensor to numpy
    print(f"numpy: {data.numpy()}")

def analyze_model(model, active_dims):
    p, n, m = model.X.shape[1], model.X.shape[0], len(active_dims)
    num_coefficients = p + m * (m - 1) // 2

    probe = jnp.zeros((2 * p + 2 * m * (m - 1)), p)
    vec = jnp.zeros((num_coefficients, 2 * p + 2 * m * (m - 1)))

def analyze_mixture(model, X_test, y_test):
    # get mixture mean and variance
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        observed_pred = model.likelihood(model(X_test))
        mean = observed_pred.mean
        var = observed_pred.variance
        posterior = model.posterior(X_test)
        print(f"kernel lengthscales: {model.covar_module.base_kernel.lengthscale}")
        print(f"kernel outputscale: {model.covar_module.outputscale}")
        print(posterior.mean.shape)
        print(posterior.variance.shape)
        print(f"Ground truth:     {y_test.squeeze(-1)}")
        print(f"Mixture mean:     {posterior.mean.squeeze(-1)}")
        print(f"Mixture variance: {posterior.variance.squeeze(-1)}")

        # analyze mixture mean and variance

def analyze_dimensions(model, dimension, samples):
    vmap_args = (
        samples["mean"],
        samples["variance"],
        samples["lengthscale"],
        samples["outputscale"],
    )
    vmap_args = vmap(lambda x: x[:, dimension])(vmap_args) # get samples for dimension

    for i, arg in enumerate(vmap_args):
        print(f"sample {i}: {arg}")
    
def analyze_dimensionality(model, samples):
    vmap_args = (
        samples["mean"],
        samples["variance"],
        samples["lengthscale"],
        samples["outputscale"],
    )
    vmap_args = vmap(lambda x: x[:, 0])(vmap_args) # get samples for dimension

    for i, arg in enumerate(vmap_args):
        print(f"sample {i}: {arg}")

def analyze_dimensions_v2(model, samples):
    vmap_args = (
        samples["mean"],
        samples["variance"],
        samples["lengthscale"],
        samples["outputscale"],
    )
    # iterate over dimensions and get samples for pairs of dimensions
    for i in range(vmap_args[0].shape[1]):
        for j in range(i + 1, vmap_args[0].shape[1]):
            vmap_args_ij = vmap(lambda x: x[:, i])(vmap_args)
            vmap_args_ij = vmap(lambda x: x[:, j])(vmap_args_ij)
            for k, arg in enumerate(vmap_args_ij):
                print(f"sample {k}: {arg}")

def analyze_mixtures(bayesian_posterior, dim_tuple):
    """"
    get the mixture mean and variance for a given dimension tuple by adding
    the mean and variance of the individual dimensions and subtracting the
    covariance of the dimensions.
    """
    mean = bayesian_posterior.mixture_mean
    variance = bayesian_posterior.mixture_variance
    for i in dim_tuple:
        mean += bayesian_posterior.mean[:, i]
        variance += bayesian_posterior.variance[:, i]
    for i in range(len(dim_tuple)):
        for j in range(i + 1, len(dim_tuple)):
            mean -= bayesian_posterior.covariance[:, i, j]
            variance -= bayesian_posterior.covariance[:, i, j]
    return mean, variance


def waic(model, likelihood, X, Y):
    model.eval()
    with torch.no_grad():
        output = model(X)
        predictive_mean = output.mean
        predictive_var = output.variance
        error = Y - predictive_mean
        log_likelihoods = -0.5 * torch.log(2 * np.pi * predictive_var) - 0.5 * (error**2) / predictive_var
        lppd = torch.sum(torch.log(torch.mean(torch.exp(log_likelihoods), dim=0)))
        p_waic = torch.sum(torch.var(log_likelihoods, dim=0))
        waic = -2 * (lppd - p_waic)
    return waic.item()

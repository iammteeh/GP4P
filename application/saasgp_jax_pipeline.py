import gpytorch
import torch
import jax.numpy as jnp
import pyro
from adapters.gpytorch.saasgp import SAASGP
from application.init_pipeline import init_pipeline, get_numpy_features
from botorch import fit_gpytorch_mll
from adapters.gpytorch.pyro_model import fit_fully_bayesian_model_nuts
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
from domain.feature_model.feature_modeling import inverse_map
from adapters.gpytorch.sampling import get_initial_points, generate_batch, draw_random_samples, draw_random_x, generate_test_x_from_tensor
from domain.metrics import get_metrics, gaussian_log_likelihood
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import mean_absolute_percentage_error, r2_score, explained_variance_score 
import datetime
from time import time
from matplotlib import pyplot as plt
import pickle

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def locate_invalid_data(data):
    if isinstance(data, torch.Tensor):
        isnan = torch.isnan(data)
        isinf = torch.isinf(data)
    elif isinstance(data, jnp.ndarray):
        isnan = jnp.isnan(data)
        isinf = jnp.isinf(data)
    elif isinstance(data, np.ndarray):
        raise ValueError("Use jax.numpy instead of numpy.")
    
    invalid_data_locs = isnan | isinf
    return invalid_data_locs

def validate_data(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and not torch.isfinite(arg).all():
            print(locate_invalid_data(arg))
            raise ValueError("Data contains NaN or inf values.")
        elif isinstance(arg, jnp.ndarray) and not jnp.isfinite(arg).all():
            print(locate_invalid_data(arg))
            raise ValueError("Data contains NaN or inf values.")
    print(f"data is fine.")


def get_data(get_ds=False):
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA)
    print(f"fit model having {X_train.shape[1]} features: {feature_names}")
    rank = np.linalg.matrix_rank(X_train)

    # slice X_test such that it has the same shape as X_train
    # TODO: this shouldn't be necessary
    if len(X_test) > len(X_train):
        X_test = X_test[:len(X_train)]
        y_test = y_test[:len(X_train)]

    if get_ds:
        return (ds, X_train, X_test, y_train, y_test, feature_names)
    else:
        return (X_train, X_test, y_train, y_test, feature_names)
    
def main():
    GP = "SAASGP"
    # init model
    data = get_data()
    X_train, X_test, y_train, y_test, feature_names = data
    # convert to jax.numpy
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    model = SAASGP()

    # check for NaN / inf
    validate_data(X_train, X_test, y_train, y_test)

    model.fit(X_train, y_train)
    print(model)

# Evaluate model
    #observed_pred = model.likelihood(model(X_test)) # same as p
    #mean = observed_pred.mean
    #var = observed_pred.variance
    mean, variance = model.posterior(X_test)
    print(mean.shape)
    print(variance.shape)
    print(f"Ground truth:     {y_test}")
    print(f"Mixture mean:     {mean}")

#print(waic(model, model.likelihood, X_test, y_test))
#print(gaussian_log_likelihood(model, X_test, y_test))
    # compare predictions to actual Y_test
    y_true = y_test
    y_pred = jnp.mean(mean, axis=0)
    test_rmse = jnp.sqrt(np.mean(jnp.square(y_test - jnp.mean(mean, axis=0))))
    test_ll = -0.5 * jnp.square(y_test - mean) / variance - 0.5 * jnp.log(2.0 * jnp.pi * variance)
    test_ll = jnp.mean(logsumexp(test_ll, axis=0)) - jnp.log(mean.shape[0])
    print("test_rmse: {:.4f}   test_ll: {:.4f}".format(test_rmse, test_ll))
    metrics = {}
    metrics["test_rmse"] = test_rmse
    metrics["test_ll"] = test_ll
    metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred),
    metrics["r2"] = r2_score(y_true, y_pred),
    metrics["explained_variance"] = explained_variance_score(y_true, y_pred),
    #print(waic(model, model.likelihood, X_test, y_test))
    # calculate KLD of posterior and prior
    #dist = jensenshannon(model.posterior(X_test).mean.squeeze(-1).detach().numpy(), model(X_test).mean.detach().numpy(), keepdims=True) # represents the KLD between the posterior and the prior
    #print(f"KLD between posterior and prior: {dist}")
    #print(f"jensen shannon: {jensenshannon(model.posterior(X_test).mean.squeeze(-1).detach().numpy(), test_prior.mean_module.detach().numpy(), keepdims=True)}")
    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"SAASGP_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_ARD={ARD}__{timestamp}"
    #torch.save(model.state_dict(), f"{MODELDIR}/{filename}.pth")
    with open(f"{MODELDIR}/{filename}.pkl", "wb") as f:
        #pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        # first collect all objects that are not part of the model
        model_dict = model.__dict__
        model_dict["X_train"] = X_train
        model_dict["y_train"] = y_train
        model_dict["X_test"] = X_test
        model_dict["y_test"] = y_test
        model.kernel = "matern52" # workaround to save the kernel type as the kernel function has to be outside the class
        model_dict["feature_names"] = feature_names
        pickle.dump(model_dict, f, pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {MODELDIR}/{filename}.pth")
if __name__ == "__main__":
    main()

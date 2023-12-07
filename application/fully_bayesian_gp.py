import gpytorch
import torch
import jax.numpy as jnp
import pyro
from adapters.gpytorch.gp_model import GPRegressionModel, SAASGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, InducingPointKernelAddedLossTerm, VariationalELBO, GammaRobustVariationalELBO
from application.init_pipeline import init_pipeline, get_numpy_features
from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
from adapters.gpytorch.sampling import get_initial_points, generate_batch, draw_random_samples, draw_random_x, generate_test_x_from_tensor
from adapters.gpytorch.evaluation import analyze_posterior, get_posterior_dimension, get_posterior_dimension_v2, decompose_covariance, calculate_conditional_expectation
from adapters.gpytorch.plotting import plot_prior, plot_pairwise_posterior_mean_variances, plot_density, plot_combined_pdf_v2, plot_interaction_pdfs
import numpy as np
import datetime
from time import time
from matplotlib import pyplot as plt


from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def locate_invalid_data(data):
    if isinstance(data, torch.Tensor):
        isnan = torch.isnan(data)
        isinf = torch.isinf(data)
    elif isinstance(data, np.ndarray):
        isnan = np.isnan(data)
        isinf = np.isinf(data)
    
    invalid_data_locs = isnan | isinf
    return invalid_data_locs

def validate_data(*args):
    for arg in args:
        if not torch.isfinite(arg).all():
            print(locate_invalid_data(arg))
            raise ValueError("Data contains NaN or inf values.")
    print(f"data is fine.")


def get_data():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)
    rank = np.linalg.matrix_rank(X_train)

    # transform test data to tensor
    X_test = torch.tensor(X_test).double()
    y_test = torch.tensor(y_test).double()

    return (X_train, X_test, y_train, y_test, feature_names)

def choose_model(model="GP", data=None):
    if not data:
        X_train, X_test, y_train, y_test, feature_names = get_data()
    else:
        X_train, X_test, y_train, y_test, feature_names = data
    
    if model == "GP":
        return GPRegressionModel(X_train, y_train, feature_names, likelihood="gaussian", kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)
    elif model == "SAASGP":
        return SAASGP(X_train, y_train, feature_names)
    else:
        raise ValueError(f"Model {model} not found.")
    
def main():
    GP = "SAASGP"
    # init model
    data = get_data()
    X_train, X_test, y_train, y_test, feature_names = data
    model = choose_model(model=GP, data=data)

    # check for NaN / inf
    validate_data(model.X, X_test, model.y, y_test)

    #plot_prior(model, X_test, y_test)

    # fit
    model.train()
    fit_fully_bayesian_model_nuts(model)

    # Evaluate model
    model.eval()
    model.likelihood.eval()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    #print(waic(model, model.likelihood, X_test, y_test))

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f"{MODELDIR}/GPY_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_ARD={ARD}__{timestamp}.pth")
if __name__ == "__main__":
    main()

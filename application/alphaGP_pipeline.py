import gpytorch
import torch
import pyro
from adapters.gpytorch.gp_model import ExactGP, SAASGP
from gpytorch.likelihoods import GaussianLikelihood, StudentTLikelihood, DirichletClassificationLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, InducingPointKernelAddedLossTerm, VariationalELBO, GammaRobustVariationalELBO, PredictiveLogLikelihood
from application.init_pipeline import init_pipeline, get_numpy_features
from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
import numpy as np
import datetime
from time import time

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
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA)
    print(f"fit model having {X_train.shape[1]} features: {feature_names}")
    rank = np.linalg.matrix_rank(X_train)

    # transform test data to tensor
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return (X_train, X_test, y_train, y_test, feature_names)

def choose_model(model="GP", data=None):
    if not data:
        X_train, X_test, y_train, y_test, feature_names = get_data()
    else:
        X_train, X_test, y_train, y_test, feature_names = data
    
    if model == "GP":
        return ExactGP(X_train, y_train, feature_names, likelihood="gaussian", kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)
    elif model == "SAASGP":
        return SAASGP(X_train, y_train, feature_names)
    else:
        raise ValueError(f"Model {model} not found.")

def main():
    GP = "GP"
    # init model
    data = get_data()
    X_train, X_test, y_train, y_test, feature_names = data
    model = choose_model(model=GP, data=data)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # check for NaN / inf
    validate_data(model.X, X_test, model.y, y_test)

    # fit
    fit_gpytorch_mll(mll, options={"maxiter": 1000, "disp": True})

    # Evaluate model
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        observed_pred = model.likelihood(model(X_test))
        mean = observed_pred.mean
        var = observed_pred.variance
        posterior = model.posterior(X_test)
        print(model.kernel.base_kernel.lengthscale.median())
        print(posterior.mean.shape)
        print(posterior.variance.shape)
        print(f"Ground truth:     {y_test.squeeze(-1)}")
        print(f"Mixture mean:     {posterior.mean.squeeze(-1)}")

    #print(waic(model, model.likelihood, X_test, y_test))

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f"{MODELDIR}/GPY_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_ARD={ARD}__{timestamp}.pth")

if __name__ == "__main__":
    main()

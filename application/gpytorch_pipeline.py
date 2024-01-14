import gpytorch
import torch
from adapters.gpytorch.gp_model import GPRegressionModel
from gpytorch.likelihoods import GaussianLikelihood
from application.init_pipeline import init_pipeline, get_numpy_features
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
from domain.metrics import get_metrics, gaussian_log_likelihood
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

def main():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA)
    print(f"fit model having {X_train.shape[1]} features: {feature_names}")
    rank = np.linalg.matrix_rank(X_train)

    # transform test data to tensor
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    # Define likelihood and model
    model = GPRegressionModel(X_train, y_train, feature_names, likelihood="gaussian", kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)

    # check for NaN / inf
    validate_data(model.X, X_test, model.y, y_test)

    # find optimal hyperparameters
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    start = time()
    for i in range(1000):  # training iterations
        optimizer.zero_grad()
        output = model(model.X)
        loss = -mll(output, model.y)
        loss.backward()

        # track parameters every 10th step
        if i % 10 == 0:
            step_time = time() - start
            print(f"Iteration {i+1}, Loss: {loss.item()}, {i} steps took: {step_time:.2f}s")
            #print(f"Iteration {i+1}, Lengthscale: {model.kernel.base_kernel.lengthscale.item()}, Outputscale: {model.kernel.outputscale.item()}")

        optimizer.step()
    print(f"Training took {time() - start:.2f} seconds.")

    # Evaluate model
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        observed_pred = model.likelihood(model(X_test)) # same as p
        mean = observed_pred.mean
        var = observed_pred.variance
        posterior = model.posterior(X_test)

    #print(waic(model, model.likelihood, X_test, y_test))
    print(gaussian_log_likelihood(model, X_test, y_test))
    metrics = get_metrics(posterior, y_test, posterior.mean, type="GP")
    print(f"metrics: {metrics}")

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f"{MODELDIR}/GPY_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_ARD={ARD}__{timestamp}.pth")

if __name__ == "__main__":
    main()

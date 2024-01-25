import gpytorch
import torch
from application.fully_bayesian_gp import get_data
from adapters.gpytorch.gp_model import MyExactGP, MyApproximateGP
from gpytorch.likelihoods import GaussianLikelihood, DirichletClassificationLikelihood, StudentTLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, InducingPointKernelAddedLossTerm, VariationalELBO, GammaRobustVariationalELBO
from application.init_pipeline import init_pipeline, get_numpy_features
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR, DATA_SLICE_AMOUNT
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

def choose_model(model="exact", data=None):
    if not data:
        X_train, X_test, y_train, y_test, feature_names = get_data()
    if model == "exact":
        model = MyExactGP(X_train, y_train, feature_names, likelihood="gaussian", kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model_params = set(model.hyperparameters())
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': list(model_params)},
            {'params': [p for p in model.likelihood.parameters() if p not in model_params]}, # add only likelihood parameters that are not already in the list
        ], lr=0.01)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif model == "approximate":
        model = MyApproximateGP(X_train, y_train, feature_names, kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)
        optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=y_train.shape[0])
        model_params = set(model.hyperparameters())
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': list(model_params)},
            {'params': [p for p in model.likelihood.parameters() if p not in model_params]}, # add only likelihood parameters that are not already in the list
        ], lr=0.01)

        #mll = VariationalELBO(model.likelihood, model, num_data=len(model.y))
        mll = GammaRobustVariationalELBO(model.likelihood, model, num_data=len(model.y), beta=1.0)
    else:
        raise ValueError("Invalid model type.")
    
    return model, optimizer, mll, hyperparameter_optimizer


def main():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA)
    print(f"fit model having {X_train.shape[1]} features: {feature_names}")
    rank = np.linalg.matrix_rank(X_train)

    # transform test data to tensor
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    # init model
    model, optimizer, mll, hyperparameter_optimizer = choose_model(model="exact")
    # check for NaN / inf
    validate_data(model.X, X_test, model.y, y_test)
    
    model.train()
    model.likelihood.train()
    #
    #prior_dist = gpytorch.priors.MultivariateNormalPrior(torch.zeros(len(model.train_inputs)), torch.eye(len(model.train_inputs)))
    #variational_dist = gpytorch.distributions.MultivariateNormal(torch.zeros(len(model.train_inputs)), torch.eye(len(model.train_inputs)))
    #mll = GammaRobustVariationalELBO(model.likelihood, model, num_data=len(model.y), prior_dist=prior_dist, variational_dist=variational_dist, beta=1.0)
    # find optimal hyperparameters
    
    start = time()
    for i in range(1000):  # training iterations
        optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = model(model.X)
        loss = -mll(output, model.y)
        loss.sum().backward()

        # track parameters every 10th step
        if i % 10 == 0:
            step_time = time() - start
            print(f"Iteration {i+1}, Loss: {loss.sum().item()}, {i} steps took: {step_time:.2f}s")
            #print(f"Iteration {i+1}, Lengthscale: {model.kernel.base_kernel.lengthscale.item()}, Outputscale: {model.kernel.outputscale.item()}")

        optimizer.step()
        hyperparameter_optimizer.step()
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
    torch.save(model.state_dict(), f"{MODELDIR}/GPY_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_{DATA_SLICE_AMOUNT}__{timestamp}.pth")

if __name__ == "__main__":
    main()

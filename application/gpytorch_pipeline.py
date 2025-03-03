import gpytorch
import torch
from domain.gp_model import MyExactGP, MyApproximateGP
from gpytorch.mlls import ExactMarginalLogLikelihood, InducingPointKernelAddedLossTerm, VariationalELBO, GammaRobustVariationalELBO
from application.init_pipeline import init_pipeline, get_data, validate_data
from domain.env import USE_DUMMY_DATA, MODELDIR, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, DATA_SLICE_AMOUNT, SWS, Y, POLY_DEGREE
from domain.scores import get_metrics, gaussian_log_likelihood
import numpy as np
import datetime
from time import time

def choose_model(model="exact", data=None):
    if not data:
        X_train, X_test, y_train, y_test, feature_names = get_data(precision="float32")
    else:
        X_train, X_test, y_train, y_test, feature_names = data
    if model == "exact":
        # check training set size
        print(f"X_train size: {X_train.shape}")
        print(f"y_train size: {y_train.shape}")
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

def fit_gpytorch_mll(model, optimizer, mll, hyperparameter_optimizer, verbose=False):
    print(f"Start training {model.__class__.__name__}...")
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
            print(f"Iteration {i+1}, Loss: {loss.sum().item()}, {i} steps took: {step_time:.2f}s") if verbose else None
            #print(f"Iteration {i+1}, Lengthscale: {model.kernel.base_kernel.lengthscale.item()}, Outputscale: {model.kernel.outputscale.item()}")
        # placeholder for early stopping with threshold on loss delta
        #if i > 0 and abs(loss.sum().item() - prev_loss) < "threshold":
        #    print("Early stopping...")
        #    break
        prev_loss = loss.sum().item()
        optimizer.step()
        hyperparameter_optimizer.step()
    print(f"Training took {time() - start:.2f} seconds. Last loss: {loss.sum().item()}")
    return loss.sum().item()

def main():
    data = get_data()
    X_train, X_test, y_train, y_test, feature_names = data
    # init model
    model, optimizer, mll, hyperparameter_optimizer = choose_model(model="exact", data=data)
    # check for NaN / inf
    validate_data(model.X, X_test, model.y, y_test)
    
    model.train()
    model.likelihood.train()
    #
    #prior_dist = gpytorch.priors.MultivariateNormalPrior(torch.zeros(len(model.train_inputs)), torch.eye(len(model.train_inputs)))
    #variational_dist = gpytorch.distributions.MultivariateNormal(torch.zeros(len(model.train_inputs)), torch.eye(len(model.train_inputs)))
    #mll = GammaRobustVariationalELBO(model.likelihood, model, num_data=len(model.y), prior_dist=prior_dist, variational_dist=variational_dist, beta=1.0)
    # find optimal hyperparameters
    
    fit_gpytorch_mll(model, optimizer, mll, hyperparameter_optimizer, verbose=True)

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
    filename = f"{SWS}_{Y}_exact_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_{DATA_SLICE_AMOUNT}_{timestamp}" if not USE_DUMMY_DATA else f"synthetic_{POLY_DEGREE}_MCMC_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_{DATA_SLICE_AMOUNT}_{timestamp}"
    torch.save(model.state_dict(), f"{MODELDIR}/{filename}.pth")

if __name__ == "__main__":
    main()

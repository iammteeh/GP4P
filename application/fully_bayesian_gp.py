import numpy as np
import torch
from adapters.gpytorch.gp_model import SAASGP
from application.init_pipeline import init_pipeline
from adapters.gpytorch.pyro_model import fit_fully_bayesian_model_nuts
from domain.env import USE_DUMMY_DATA, MODELDIR, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE
from domain.metrics import get_metrics
import datetime


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


def get_data(get_ds=False):
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA)
    print(f"fit model having {X_train.shape[1]} features: {feature_names}")
    rank = np.linalg.matrix_rank(X_train)

    # slice X_test such that it has the same shape as X_train
    # TODO: this shouldn't be necessary
    if len(X_test) > len(X_train):
        X_test = X_test[:len(X_train)]
        y_test = y_test[:len(X_train)]

    # transform test data to tensor
    X_test = torch.tensor(X_test).double()
    y_test = torch.tensor(y_test).double()

    if get_ds:
        return (ds, X_train, X_test, y_train, y_test, feature_names)
    else:
        return (X_train, X_test, y_train, y_test, feature_names)
    
def main():
    GP = "SAASGP"
    # init model
    data = get_data()
    X_train, X_test, y_train, y_test, feature_names = data
    model = SAASGP(X_train, y_train, feature_names, mean_func="constant", kernel_structure=KERNEL_STRUCTURE, kernel_type=KERNEL_TYPE)

    # check for NaN / inf
    validate_data(model.X, X_test, model.y, y_test)

    model.eval()
    print(model)
    #likelihood = GaussianLikelihood()
    #test_prior = likelihood(X_test)
    # instead, we can sample from the prior after training
    test_prior = model.pyro_sample_from_prior()

    # fit
    model.train()
    #TODO: add ENV for sampling parameters
    fit_fully_bayesian_model_nuts(model, jit_compile=True)
    print(model)

    # Evaluate model
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        #observed_pred = model.likelihood(model(X_test)) # same as p
        #mean = observed_pred.mean
        #var = observed_pred.variance
        posterior = model.posterior(X_test)
        print(posterior.mean.shape)
        print(posterior.variance.shape)
        print(f"Ground truth:     {y_test.squeeze(-1)}")
        print(f"Mixture mean:     {posterior.mean.squeeze(-1)}")

    #print(waic(model, model.likelihood, X_test, y_test))
    #print(gaussian_log_likelihood(model, X_test, y_test))
    metrics = get_metrics(posterior, y_test, posterior.mixture_mean.squeeze(), type="GP")
    print(f"metrics: {metrics}")

    #print(waic(model, model.likelihood, X_test, y_test))
    # calculate KLD of posterior and prior
    #dist = jensenshannon(model.posterior(X_test).mean.squeeze(-1).detach().numpy(), model(X_test).mean.detach().numpy(), keepdims=True) # represents the KLD between the posterior and the prior
    #print(f"KLD between posterior and prior: {dist}")
    #print(f"jensen shannon: {jensenshannon(model.posterior(X_test).mean.squeeze(-1).detach().numpy(), test_prior.mean_module.detach().numpy(), keepdims=True)}")
    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"SAASGP_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_{timestamp}"
    torch.save(model.state_dict(), f"{MODELDIR}/{filename}.pth")
    print(f"Model saved to {MODELDIR}/{filename}.pth")
if __name__ == "__main__":
    main()

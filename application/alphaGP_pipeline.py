import numpy as np
import torch
from adapters.gpytorch.gp_model import MyExactGP, SAASGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from adapters.preprocessing import prepare_dataset
from application.init_pipeline import init_pipeline, yield_experiments
from application.gpytorch_pipeline import fit_gpytorch_mll
from adapters.gpytorch.pyro_model import fit_fully_bayesian_model_nuts
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
import datetime
from time import time

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

def get_data(training_sizes=[100]):
    ds = prepare_dataset()
    # fetch data with the desired training size
    data = yield_experiments(ds, training_size=training_sizes)
    try:
        while True:
            feature_names, X_train, X_test, y_train, y_test = next(data)

            yield (X_train, X_test, y_train, y_test, feature_names)
    except StopIteration:
        print("Finished fetching data.")
        return

def choose_model(inference="MCMC", mean_func=MEAN_FUNC, kernel_type=KERNEL_TYPE, kernel_structure=KERNEL_STRUCTURE, data=None):
    if data:
        X_train, X_test, y_train, y_test, feature_names = data
    else:
        raise ValueError("Data is required.")
    if inference == "exact":
        model = MyExactGP(X_train, y_train, feature_names, likelihood="gaussian", kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model_params = set(model.hyperparameters())
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': list(model_params)},
            {'params': [p for p in model.likelihood.parameters() if p not in model_params]}, # add only likelihood parameters that are not already in the list
        ], lr=0.01)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return model, optimizer, mll, hyperparameter_optimizer
    elif inference == "MCMC":
        model = SAASGP(X_train, y_train, feature_names, mean_func="constant", kernel_structure=KERNEL_STRUCTURE, kernel_type=KERNEL_TYPE)
        return model, (None)
    else:
        raise ValueError(f"Model {model} not found.")

def main():
    training_sizes = [50]#, 100, 200, 500, 1000]
    kernel_types = ["RBF"]#, "matern52", "spectral_mixture", "RFF"]
    kernel_structure = ["simple"]#, "additive"]
    inference_methods = ["exact", "MCMC"]
    i = 0
    for data in get_data(training_sizes=training_sizes):
        training_size = training_sizes[i]
        for struct in kernel_structure:
            for kernel_type in kernel_types:
                for inference_type in inference_methods:
                    X_train, X_test, y_train, y_test, feature_names = data
                    print(f"fit model having {X_train.shape[1]} features: {feature_names}")
                    model, *context = choose_model(inference=inference_type, data=data)
                    # check for NaN / inf
                    validate_data(model.X, X_test, model.y, y_test)

                    # Fit model
                    start = time()
                    if inference_type == "exact":
                        optimizer, mll, hyperparameter_optimizer = context
                        fit_gpytorch_mll(model, optimizer, mll, hyperparameter_optimizer)
                    elif inference_type == "MCMC":
                        fit_fully_bayesian_model_nuts(model, jit_compile=True)
                    end = time() - start
                    print(f"Training time: {end:.2f}s")

                    # set evaluation mode
                    model.eval()
                    model.likelihood.eval()
                    # Save model
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    torch.save(model.state_dict(), f"{MODELDIR}/{inference_type}_{kernel_type}_{kernel_structure}_{training_size}_{timestamp}.pth")
                    i += 1
if __name__ == "__main__":
    main()

import numpy as np
import torch
from adapters.gpytorch.gp_model import MyExactGP, SAASGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from adapters.preprocessing import prepare_dataset
from application.init_pipeline import init_pipeline, yield_experiments
from application.gpytorch_pipeline import fit_gpytorch_mll
from adapters.gpytorch.pyro_model import fit_fully_bayesian_model_nuts
from adapters.model_store import init_store, update_store
from domain.env import USE_DUMMY_DATA, MODELDIR, SWS, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
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

def get_data(training_sizes=[100], use_synthetic_data=False, sws=SWS):
    ds = prepare_dataset(dummy_data=use_synthetic_data, sws=sws)
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
    training_sizes = [50, 100, 200, 500, 1000]
    kernel_types = ["poly2", "poly3", "poly4", "piecewise_polynomial", "RBF", "matern32", "matern52", "RFF", "spectral_mixture"]
    kernel_structure = ["simple", "additive"]
    inference_methods = ["exact", "MCMC"]
    init_store(store_path=f"{RESULTS_DIR}/modelstorage_{timestamp}_TEST.csv")
    i = 0
    total_running_time_start = time()
    for data in get_data(training_sizes=training_sizes, use_synthetic_data=USE_DUMMY_DATA, sws=SWS):
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
                        loss = fit_gpytorch_mll(model, optimizer, mll, hyperparameter_optimizer)
                    elif inference_type == "MCMC":
                        trace = fit_fully_bayesian_model_nuts(model, jit_compile=True)
                    end = time() - start
                    print(f"Training time: {end:.2f}s")

                    # set evaluation mode
                    model.eval()
                    model.likelihood.eval()
                    # Save model
                    filename = f"{inference_type}_{kernel_type}_{kernel_structure}_{training_size}_{timestamp}"
                    torch.save(model.state_dict(), f"{MODELDIR}/{filename}.pth")
                    update_store(
                        index=f"{filename}",
                        last_loss=loss if inference_type == "exact" else trace,
                        loss_curve="loss_curve",
                        model_scores="model_scores",
                        timestamp=timestamp
                    )

                    
                    i += 1
    total_running_time_end = time() - total_running_time_start
    print(f"Total running time: {total_running_time_end:.2f}s")
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    main()

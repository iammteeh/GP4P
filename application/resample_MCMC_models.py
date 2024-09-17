import numpy as np
import torch
from adapters.gpytorch.gp_model import MyExactGP, SAASGP, SAASGPJAX
from gpytorch.mlls import ExactMarginalLogLikelihood
from adapters.preprocessing import prepare_dataset
from application.init_pipeline import init_pipeline, yield_experiments
from application.gpytorch_pipeline import fit_gpytorch_mll
from adapters.pyro.pyro_model import fit_fully_bayesian_model_nuts
from adapters.pyro.pyro_model_jax import fit_fully_bayesian_model_nuts as fit_fully_bayesian_model_nuts_jax # for additive models
from adapters.model_store import init_store, update_store
from domain.env import MODELDIR, RESULTS_DIR
from domain.metrics import get_metrics, get_BIC
import datetime
from time import time
import warnings
from builtins import UserWarning
from botorch.models.utils.assorted import InputDataWarning
from torch.linalg import LinAlgError

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)

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

def get_data(training_sizes=[100], use_synthetic_data=False, sws="LLVM_energy", y_type="performance"):
    print(f"Fetching data from {sws if not use_synthetic_data else 'synthetic ds'} with training sizes: {training_sizes}")
    ds = prepare_dataset(dummy_data=use_synthetic_data, sws=sws, y_type=y_type)
    # fetch data with the desired training size
    data = yield_experiments(ds, training_size=training_sizes)
    try:
        while True:
            feature_names, X_train, X_test, y_train, y_test = next(data)

            yield (X_train, X_test, y_train, y_test, feature_names)
    except StopIteration:
        print("Finished fetching data.")
        return

def choose_model(inference="MCMC", mean_func="linear", kernel_type="matern52", kernel_structure="simple", data=None):
    if data:
        X_train, X_test, y_train, y_test, feature_names = data
    else:
        raise ValueError("Data is required.")
    if inference == "exact":
        model = MyExactGP(X_train, y_train, feature_names, likelihood="gaussian", kernel=kernel_type, mean_func=mean_func, structure=kernel_structure)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model_params = set(model.hyperparameters())
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': list(model_params)},
            {'params': [p for p in model.likelihood.parameters() if p not in model_params]}, # add only likelihood parameters that are not already in the list
        ], lr=0.01)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return model, optimizer, mll, hyperparameter_optimizer
    elif inference == "MCMC" and kernel_structure == "simple":
        model = SAASGP(X_train, y_train, feature_names, mean_func="constant", kernel_structure=kernel_structure, kernel_type=kernel_type)
        return model, (None)
    elif inference == "MCMC" and kernel_structure == "additive":
        model = SAASGPJAX(X_train, y_train, feature_names, mean_func="constant", kernel_structure=kernel_structure, kernel_type=kernel_type)
        return model, (None)
    else:
        raise ValueError(f"Model {model} not found.")

def main(timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), sws=None, y="performance"):
    use_dummy_data = True if not sws else False
    training_sizes = [20, 50, 100, 250, 500]
    kernel_types = ["poly2", "poly3", "poly4", "RBF", "matern32", "matern52"]
    kernel_structures = ["simple", "additive"] # run additive MCMC models using JAX only
    inference_methods = ["MCMC"]
    STORAGE_PATH = f"{RESULTS_DIR}/modelstorage_{sws}_{y}.json" if not use_dummy_data else f"{RESULTS_DIR}/modelstorage_synthetic_p{POLY_DEGREE}.json"
    init_store(store_path=f"{STORAGE_PATH}")
    i = 0
    total_running_time_start = time()
    for data in get_data(training_sizes=training_sizes, use_synthetic_data=use_dummy_data, sws=sws, y_type=y):
        training_size = training_sizes[i]
        for kernel_structure in kernel_structures:
            for kernel_type in kernel_types:
                for inference_type in inference_methods:
                    X_train, X_test, y_train, y_test, feature_names = data
                    print(f"fit model having {X_train.shape[1]} features: {feature_names}\ninference: {inference_type}, kernel: {kernel_type}, structure: {kernel_structure}")
                    model, *context = choose_model(inference=inference_type, kernel_type=kernel_type, kernel_structure=kernel_structure, data=data)
                    # check for NaN / inf
                    validate_data(model.X, X_test, model.y, y_test)

                    # Fit model
                    start = time()
                    try:
                        if inference_type == "exact":
                            optimizer, mll, hyperparameter_optimizer = context
                            loss = fit_gpytorch_mll(model, optimizer, mll, hyperparameter_optimizer)
                        elif inference_type == "MCMC" and kernel_structure == "simple":
                            trace = fit_fully_bayesian_model_nuts(model, jit_compile=True)
                        elif inference_type == "MCMC" and kernel_structure == "additive":
                            trace = fit_fully_bayesian_model_nuts_jax(model)
                        end = time() - start
                        print(f"Training time: {end:.2f}s")

                        # set evaluation mode
                        model.eval()
                        model.likelihood.eval()
                        with torch.no_grad():
                            posterior = model.posterior(X_test)
                            if inference_type == "exact":
                                mean = posterior.mean
                            elif inference_type == "MCMC":
                                mean = posterior.mixture_mean
                        metrics = get_metrics(posterior, y_test, mean, type="GP")
                        print(f"metrics: {metrics}")
                        log_likelihood = model.likelihood.log_marginal(y_test, model(X_test)) if inference_type == "exact" else model.likelihood.log_marginal(y_test, model(X_test.double()))
                        log_likelihood = log_likelihood.sum().detach().numpy()
                        BIC = get_BIC(log_likelihood, y_test.shape[0], model.num_outputs)
                        # Save model
                        filename = f"{sws}_{y}_{inference_type}_{kernel_type}_{kernel_structure}_{training_size}_{timestamp}" if not use_dummy_data else f"synthetic_{POLY_DEGREE}_{inference_type}_{kernel_type}_{kernel_structure}_{training_size}_{timestamp}"
                        torch.save(model.state_dict(), f"{MODELDIR}/{filename}.pth")
                        
                        modeldata = {
                            "filename": f"{filename}.pth",
                            "model": {
                                "dataset": sws if not use_dummy_data else "synthetic_p{POLY_DEGREE}",
                                "benchmark": y if not use_dummy_data else "random normal",
                                "kernel_type": kernel_type,
                                "kernel_structure": kernel_structure,
                                "inference_type": inference_type,
                                "training_size": training_size,
                                "timestamp": timestamp,
                            },
                            "scores": {
                                "RMSE": metrics["RMSE"],
                                "BIC": BIC,
                                "MAPE": metrics["MAPE"],
                                "ESS": metrics["explained_variance"],
                                "last_loss": loss if inference_type == "exact" else "trace.tolist()", # TODO: fix this
                                "loss_curve": "loss_curve",
                                "training_time": end,
                            },
                        }
                        update_store(
                            index=f"{filename}",
                            modeldata=modeldata,
                            store_path=f"{STORAGE_PATH}"
                        )
                    except LinAlgError or ValueError as e:
                        print(f"Model having {X_train.shape[1]} features: {feature_names}\ninference: {inference_type}, kernel: {kernel_type}, structure: {kernel_structure} failed to converge. Skipping...")
                        pass  
        i += 1
    total_running_time_end = time() - total_running_time_start
    print(f"Total running time: {total_running_time_end:.2f}s")

def run_pipeline(use_dummy_data=False, swss=["LLVM_energy"]):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if use_dummy_data:
        for degree in [2,3,4]:
            global POLY_DEGREE
            POLY_DEGREE = degree
            print(f"Running pipeline with dummy data for polynomial degree: {POLY_DEGREE}")
            main(timestamp=timestamp)
    elif swss:
        for sws,y in swss.items():
                print(f"Running pipeline with data from {sws}, estimating '{y}'...")
                main(timestamp=timestamp, sws=sws, y=y)
    else:
        raise ValueError("No data source provided.")

if __name__ == "__main__":
    swss = {
        "x264_energy": "fixed-energy",
        "Apache_energy_large": "performance",
        "HSQLDB_energy": "fixed-energy",
        "HSQLDB_pervolution_energy_bin": "performance",
        "LLVM_energy": "performance",
        "PostgreSQL_pervolution_energy_bin": "performance",
        "VP8_pervolution_energy_bin": "performance",
    }
    run_pipeline(use_dummy_data=True)
    run_pipeline(swss=swss)
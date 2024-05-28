import numpy as np
import torch
from adapters.gpytorch.gp_model import MyExactGP, SAASGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from adapters.preprocessing import prepare_dataset
from application.init_pipeline import init_pipeline, yield_experiments
from application.gpytorch_pipeline import fit_gpytorch_mll
from adapters.gpytorch.pyro_model import fit_fully_bayesian_model_nuts
from adapters.model_store import init_store, update_store
from domain.env import USE_DUMMY_DATA, MODELDIR, SWS, Y, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, RESULTS_DIR
from domain.metrics import get_metrics
import datetime
from time import time
import warnings
from builtins import UserWarning
from botorch.models.utils.assorted import InputDataWarning

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

def get_data(training_sizes=[100], use_synthetic_data=False, sws=SWS):
    print(f"Fetching data from {SWS if not use_synthetic_data else 'synthetic ds'} with training sizes: {training_sizes}")
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

def main(timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
    training_sizes = [20, 50, 100, 250, 500, 1000]
    kernel_types = ["poly2", "poly3", "poly4", "piecewise_polynomial", "RBF", "matern32", "matern52", "RFF", "spectral_mixture"]
    kernel_structures = ["simple", "additive"]
    inference_methods = ["exact", "MCMC"]
    STORAGE_PATH = f"{RESULTS_DIR}/modelstorage_{SWS}_{timestamp}_TEST.csv"
    init_store(store_path=f"{STORAGE_PATH}")
    i = 0
    total_running_time_start = time()
    for data in get_data(training_sizes=training_sizes, use_synthetic_data=USE_DUMMY_DATA, sws=SWS):
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
                    with torch.no_grad():
                        posterior = model.posterior(X_test)
                        if inference_type == "exact":
                            mean = posterior.mean
                        elif inference_type == "MCMC":
                            mean = posterior.mixture_mean
                    metrics = get_metrics(posterior, y_test, mean, type="GP")
                    print(f"metrics: {metrics}")
                    # Save model
                    filename = f"{SWS}_{Y}_{inference_type}_{kernel_type}_{kernel_structure}_{training_size}_{timestamp}" if not USE_DUMMY_DATA else f"synthetic_{POLY_DEGREE}_{inference_type}_{kernel_type}_{kernel_structure}_{training_size}_{timestamp}"
                    torch.save(model.state_dict(), f"{MODELDIR}/{filename}.pth")
                    update_store(
                        index=f"{filename}",
                        filename=f"{filename}.txt",
                        last_loss=loss if inference_type == "exact" else "trace.tolist()", # TODO: fix this
                        loss_curve="loss_curve",
                        RMSE=metrics["RMSE"].tolist(),
                        MAPE=metrics["MAPE"].tolist(),
                        ESS=metrics["explained_variance"].tolist(),
                        timestamp=timestamp,
                        training_time=end,
                        training_size=training_size,
                        store_path=f"{STORAGE_PATH}"
                    )

                    
                    i += 1
    total_running_time_end = time() - total_running_time_start
    print(f"Total running time: {total_running_time_end:.2f}s")

def run_pipeline(use_dummy_data=USE_DUMMY_DATA, swss=["LLVM_energy"]):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if use_dummy_data:
        for degree in [2,3,4]:
            global POLY_DEGREE
            POLY_DEGREE = degree
            print(f"Running pipeline with dummy data for polynomial degree: {POLY_DEGREE}")
            main(timestamp=timestamp)
    elif swss:
        for sws,y in swss.items():
                global SWS, Y
                SWS = sws
                Y = y
                print(f"Running pipeline with data from {SWS}, estimating '{Y}'...")
                main(timestamp=timestamp)
    else:
        raise ValueError("No data source provided.")

if __name__ == "__main__":
    swss = {
        "x264_energy": "fixed-energy",
        "Apache_energy_large": "performance",
        "HSQLDB_energy": "fixed-energy",
        "HSQLDB_pervolution_energy_bin": "performance",
        "LLVM_energy": "fixed-energy",
        "PostgreSQL_pervolution_energy_bin": "performance",
        "VP8_pervolution_energy_bin": "performance",
    }
    run_pipeline(use_dummy_data=USE_DUMMY_DATA, swss=swss)
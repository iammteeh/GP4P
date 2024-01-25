import os
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR, DATA_SLICE_AMOUNT
from application.init_pipeline import init_pipeline, get_numpy_features

import torch
from torch.quasirandom import SobolEngine
import numpy as np
from itertools import combinations

from botorch.settings import debug
from botorch.models.transforms.input import Normalize
from application.fully_bayesian_gp import get_data
from adapters.botorch.model_prep import init_model
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin, Hartmann, SyntheticTestFunction, Ackley, Beale, Bukin, DixonPrice, DropWave, EggHolder, Griewank, Levy, Michalewicz, Rastrigin, Rosenbrock, SixHumpCamel, StyblinskiTang, ThreeHumpCamel
from adapters.botorch.utility import LinearPredictionUtility
from botorch.generation import gen_candidates_torch, get_best_candidates

from adapters.botorch.data_generation import get_X_y, for_X_get_y, update_random_observations, generate_candidates, get_observations
from adapters.botorch.aquisitions import get_acq

from domain.metrics import get_metrics

"""
mainly from https://botorch.org/tutorials/closed_loop_botorch_only
"""
debug._set_state(True)
tkwargs = {
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

WARMUP_STEPS = 256
NUM_SAMPLES = 128
THINNING = 16

# GET DATA
# TODO: merge data import
ds, X_train, X_test, y_train, y_test, feature_names = get_X_y()
# get some properties of the input matrix
dim = X_train.shape[-1]
rank = np.linalg.matrix_rank(X_train)
#train_con = torch.tensor(np.zeros((X_train.shape[0], 1))).float()
# make y 2D
y_train = y_train.unsqueeze(-1)

# Evaluation budget
# TODO: find good values 
N_INIT = 10
N_TRIALS = 3
N_ITERATIONS = 8
BATCH_SIZE = 50
NUM_RESTARTS = 10
RAW_SAMPLES = DATA_SLICE_AMOUNT
NOISE_SE = 0.1
#N_BATCH = 20 if not SMOKE_TEST else 2
#MC_SAMPLES = 256 if not SMOKE_TEST else 32
print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")

#TODO: add a function that returns the best observation from the training data
best_observed_all, best_random_all = [], []
# run N_TRIALS rounds of BayesOpt after the initial random batch
for trial in range(1, N_TRIALS + 1):

    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    best_observed, best_random = [], []
    
    # INIT MODEL
    # TODO: start with a random model
    X_train = X_train[:N_INIT]
    y_train = y_train[:N_INIT]
    MODEL = "FixedNoise"
    gp, optimizer, mll, hyperparameter_optimizer, trainer = init_model(model=MODEL, data=(X_train, X_test, y_train, y_test, feature_names))
    best_observed.append(y_train.min().item())
    best_random.append(y_train.min().item())
                       
    # run N_ITERATIONS rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_ITERATIONS + 1):
    
        #trainer = fit_gpytorch_mll(mll, options={"maxiter": 1000, "disp": True})
        if MODEL != "SAASGP":
            trainer(mll, options={"maxiter": 1000, "disp": True})
        else:
            trainer(
                gp,
                warmup_steps=WARMUP_STEPS,
                num_samples=NUM_SAMPLES,
                thinning=THINNING,
            )

        # INIT ACQUISITION FUNCTION
        # meta parameters for the optimization
        ACQ = "PI" # possible values: "EI", "PI", "UCB"
        acq = get_acq(gp, y_train.min(), acq="PI")
        bounds = torch.stack([torch.zeros(X_train.shape[-1]), torch.ones(X_train.shape[-1])])
        outcome_constraint = lambda X: X.sum(dim=-1) - 3
        # Optimize the acquisition function and get new observation
        candidates = generate_candidates(num_samples=BATCH_SIZE, mode="acqf", acq_func=acq, bounds=bounds, batch_size=BATCH_SIZE, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
        new_x, new_obj, new_con = get_observations(candidates, NOISE_SE ,outcome_constraint)

        # Update training data
        X_train = torch.cat([X_train, new_x])
        y_train = torch.cat([y_train, new_obj])
        #train_con = torch.cat([train_con, new_con])

        # update progress
        #best_value = train_obj.min().item()
        #best_value = for_X_get_y(ds, train_x, feature_names).min().item()

        # Re-fit the model with updated data
        print("Re-fitting the model on the updated dataset")
        gp, optimizer, mll, hyperparameter_optimizer, trainer = init_model(model=MODEL, data=(X_train, X_test, y_train, y_test, feature_names), state_dict=gp.state_dict())

# Evaluate model
gp.eval()
gp.likelihood.eval()
with torch.no_grad():
    observed_pred = gp.likelihood(gp(X_test)) # same as p
    mean = observed_pred.mean
    var = observed_pred.variance
    posterior = gp.posterior(X_test)

print(gp.covar_module.base_kernel.lengthscale.median())
print(posterior.mean.shape)
print(posterior.variance.shape)
print(f"Ground truth:     {y_test.squeeze(-1)}")
print(f"Mixture mean:     {posterior.mean.squeeze(-1)}")

#print(waic(model, model.likelihood, X_test, y_test))
metrics = get_metrics(posterior, y_test, posterior.mean, type="GP")
print(f"metrics: {metrics}")
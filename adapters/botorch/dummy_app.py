import os
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
from application.init_pipeline import init_pipeline, get_numpy_features

import torch
from torch.quasirandom import SobolEngine
import numpy as np
from itertools import combinations

from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.fully_bayesian import SaasPyroModel, SaasFullyBayesianSingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from application.fully_bayesian_gp import get_data
from adapters.botorch.model_prep import init_model
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin, Hartmann, SyntheticTestFunction, Ackley, Beale, Bukin, DixonPrice, DropWave, EggHolder, Griewank, Levy, Michalewicz, Rastrigin, Rosenbrock, SixHumpCamel, StyblinskiTang, ThreeHumpCamel
from adapters.botorch.utility import LinearPredictionUtility

from adapters.botorch.data_generation import get_X_y, for_X_get_y, update_random_observations
from adapters.botorch.aquisitions import get_acq, optimize_acqf_and_get_observation

from domain.metrics import get_metrics

SMOKE_TEST = os.environ.get("SMOKE_TEST")
tkwargs = {
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

WARMUP_STEPS = 256 if not SMOKE_TEST else 32
NUM_SAMPLES = 128 if not SMOKE_TEST else 16
THINNING = 16 if not SMOKE_TEST else 4

# GET DATA
ds, X_train, X_test, y_train, y_test, feature_names = get_X_y()
# get some properties of the input matrix
dim = X_train.shape[-1]
rank = np.linalg.matrix_rank(X_train)
#train_con = torch.tensor(np.zeros((X_train.shape[0], 1))).float()
# make y 2D
y_train = y_train.unsqueeze(-1)

# Evaluation budget
N_INIT = 10
N_ITERATIONS = 8 if not SMOKE_TEST else 1
BATCH_SIZE = 5 if not SMOKE_TEST else 1
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
NOISE_SE = 0.1
N_TRIALS = 3 if not SMOKE_TEST else 2
N_BATCH = 20 if not SMOKE_TEST else 2
MC_SAMPLES = 256 if not SMOKE_TEST else 32
print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")

best_observed_all, best_random_all = [], []
# run N_TRIALS rounds of BayesOpt after the initial random batch
for trial in range(1, N_TRIALS + 1):

    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    best_observed, best_random = [], []
    
    # INIT MODEL
    gp, optimizer, mll, hyperparameter_optimizer = init_model(model="FixedNoise", data=(X_train, X_test, y_train, y_test, feature_names))
    best_observed.append(y_train.min().item())
    best_random.append(y_train.min().item())
                       
    # run N_ITERATIONS rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_ITERATIONS + 1):
    
        trainer = fit_gpytorch_mll(mll, options={"maxiter": 1000, "disp": True})


        # INIT ACQUISITION FUNCTION
        # meta parameters for the optimization
        ACQ = "PI" # possible values: "EI", "PI", "UCB"

        acq = get_acq(gp, y_train.min(), acq="PI")
        # Optimize the acquisition function and get new observation
        bounds = torch.stack([torch.zeros(X_train.shape[-1]), torch.ones(X_train.shape[-1])])
        outcome_constraint = lambda X: X.sum(dim=-1) - 3
        #new_x, new_obj = optimize_acqf_and_get_observation(acq, dim, bounds, outcome_constraint, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, NOISE_SE)
        candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
            )
        new_x = candidates.detach()
        print(f"new_x: {new_x} (shape: {new_x.shape}) with type {type(new_x)}")
        new_obj = for_X_get_y(ds, new_x, feature_names)

        # Update training data
        train_x = torch.cat([X_train, new_x])
        train_obj = torch.cat([y_train, new_obj])
        #train_con = torch.cat([train_con, new_con])

        # update progress
        #best_value = train_obj.min().item()
        #best_value = for_X_get_y(ds, train_x, feature_names).min().item()

        # Re-fit the model with updated data
        print("Re-fitting the model on the updated dataset")

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
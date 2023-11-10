import os
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, LIKELIHOOD ,ARD, RESULTS_DIR
from application.init_pipeline import init_pipeline, get_numpy_features

import torch
from torch.quasirandom import SobolEngine
import numpy as np

from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.models.fully_bayesian import SaasPyroModel, SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin, Hartmann, SyntheticTestFunction, Ackley, Beale, Bukin, Colville, CosineMixture, DixonPrice, DropWave, EggHolder, Exponential, GoldsteinPrice, Griewank, Levy, Michalewicz, Perm, Rastrigin, Rosenbrock, Schaffer, SixHumpCamel, StyblinskiTang, ThreeHumpCamel, XinSheYang, Zakharov

SMOKE_TEST = os.environ.get("SMOKE_TEST")
tkwargs = {
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

WARMUP_STEPS = 256 if not SMOKE_TEST else 32
NUM_SAMPLES = 128 if not SMOKE_TEST else 16
THINNING = 16 if not SMOKE_TEST else 4

ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)
rank = np.linalg.matrix_rank(X_train)

# transform test data to tensor
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

gp = SaasFullyBayesianSingleTaskGP(X_train, y_train, rank=rank, ard_num_dims=rank, tkwargs=tkwargs)
fit_fully_bayesian_model_nuts(
    gp,
    warmup_steps=WARMUP_STEPS,
    num_samples=NUM_SAMPLES,
    thinning=THINNING,
)

with torch.no_grad():
    posterior = gp.posterior(X_test)
print(gp.median_lengthscale.detach())
print(posterior.mean.shape)
print(posterior.variance.shape)
print(f"Ground truth:     {y_test.squeeze(-1)}")
print(f"Mixture mean:     {posterior.mixture_mean.squeeze(-1)}")

# Evaluation budget
N_INIT = 10
N_ITERATIONS = 8 if not SMOKE_TEST else 1
BATCH_SIZE = 5 if not SMOKE_TEST else 1
print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")

# Define the acquisition function
def qExpectedImprovement(model, best_f):
    qEI = qExpectedImprovement(model, best_f=best_f)
    return qEI

f = Branin(negate=True).to(**tkwargs)

# Define the optimization routine
def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and observation."""
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=256,
    )
    # observe new values
    new_x = candidate.detach()
    new_y = f(new_x).detach()
    return new_x, new_y
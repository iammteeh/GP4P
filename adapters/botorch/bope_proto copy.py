import os
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, LIKELIHOOD ,ARD, RESULTS_DIR
from application.init_pipeline import init_pipeline, get_numpy_features

import torch
from torch.quasirandom import SobolEngine
import numpy as np
from itertools import combinations

from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.models.fully_bayesian import SaasPyroModel, SaasFullyBayesianSingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin, Hartmann, SyntheticTestFunction, Ackley, Beale, Bukin, DixonPrice, DropWave, EggHolder, Griewank, Levy, Michalewicz, Rastrigin, Rosenbrock, SixHumpCamel, StyblinskiTang, ThreeHumpCamel
from adapters.botorch.utility import LinearPredictionUtility

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



train_comp = generate_comparisons(y_train, 100 if not SMOKE_TEST else 10)
X, comps = make_new_data(X_train, train_comp, 10)

# transform data to tensor
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

gp = PairwiseGP(X_train, train_comp, input_transform=Normalize(d=X_train.shape[-1])) # X_train.shape[-1] is the rank of the matrix
mll = PairwiseLaplaceMarginalLogLikelihood(gp.likelihood, gp)
mll = fit_gpytorch_mll(mll, options={"maxiter": 1000, "disp": True})

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

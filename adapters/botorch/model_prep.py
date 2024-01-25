from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
from application.fully_bayesian_gp import get_data
from adapters.gpytorch.gp_model import MyExactGP, SAASGP
import torch, gpytorch, botorch
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll

def init_model(model="exact", data=None):
    if not data:
        X_train, X_test, y_train, y_test, feature_names = get_data()
    else:
        X_train, X_test, y_train, y_test, feature_names = data
    if model == "exact":
        model = MyExactGP(X_train, y_train, feature_names, likelihood="gaussian", kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model_params = set(model.hyperparameters())
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': list(model_params)},
            {'params': [p for p in model.likelihood.parameters() if p not in model_params]}, # add only likelihood parameters that are not already in the list
        ], lr=0.01)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif model == "FixedNoise":
        model = botorch.models.FixedNoiseGP(X_train, y_train, train_Yvar=torch.full_like(y_train, 1e-6), input_transform=Normalize(d=X_train.shape[-1])) # X_train.shape[-1] is the rank of the matrix
        #mll = SumMarginalLogLikelihood(model.likelihood, model)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model_params = set(model.hyperparameters())
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': list(model_params)},
            {'params': [p for p in model.likelihood.parameters() if p not in model_params]}, # add only likelihood parameters that are not already in the list
        ], lr=0.01)
        trainer = fit_gpytorch_mll

    elif model == "SAASGP":
        model = SAASGP(X_train, y_train, feature_names)
        mll = None
        optimizer = None
        hyperparameter_optimizer = None
        trainer = fit_fully_bayesian_model_nuts

    else:
        raise ValueError("Invalid model type.")
    
    return model, optimizer, mll, hyperparameter_optimizer, trainer
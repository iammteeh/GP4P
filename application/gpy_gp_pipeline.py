import GPy as gp
from GPy.examples import regression, classification, dimensionality_reduction
from GPy.core.parameterization.priors import Gaussian, Gamma, StudentT
from adapters.pymc.prior_construction import Priors
from GPy.likelihoods import Gaussian, StudentT, Gamma, LogLogistic, MixedNoise, Poisson, Weibull
from GPy.kern import Hierarchical, RBF, Coregionalize, Linear, Matern52
from GPyOpt.acquisitions import AcquisitionEI_MCMC #HierarchicalExpectedImprovement
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
import numpy as np
from domain.feature_model.feature_modeling import additive_kernel_permutation
from adapters.gpy.gpy_prior_construction import GPy_Prior
from application.init_pipeline import init_pipeline, get_numpy_features
from adapters.gpy.util import save_model, load_model
from GPyOpt.models.gpmodel import GPModel
from GPyOpt.objective_examples.experiments2d import branin
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.core.task.space import Design_space, bounds_to_space
import datetime

def waic(model, X, Y):
    mean, var = model.predict(X)
    error = Y - mean
    log_likelihoods = -0.5 * np.log(2 * np.pi * var) - 0.5 * (error**2) / var
    lppd = np.sum(np.log(np.mean(np.exp(log_likelihoods), axis=0)))
    p_waic = np.sum(np.var(log_likelihoods, axis=0))
    waic = -2 * (lppd - p_waic)
    return waic

def main():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    # use ndarrays of X and y
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)
    # get rank of X_train
    rank = np.linalg.matrix_rank(X_train)

    # define GP_Prior and pretrain weights
    gp_prior = GPy_Prior(X_train, y_train, feature_names, mean_func=MEAN_FUNC, kernel_type=KERNEL_TYPE, kernel_structure=KERNEL_STRUCTURE, ARD=ARD)
    #weights = np.array(means_weighted).reshape(-1, 1)
    # init Gaussian Process model
    likelihood = gp.likelihoods.Gaussian() # important for classification of space state representation
    model = gp.models.GPRegression(gp_prior.X, gp_prior.y, kernel=gp_prior.kernel, mean_function=gp_prior.mean_func, noise_var=gp_prior.noise_sd_over_all_regs)
    # optimize by maximizing marginal likelihood
    model.optimize(messages=True)
    # calculate posterior predictive distribution
    mean, var = model.predict(X_test)
    # score model with GPy
    # plot posterior predictive distribution
    print(waic(model, X_test, y_test))

    # save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_model(f"{MODELDIR}/GPY_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_ARD={ARD}__{timestamp}.npy", model.param_array)

    # prepare BOA
    objective = model.log_likelihood()  
if __name__ == "__main__":
    main()
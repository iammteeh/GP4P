import GPy as gp
from GPy.examples import regression, classification, dimensionality_reduction
import pymc3 as pm
from GPy.core.parameterization.priors import Gaussian, Gamma, StudentT
from adapters.pymc.prior_construction import Priors
from GPy.likelihoods import Gaussian, StudentT, Gamma, LogLogistic, MixedNoise, Poisson, Weibull
from GPy.kern import Hierarchical, RBF, Coregionalize, Linear, Matern52
from GPyOpt.acquisitions import AcquisitionEI_MCMC #HierarchicalExpectedImprovement
from domain.env import USE_DUMMY_DATA, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE
import numpy as np
from domain.feature_model.feature_modeling import additive_kernel_permutation
from application.init_pipeline import init_pipeline, get_numpy_features

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
    pretrained_priors = Priors(X_train, y_train, feature_names)
    root_mean, root_std, means_weighted, stds_weighted, coef_matrix, noise_sd_over_all_regs = pretrained_priors.get_weighted_mvnormal_params(gamma=1, stddev_multiplier=3)
    weights = np.array(means_weighted).reshape(-1, 1)
    pretrained_priors.y = np.atleast_2d(y_train).T
    # init Gaussian Process model
    likelihood = gp.likelihoods.Poisson() # important for classification of space state representation
    #mean_func = gp.core.Mapping(1, 1)
    #mean_func.f = lambda x: np.dot(x, weights)
    inference_method = gp.inference.latent_function_inference.Laplace()
    #model = gp.models.GPRegression(pretrained_priors.X, pretrained_priors.y, kernel=gp.kern.Linear(input_dim=X_train.shape[1]))
    # build additive kernels
    components = [gp.kern.Linear(input_dim=X_train.shape[1]) for item in range(X_train.shape[1])]
    base_kernel = gp.kern.Kern(input_dim=X_train.shape[1], active_dims=None, name='basis_kernel')
    additive_kernel = additive_kernel_permutation(base_kernel, components, k=3)
    model = gp.core.GP(pretrained_priors.X, pretrained_priors.y, kernel=additive_kernel, likelihood=likelihood, inference_method=inference_method)
    # update kernel priors and likelihood
    #model.kern.variances.set_prior(gp.priors.MultivariateGaussian(np.array(means_weighted).reshape(-1, 1), np.array(stds_weighted).reshape(-1, 1)))
    model.optimize(messages=True)
    # calculate posterior predictive distribution
    mean, var = model.predict(X_test)
    
    # score model with GPy
    # plot posterior predictive distribution
    
    print(waic(model, X_test, y_test))
if __name__ == "__main__":
    main()
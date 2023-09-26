from domain.env import USE_DUMMY_DATA, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, MODELDIR, JAX
import numpy as np
from application.init_pipeline import init_pipeline, get_numpy_features
from adapters.pymc.prior_construction import PM_GP_Prior
from adapters.pymc.kernel_construction import get_additive_lr_kernel
from adapters.pymc.pm_gp import define_gp, get_kronecker_gp
#from adapters.pymc.util import save_model, load_model
import pickle
import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts
from pymc import Model, sample, sample_posterior_predictive, find_MAP, traceplot, summary, compute_log_likelihood
from pymc import gp as GP
from arviz import loo, waic
from sklearn.metrics import mean_squared_error, r2_score
from adapters.visualization import plot_dist, plot_gp_feature, plot
import matplotlib.pyplot as plt
import arviz as az
import datetime
import jax

def eval_gp(posterior_predictive_distribution, X_test, y_test):
    # Calculate mean and standard deviation
    mean_pred = np.mean(posterior_predictive_distribution['y_obs'], axis=0)
    std_pred = np.std(posterior_predictive_distribution['y_obs'], axis=0)

    # Sort the values for plotting
    idx = np.argsort(X_test)
    X_test = X_test[idx]
    mean_pred = mean_pred[idx]
    std_pred = std_pred[idx]

    return mean_pred, std_pred

def main():
    # test if TPU is used
    print(jax.device_count())
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    # use ndarrays of X and y
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)

    with Model() as model:
        # apply prior knowledge to gp => Kernel Ridge Regression to estimate c_i
        #if KERNEL_TYPE == "linear":
        f, gp, y_obs = define_gp(X_train, y_train, feature_names, mean_func=MEAN_FUNC, kernel=KERNEL_TYPE, structure=KERNEL_STRUCTURE)
        #else:
        #    gp, y_obs = define_gp(X_train, y_train, feature_names, mean_func=MEAN_FUNC, kernel=KERNEL_TYPE, structure=KERNEL_STRUCTURE)

        # execute inference and sampling
        #mp = find_MAP(method="BFGS") # deprecated
        if JAX:
            inference_data = sample_numpyro_nuts(draws=1000)
        else:
            inference_data = sample(draws=1000)
        inference_data.extend(sample_posterior_predictive(trace=inference_data, model=model))
        inference_data.extend(compute_log_likelihood(inference_data, model=model))
        
        # save InferenceData object for later purposes
        inference_data_filename = MODELDIR + f"PMGP_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.tc"
        inference_data.to_netcdf(inference_data_filename)
        # test inference data
        print(f"test inference data: {inference_data}")
        print(f"posterior inference data: {inference_data.posterior}")
        print(f"log likelihood: {inference_data.log_likelihood}")
        #print(f"test trace: {inference_data.posterior['f'].values}") if kernel_type == "linear" else print(f"test trace: {inference_data.posterior['y_obs'].values}")
        print(f"sample_stats: {inference_data.sample_stats}")
        print(f"observed data: {inference_data.observed_data}")
        #print(f"test posterior predictive: {post_pred}")
        # score gp ONLY IF LOGLIKELIHOOD IS AVAILABLE
        waic_score = waic(inference_data, pointwise=True)
        loo_score = loo(inference_data, pointwise=True)
        print(f"waic: {waic_score}")
        print(f"loo: {loo_score}")

        #ppc(post_pred, y_test)
        #az.plot_ppc(az.from_pymc3(posterior_predictive=post_pred, model=model))
        #plot(title=f"GP_PPC_{MEAN_FUNC}_{KERNEL_TYPE}")

if __name__ == "__main__":
    main()
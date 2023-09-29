from domain.env import USE_DUMMY_DATA, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, MODELDIR, JAX, GP_MODE, NUTS_SAMPLER
import numpy as np
from application.init_pipeline import init_pipeline, get_numpy_features
from adapters.pymc.prior_construction import PM_GP_Prior
from adapters.pymc.kernel_construction import get_additive_lr_kernel
from adapters.pymc.pm_gp import define_gp, define_marginal_gp
#from adapters.pymc.util import save_model, load_model
import pickle
import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts, sample_blackjax_nuts
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
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    # use ndarrays of X and y
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)

    # register models
    KERNEL_TYPE = ["linear", "RBF", "matern12", "matern32", "matern52"]
    KERNEL_STRUCTURE = ["improper", "simple", "additive"]

    model_data = {}
    for kernel_structure in KERNEL_STRUCTURE:
        for kernel_type in KERNEL_TYPE:

            with Model() as model:
                # apply prior knowledge to gp
                if GP_MODE == "marginal":
                    y_obs, gp = define_marginal_gp(X_train, y_train, feature_names, mean_func=MEAN_FUNC, kernel=kernel_type, structure=kernel_structure)
                else:
                    f, gp, y_obs = define_gp(X_train, y_train, feature_names, mean_func=MEAN_FUNC, kernel=kernel_type, structure=kernel_structure)

                # execute inference and sampling
                #mp = find_MAP(method="BFGS") # deprecated
                print(f"sampling with {NUTS_SAMPLER} (JAX={JAX})")
                if JAX and NUTS_SAMPLER == "numpyro":
                    inference_data = sample_numpyro_nuts(draws=1000)
                elif JAX and NUTS_SAMPLER == "blackjax":
                    inference_data = sample_blackjax_nuts(draws=1000)
                elif NUTS_SAMPLER == "numpyro":
                    inference_data = sample(draws=1000, nuts_sampler="numpyro")
                elif NUTS_SAMPLER == "blackjax":
                    inference_data = sample(draws=1000, nuts_sampler="blackjax")
                else:
                    inference_data = sample(draws=1000)
                inference_data.extend(sample_posterior_predictive(trace=inference_data, model=model))
                inference_data.extend(compute_log_likelihood(inference_data, model=model))
                    
                # save InferenceData object for later purposes
                inference_data_filename = MODELDIR + f"PMGP_{MEAN_FUNC}_{kernel_type}_{kernel_structure}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.tc"
                inference_data.to_netcdf(inference_data_filename)
                model_data[f"{kernel_structure}_{kernel_type}"] = inference_data
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

    # compare models
    az.compare(model_data, ic="loo", scale="deviance")
    # Plot convergence
#    for i, (structure, kernel) in #enumerate(zip(KERNEL_STRUCTURE, KERNEL_TYPE)):
#        az.plot_trace(model_data[i])
#        plt.title(f"Convergence plot for kernel: {type(kernel).__name__}")
#        plt.show()
if __name__ == "__main__":
    main()
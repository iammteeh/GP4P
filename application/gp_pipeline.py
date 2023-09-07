from domain.env import USE_DUMMY_DATA, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE
import numpy as np
from application.init_pipeline import init_pipeline, get_numpy_features
from adapters.pymc.prior_construction import PM_GP_Prior
from adapters.pymc.kernel_construction import get_additive_lr_kernel
from adapters.pymc.pm_gp import define_gp, get_kronecker_gp
from adapters.pymc.util import save_model, load_model
import pymc3 as pm
from pymc3 import Model, sample, sample_posterior_predictive, find_MAP, traceplot, summary
from pymc3 import gp as GP
from arviz import loo, waic
from sklearn.metrics import mean_squared_error, r2_score
from adapters.visualization import plot_dist, plot_gp_feature, plot
import matplotlib.pyplot as plt
import arviz as az

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
    KERNEL_TYPE = ["linear", "expquad", "matern52"]
    KERNEL_STRUCTURE = ["simple", "additive"]

    for kernel_structure in KERNEL_STRUCTURE:
        for kernel_type in KERNEL_TYPE:

            traces = []
            with Model() as model:
                # apply prior knowledge to gp
                if kernel_type == "linear":
                    f, gp, y_obs = define_gp(X_train, y_train, feature_names, mean_func=MEAN_FUNC, kernel=kernel_type, structure=kernel_structure)
                else:
                    gp, y_obs = define_gp(X_train, y_train, feature_names, mean_func=MEAN_FUNC, kernel=kernel_type, structure=kernel_structure)

                trace = sample(draws=1000, cores=1)
                saved_trace = pm.save_trace(trace)
                traces.append(trace)
                print(f"feature names: {feature_names}")
                post_pred = sample_posterior_predictive(trace=trace, model=model, var_names=['y_obs'], samples=1000)
                save_model(model, saved_trace, X_train)
                mp = find_MAP(method="BFGS")

                # score gp
                waic_score = waic(trace, model)
                loo_score = loo(trace, model)
                print(f"waic: {waic_score}")
                print(f"loo: {loo_score}")

                #ppc(post_pred, y_test)
                az.plot_ppc(az.from_pymc3(posterior_predictive=post_pred, model=model))
                plot(title=f"GP_PPC_{MEAN_FUNC}_{KERNEL_TYPE}")

                # Plot convergence
                for i, kernel in enumerate(zip(kernel_structure, kernel_type)):
                    pm.traceplot(traces[i])
                    plt.title(f"Convergence plot for kernel: {type(kernel).__name__}")
                    plt.show()
if __name__ == "__main__":
    main()
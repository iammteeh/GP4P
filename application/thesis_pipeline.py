from domain.env import USE_DUMMY_DATA
import numpy as np
from application.init_pipeline import init_pipeline, get_numpy_features
from adapters.pca import kernel_pca
from adapters.calculate_prior_information import Priors
from adapters.pm_gp import define_gp
from pymc3 import Model, sample, sample_posterior_predictive, traceplot, summary, waic, loo
from pymc3 import gp as GP
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from adapters.visualization import plot_dist, plot_gp_feature

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
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features=None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    # use ndarrays of X and y
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)

    # calculate prior weighted multivariate normal
    priors = Priors(X_train, y_train, feature_names)
    root_mean, root_std, means_weighted, stds_weighted, coef_matrix, noise_sd_over_all_regs = priors.get_weighted_normal_params(X_train, y_train, feature_names, gamma=1, stddev_multiplier=3)
    cov_func = priors.get_gp_cov_func(X_train, means_weighted, coef_matrix, noise_sd_over_all_regs, kernel="linear")
    # cov matrix may already be noised

    # reduce dimensionality
    X_train = kernel_pca(X_train, y_train, kernel="poly", degree=2, gamma=0.03)
    
    with Model() as model:
        # apply prior knowledge to gp
        f, gp, y_obs = define_gp(X_train, root_mean, cov_func, y_train, noise=noise_sd_over_all_regs)
        # Define Gaussian Process likelihood
        #y_obs = GP.GP('y_obs', gp, sigma=error_prior, observed={'X': X_train, 'Y': y_train})
        trace = sample(1000)
        #feature_names.delete(0)
        print(f"feature names: {feature_names}")
        post_pred = sample_posterior_predictive(trace=trace, model=model, var_names=['y_obs'], samples=1000)
        #plot_dist(post_pred, "GP predictive posterior")

        #mean_pred, std_pred = eval_gp(post_pred, X_test, y_test)
        mean_pred = np.mean(post_pred['y_obs'], axis=0)

        print(f"mean_pred: {mean_pred}")
        #print(f"std_pred: {std_pred}")

        # score gp
        waic_score = waic(trace, model)
        loo_score = loo(trace, model)
        print(f"waic: {waic_score}")
        print(f"loo: {loo_score}")
        #MSE = mean_squared_error(y_test, mean_pred)
        #print(f"MSE: {MSE}")

if __name__ == "__main__":
    main()
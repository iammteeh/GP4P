from domain.env import USE_DUMMY_DATA
import numpy as np
from application.basic_pipeline import init_pipeline
from adapters.calculate_pretrained_priors import Priors
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
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features=None)
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    # use ndarrays of X and y
    X_train = X_train[1]
    X_test = X_test[1]
    y_train = y_train[1]
    y_test = y_test[1]

    # calculate prior weighted multivariate normal
    priors = Priors(X_train, y_train, feature_names)
    µ_vector, cov_matrix, base_prior, error_prior, weighted_errs_per_sample, weighted_rel_errs_per_sample = priors.get_prior_weighted_normal(X_train, y_train, feature_names, gamma=1, stddev_multiplier=3, kernel="expquad")
    # cov matrix may already be noised
    
    with Model() as model:
        # apply prior knowledge to gp
        f, gp, y_obs = define_gp(X_train, µ_vector, cov_matrix, y_train, prior=None, noise=None)
        # Define Gaussian Process likelihood
        #y_obs = GP.GP('y_obs', gp, sigma=error_prior, observed={'X': X_train, 'Y': y_train})
        trace = sample(1000)
        #feature_names.delete(0)
        print(f"feature names: {feature_names}")
        post_pred = sample_posterior_predictive(trace=trace, model=model, var_names=['y_obs'], keep_size=True)
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
        MSE = mean_squared_error(y_test, mean_pred)
        print(f"MSE: {MSE}")

if __name__ == "__main__":
    main()
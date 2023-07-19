from domain.env import USE_DUMMY_DATA, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE
import numpy as np
from application.init_pipeline import init_pipeline, get_numpy_features
from adapters.pymc.prior_construction import GP_Prior
from adapters.pymc.kernel_construction import get_gp_cov_func, get_additive_lr_kernel
from adapters.pymc.pm_gp import define_gp, get_kronecker_gp
from pymc3 import Model, sample, sample_posterior_predictive, find_MAP, traceplot, summary
from pymc3 import gp as GP
from arviz import loo, waic
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
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    # use ndarrays of X and y
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)

    with Model() as model:
        # apply prior knowledge to gp
        f, gp, y_obs = define_gp(X_train, y_train, feature_names, kernel="linear")
        #f, gp = get_kronecker_gp(X_train, y_train, (root_mean, root_std), (means_weighted, stds_weighted), noise=noise_sd_over_all_regs)
        # Define Gaussian Process likelihood
        #y_obs = gp.marginal_likelihood("y_obs", X=X_train, y=y_train, noise=noise_sd_over_all_regs)
        trace = sample(1000)
        print(f"feature names: {feature_names}")
        post_pred = sample_posterior_predictive(trace=trace, model=model, var_names=['y_obs'], samples=1000)
        #plot_dist(post_pred, "GP predictive posterior")

        #mean_pred, std_pred = eval_gp(post_pred, X_test, y_test)
        mp = find_MAP(method="BFGS")
        import random
        #TODO: develop systemic sampling method (e.g. stratified sampling with k-fold cross validation)
        # pick random sample from the dataframe test set
        X_test = X_test.reshape(-1, 1) # reshapes X_test into a 2D array with one column
        Xnew = X_test[[random.randint(0, len(X_test)-1)]] # selects one row from X_test, keeps it as a 2D array
        #mu, var = gp.conditional(f"{Xnew}", Xnew, point=mp, diag=True)

        #print(f"mean_pred: {mean_pred}")
        #print(f"std_pred: {std_pred}")
        #print(f"mu: {mu}")
        #print(f"var: {var}")

        # score gp
        waic_score = waic(trace, model)
        loo_score = loo(trace, model)
        print(f"waic: {waic_score}")
        print(f"loo: {loo_score}")
        #MSE = mean_squared_error(y_test, mean_pred)
        #print(f"MSE: {MSE}")

if __name__ == "__main__":
    main()
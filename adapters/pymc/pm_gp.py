import pymc as pm
import numpy as np
from adapters.pymc.prior_construction import PM_GP_Prior

def define_gp(X, y, feature_names, mean_func="linear", kernel="linear", structure="simple", gp=None):
    """
    using the exact marginal without approximation as with Kernel Ridge Regression (Elastic Net)
    it is ensured that the matrix is invertible and lambda is positive definite
    """
    gp_prior = PM_GP_Prior(X, y, feature_names, mean_func=mean_func, kernel=kernel, structure=structure)
    #print(f"shape of gp_prior.X: {gp_prior.X.shape}")
    print(f"mean_func is {mean_func}")
    print(f"kernel is {kernel}")
    gp = pm.gp.Latent(mean_func=gp_prior.mean_func, cov_func=gp_prior.kernel) #if kernel == "linear" else pm.gp.Marginal(mean_func=gp_prior.mean_func, cov_func=gp_prior.kernel)
    print(f"gp is {type(gp)}")
    print(f"X is {type(X)}")
    #if kernel == "linear":
    f = gp.prior("f", X=gp_prior.X)
    # define y_obs as StudentT to ensure observations are independet from additional extrafunctional features 
    # that are NOT in the data
    print(f"shape of f: {f.shape}")
    nu = 1 + pm.Gamma(
        "nu", alpha=2, beta=0.1
    )
    #y_obs = pm.StudentT("y_obs", mu=f, lam=1.0 / gp_prior.noise_sd_over_all_regs, nu=nu, observed=gp_prior.y)
    # take normal distribution 
    y_obs = pm.Normal("y_obs", mu=f, sigma=gp_prior.noise_sd_over_all_regs, observed=gp_prior.y)
    return f, gp, y_obs
    #else: 
    #    y_obs = gp.marginal_likelihood("y_obs", X=gp_prior.X, y=gp_prior.y, noise=gp_prior.noise_sd_over_all_regs)
    #    return gp, y_obs

def define_marginal_gp(X, y, feature_names, mean_func="linear", kernel="linear", structure="simple", gp=None):
    """
    using the exact marginal without approximation as with Kernel Ridge Regression (Elastic Net)
    it is ensured that the matrix is invertible and lambda is positive definite
    """
    gp_prior = PM_GP_Prior(X, y, feature_names, mean_func=mean_func, kernel=kernel, structure=structure)
    #print(f"shape of gp_prior.X: {gp_prior.X.shape}")
    print(f"mean_func is {mean_func}")
    print(f"kernel is {kernel}")
    gp = pm.gp.Marginal(mean_func=gp_prior.mean_func, cov_func=gp_prior.kernel)
    sigma = pm.HalfNormal("sigma", sigma=gp_prior.noise_sd_over_all_regs)
    y_obs = gp.marginal_likelihood("f", X=gp_prior.X, y=gp_prior.y, sigma=sigma)
    print(f"gp is {type(gp)}")
    print(f"X is {type(X)}")
    return y_obs, gp

def get_kronecker_gp(X, y, intercept, coefs, noise):
    root_mean, root_std = intercept
    means_weighted, stds_weighted = coefs

    intercept = pm.Normal("intercept", mu=root_mean, sigma=root_std)
    GPs = []
    for i, (mu, sigma) in enumerate(zip(means_weighted, stds_weighted)):
        coef = pm.Normal(f'coef_{i}', mu=mu, sigma=sigma)  # Prior for coefficient
        ls = pm.Gamma(f"ls_{i}", alpha=2, beta=2)
        eta = pm.HalfNormal(f"eta_{i}", sigma=2)

        mean_func = pm.gp.mean.Linear(coeffs=coef, intercept=root_mean)  # Define mean function
        cov_funcs = [eta**2 * pm.gp.cov.ExpQuad(1, ls=ls) for i in range(X.shape[1])]  # One covariance function per feature
        print(f"cov_funcs: {cov_funcs}")
        print(f"len(cov_funcs): {len(cov_funcs)}")
        print(f"X.shape[1]: {X.shape}")
        # cov func with estimated lengthscale according to pca components
        #kron_product = pm.gp.cov.Kron(cov_funcs)
        Xs = X@X.T
        #Xs = [X[:, None] for i in range(X.shape[1])]
        print(len(Xs))
        gp = pm.gp.MarginalKron(mean_func=mean_func, cov_funcs=cov_funcs)
        f = gp.marginal_likelihood(f"f_{i}", Xs=Xs, y=y, sigma=noise)

    return f, gp
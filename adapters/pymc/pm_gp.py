import pymc3 as pm
import theano.tensor as tt
import numpy as np
from adapters.pymc.prior_construction import GP_Prior

def define_gp(X, y, feature_names, mean_func="linear", kernel="linear", noise=None, gp=None):
    gp_prior = GP_Prior(X, y, feature_names, mean_func=mean_func, kernel=kernel)
    print(f"shape of gp_prior.X: {gp_prior.X.shape}")
    gp = pm.gp.Latent(mean_func=gp_prior.mean_func, cov_func=gp_prior.kernel) #if noise is None else pm.gp.Marginal(mean_func=µ_vector, cov_func=cov_func)
    #f = gp.marginal_likelihood("f", X=X)
    print(f"X is {type(X)}")
    f = gp.prior("f", X=X)
    print(f"shape of f: {f.shape}")
    y_obs = pm.Normal("y_obs", mu=f, sigma=noise, observed=gp_prior.y)
    return f, gp, y_obs

def get_additive_gp(X, y, intercept, coefs, noise):
    root_mean, root_std = intercept
    means_weighted, stds_weighted = coefs

    intercept = pm.Normal("intercept", mu=root_mean, sigma=root_std)
    GPs = []
    for i, (mu, sigma) in enumerate(zip(means_weighted, stds_weighted)):
        coef = pm.Normal(f'coef_{i}', mu=mu, sigma=sigma)  # Prior for coefficient
        #mean_func = pm.gp.mean.Constant(coef * X[:, i])  # Define mean function
        mean_func = pm.gp.mean.Linear(coeffs=coef * X[:, i], intercept=root_mean)
        #cov_func = pm.gp.cov.ExpQuad(X.shape[1], ls=1)  # Assume a simple squared exponential kernel
        cov_func = pm.gp.cov.Linear(input_dim=X.shape[1], c=1)
        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
        globals()['f_%s' % i] = gp.conditional("f_%s" % i, X=X[:, i].reshape(-1, 1), noise=noise, )
        GPs.append(gp)

    f_prime = intercept + tt.sum([gp.marginal_likelihood("f", X=X, y=y) for gp in GPs])

    return f_prime, GPs

def get__additive_kronecker_gp(X, y, intercept, coefs, noise):
    root_mean, root_std = intercept
    means_weighted, stds_weighted = coefs

    intercept = pm.Normal("intercept", mu=root_mean, sigma=root_std)
    GPs = []
    for i, (mu, sigma) in enumerate(zip(means_weighted, stds_weighted)):
        coef = pm.Normal(f'coef_{i}', mu=mu, sigma=sigma)  # Prior for coefficient
        #mean_func = pm.gp.mean.Constant(coef * X[:, i])  # Define mean function
        mean_func = pm.gp.mean.Linear(coeffs=coef * X[:, i], intercept=root_mean)
        #cov_func = pm.gp.cov.ExpQuad(X.shape[1], ls=1)  # Assume a simple squared exponential kernel
        #cov_func = pm.gp.cov.Linear(input_dim=[i, i+1], c=1)
        cov_func_1 = pm.gp.cov.Linear(input_dim=i, c=1)
        cov_func_2 = pm.gp.cov.Linear(input_dim=i+1, c=1)
        kronecker_product = pm.gp.cov.Kron([cov_func_1, cov_func_2])
        gp = pm.gp.MarginalKron(mean_func=mean_func, cov_funcs=kronecker_product)
        #globals()['f_%s' % i] = gp.conditional("f_%s" % i, X=X[:, i].reshape(-1, 1), noise=noise, )
        GPs.append(gp)
    # take the list of GPs and sum them up
    for i in range(len(GPs)):
        gp += GPs[i]
    f_prime = intercept + tt.sum([gp.marginal_likelihood("f", Xs=[X[:, None], X[:, None]], y=y, sigma=noise) for gp in GPs])

    return f_prime, GPs

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
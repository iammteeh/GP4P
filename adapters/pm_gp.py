import pymc3 as pm
import theano.tensor as tt
import numpy as np

def define_gp(X, Y, µ_vector, cov_func, noise=None, gp=None):
    # convert µ_vector to pm mean object
    mean_func = pm.gp.mean.Constant(c=np.mean(µ_vector))
    gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func) #if noise is None else pm.gp.Marginal(mean_func=µ_vector, cov_func=cov_func)
    #f = gp.marginal_likelihood("f", X=µ_vector, y=prior, noise=0.1)
    print(f"X is {type(X)}")
    f = gp.prior("f", X=X)
    y_obs = pm.Normal("y_obs", mu=f, sigma=noise, observed=Y)
    return f, gp, y_obs

def define_gp_alt(X, Y, cov_func, **kwargs):
    mean_func = pm.gp.mean.Constant(c=np.mean(kwargs["µ_vector"])) if "µ_vector" in kwargs else pm.gp.mean.Constant(c=np.mean(Y))

    gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
    f = gp.prior("f", X=X)
    y_obs = pm.Normal("y_obs", mu=f, sigma=0.1, observed=Y)
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

def get_kronecker_gp(X, y, intercept, coefs, noise):
    root_mean, root_std = intercept
    means_weighted, stds_weighted = coefs

    intercept = pm.Normal("intercept", mu=root_mean, sigma=root_std)
    GPs = []
    for i, (mu, sigma) in enumerate(zip(means_weighted, stds_weighted)):
        coef = pm.Normal(f'coef_{i}', mu=mu, sigma=sigma)  # Prior for coefficient
        #mean_func = pm.gp.mean.Constant(coef * X[:, i])  # Define mean function
        mean_func = pm.gp.mean.Linear(coeffs=coef * X[:, i], intercept=root_mean)
        #cov_func = pm.gp.cov.ExpQuad(X.shape[1], ls=1)  # Assume a simple squared exponential kernel
        cov_func = pm.gp.cov.Linear(input_dim=[i, i+1], c=1)
        cov_func_1 = pm.gp.cov.Linear(input_dim=[i], c=1)
        cov_func_2 = pm.gp.cov.Linear(input_dim=[i+1], c=1)
        kronecker_product = pm.gp.cov.Kron([cov_func_1, cov_func_2])
        gp = pm.gp.MarginalKron(mean_func=mean_func, cov_func=kronecker_product)
        globals()['f_%s' % i] = gp.conditional("f_%s" % i, X=X[:, i].reshape(-1, 1), noise=noise, )
        GPs.append(gp)

    f_prime = intercept + tt.sum([gp.marginal_likelihood("f", X=X, y=y) for gp in GPs])

    return f_prime, GPs

def get_additive_kronecker_gp(Xs, y, intercept, coefs, noise):
    root_mean, root_std = intercept
    means_weighted, stds_weighted = coefs

    intercept = pm.Normal("intercept", mu=root_mean, sigma=root_std)
    GPs = []
    for i, (mu, sigma) in enumerate(zip(means_weighted, stds_weighted)):
        coef = pm.Normal(f'coef_{i}', mu=mu, sigma=sigma)  # Prior for coefficient
        ls = pm.Gamma(f"ls_{i}", alpha=2, beta=2)
        eta = pm.HalfNormal(f"eta_{i}", sigma=2)

        mean_func = pm.gp.mean.Linear(coeffs=coef, intercept=root_mean)  # Define mean function
        cov_funcs = [eta**2 * pm.gp.cov.ExpQuad(1, ls=ls) for _ in range(Xs[i].shape[1])]  # One covariance function per feature
        # cov func with estimated lengthscale according to pca components
        kron_product = pm.gp.cov.Kron(cov_funcs)

        gp = pm.gp.MarginalKron(mean_func=mean_func, cov_funcs=kron_product)
        gp.marginal_likelihood(f"f_{i}", Xs=Xs[i], y=y, sigma=noise)

        GPs.append(gp)

    return GPs
import pymc3 as pm
import theano.tensor as tt
import numpy as np

def define_gp(X, Y, µ_vector, cov_func, kernel=None, noise=None):
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
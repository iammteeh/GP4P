import pymc3 as pm
import theano.tensor as tt
import numpy as np

def define_gp(X, µ_vector, cov_matrix, Y, prior=None, kernel=None, noise=None):
    ### override arguments for test purpose only
    kernel = "expquad"
    prior = None
    ###
    # convert µ_vector to pm mean object
    mean_func = pm.gp.mean.Constant(c=np.mean(µ_vector))
    # create covariance object that can be used by pm
    if not isinstance(cov_matrix, pm.gp.cov.ExpQuad):
        cov_func = pm.gp.cov.ExpQuad(len(µ_vector), ls=1.0) if kernel is not None else tt.as_tensor_variable(cov_matrix)
    gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func if cov_func else cov_matrix) if noise is None else pm.gp.Marginal(mean_func=µ_vector, cov_func=cov_func)
    #f = gp.marginal_likelihood("f", X=µ_vector, y=prior, noise=0.1)
    #if not isinstance(prior, pm.MvNormal):
    #    prior = pm.MvNormal("coef", mu=np.array(µ_vector), cov=cov) 
    print(f"X is {type(X)}")
    f = gp.prior("f", X=X) if prior is None else prior # maybe need to convert numpyro dist object to pymc3 dist object first
    y_obs = pm.Normal("y_obs", mu=f, sigma=noise, observed=Y)
    return f, gp, y_obs
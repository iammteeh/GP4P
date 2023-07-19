from pymc3.gp.cov import Covariance, Linear, WhiteNoise, Constant, Periodic, Matern52, ExpQuad, Polynomial
from pymc3.gp.cov import Combination, Add, Prod, ScaledCov, Kron
from pymc3.distributions import Gamma, Normal, HalfNormal, Uniform, HalfCauchy, MvNormal, MvStudentT
import numpy as np
from scipy.linalg import sqrtm

# deprecated
def get_gp_cov_func(self, X, µ_vector, coef_matrix=None, noise=None, kernel="expquad", **kernel_params):
    def is_positive_semi_definite(matrix):
        return np.all(np.linalg.eigvals(matrix) >= 0)
    
    if kernel == "linear" and coef_matrix is not None:
        cov_matrix = np.corrcoef(coef_matrix) # perturb coef_matrix to get cov_matrix
        cov_matrix = np.cov(coef_matrix, rowvar=False)

        if np.isnan(µ_vector).any():
            µ_vector = np.nan_to_num(µ_vector)
        if np.isnan(cov_matrix).any() and np.isinf(cov_matrix).any():
            print("cov_matrix contains nan or inf")
            cov_matrix = np.eye(len(µ_vector))

        if not is_positive_semi_definite(cov_matrix):
            print("Covariance Matrix is not positive semi definite. Adding noise.")
            cov_matrix += np.eye(len(µ_vector)) * noise
            if not is_positive_semi_definite(cov_matrix):
                print("Covariance Matrix is still not positive semi definite. Compute PSD approximation.")
                cov_matrix = sqrtm(cov_matrix)
                cov_matrix = np.dot(cov_matrix, cov_matrix.T)
                if not is_positive_semi_definite(cov_matrix):
                    raise ValueError("Covariance Matrix is still not positive semi definite. Cannot compute GP prior.")
        
        #coef_prior = dist.MultivariateNormal(jnp.array(µ_vector), covariance_matrix=jnp.array(cov_matrix)
        return Linear(input_dim=len(X.T), c=1)
                
    elif kernel == "expquad":
        return ExpQuad(len(µ_vector), ls=1.0)
    #elif kernel == "rbf":
    #    return RBF(1, coef_matrix.shape[1]).full(X.T) # Compute the covariance matrix with a radial basis function kernel
    elif kernel == "matern":
        return Matern52(1, coef_matrix.shape[1]).full(X.T)
    elif kernel == "poly":
        tau = 0.1
        cov_func = tau * Polynomial(input_dim=len(X.T), **{k: v for k, v in kernel_params.items() if k in ["c", "d", "offset"]})
        return cov_func

# atomic kernel construction
def get_linear_kernel(X, active_dims=None):
    if active_dims is not None:
        print(f"input_dim=X.T: {X.T}")
        return Linear(input_dim=len(X.T), c=1, active_dims=active_dims)
    else:
        return Linear(input_dim=len(X.T), c=1, active_dims=[i for i in range(X.shape[1])])

def get_white_noise_kernel(sigma=1.0):
    return WhiteNoise(sigma=sigma)

def get_constant_kernel(c=1.0):
    return Constant(c=c)

def get_periodic_kernel(X, active_dims=None):
    if active_dims is not None:
        return Periodic(input_dim=X.T, active_dims=active_dims)
    else:
        return Periodic(input_dim=X.T, active_dims=[i for i in range(X.shape[1])])

def get_matern52_kernel(X, active_dims=None, hyper_priors=True, **hyper_prior_params):    
    if hyper_priors:
        ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"], shape=X.shape[1])
        eta = HalfCauchy("eta", beta=2, shape=X.shape[1])
    if active_dims is not None:
        return eta ** 2 * Matern52(input_dim=X.T, ls=ls, active_dims=active_dims)
    else:
        return eta ** 2 * Matern52(input_dim=X.T, active_dims=[i for i in range(X.shape[1])])
    

# composite kernel construction
def get_additive_kernel(*kernels, mode="LR"):
    if any([isinstance(kernel, Covariance) for kernel in kernels]):
        raise ValueError("Cannot add kernels that are already additive")
    return Add(*kernels)

def get_additive_lr_kernel(X, root_mean, root_std):
    return Add(get_constant_kernel(c=root_mean), get_linear_kernel(X), get_white_noise_kernel(sigma=root_std))


# regularized kernel construction

# coregionalized kernel construction
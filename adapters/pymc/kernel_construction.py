from pymc.gp.cov import Covariance, Combination, Linear, WhiteNoise, Constant, Polynomial, Periodic, Matern12, Matern32, Matern52, ExpQuad, RatQuad
from pymc.gp.cov import Combination, Add, Prod, ScaledCov, Kron
from pymc.distributions import Gamma, Normal, HalfNormal, Uniform, HalfCauchy, MvNormal, MvStudentT
import numpy as np
from scipy.linalg import sqrtm

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
    if active_dims:
        return Periodic(input_dim=len(X.T), active_dims=active_dims)
    else:
        return Periodic(input_dim=len(X.T), active_dims=[i for i in range(X.shape[1])])

def get_matern12_kernel(X, active_dims=None, **hyper_prior_params):
    if hyper_prior_params:
        ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
        eta = HalfCauchy("eta", beta=2)
        if active_dims:
            return eta ** 2 * Matern12(input_dim=len(X.T), ls=ls, active_dims=active_dims)
        else:
            return eta ** 2 * Matern12(input_dim=len(X.T), ls=ls)
    else:
        if not active_dims:
            return Matern12(input_dim=len(X.T), ls=1)# active_dims=[i for i in range(X.shape[1])])

def get_matern32_kernel(X, active_dims=None, **hyper_prior_params):
    if hyper_prior_params:
        ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
        eta = HalfCauchy("eta", beta=2)
        if active_dims:
            return eta ** 2 * Matern32(input_dim=len(X.T), ls=ls, active_dims=active_dims)
        else:
            return eta ** 2 * Matern32(input_dim=len(X.T), ls=ls)
    else:
        if not active_dims:
            return Matern32(input_dim=len(X.T), ls=1)# active_dims=[i for i in range(X.shape[1])])

def get_matern52_kernel(X, active_dims=None, **hyper_prior_params):
    if hyper_prior_params:
        ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
        eta = HalfCauchy("eta", beta=2)
        if active_dims:
            return eta ** 2 * Matern52(input_dim=len(X.T), ls=ls, active_dims=active_dims)
        else:
            return eta ** 2 * Matern52(input_dim=len(X.T), ls=ls)
    else:
        if not active_dims:
            return Matern52(input_dim=len(X.T), ls=1)# active_dims=[i for i in range(X.shape[1])])
    
def get_squared_exponential_kernel(X, active_dims=None, hyper_priors=True, **hyper_prior_params):
    if hyper_prior_params:       
        ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
        eta = HalfCauchy("eta", beta=2)
        if active_dims:
            return eta ** 2 * ExpQuad(input_dim=len(X.T), ls=ls)
        else:
            return eta ** 2 * ExpQuad(input_dim=len(X.T), ls=ls, active_dims=active_dims)
    else:
        if not active_dims:
            return ExpQuad(input_dim=len(X.T), ls=1)

def get_experimental_kernel(X):
    #return Linear(len(µ_vector), cov_matrix)
    return Linear(input_dim=len(X.T), c=1)

# composite kernel construction
def get_base_kernels(X, kernel="linear", **hyper_prior_params):
    if kernel == "linear":
        base_kernels = [Linear(input_dim=len(X.T), c=1) for item in range(X.shape[1])]
        return base_kernels
    elif kernel == "RBF" or kernel == "ExpQuad":
        if hyper_prior_params:
            ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
            eta = HalfCauchy("eta", beta=2)
            base_kernels = [eta ** 2 * ExpQuad(input_dim=len(X.T), ls=ls) for item in range(X.shape[1])]
        else:
            base_kernels = [ExpQuad(input_dim=len(X.T), ls=1) for item in range(X.shape[1])]
    elif kernel == "matern12":
        if hyper_prior_params:
            ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
            eta = HalfCauchy("eta", beta=2)
            base_kernels = [eta ** 2 * Matern12(input_dim=len(X.T), ls=ls) for item in range(X.shape[1])]
        else:
            base_kernels = [Matern12(input_dim=len(X.T), ls=1) for item in range(X.shape[1])]
    elif kernel == "matern32":
        if hyper_prior_params:
            ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
            eta = HalfCauchy("eta", beta=2)
            base_kernels = [eta ** 2 * Matern32(input_dim=len(X.T), ls=ls) for item in range(X.shape[1])]
        else:
            base_kernels = [Matern32(input_dim=len(X.T), ls=1) for item in range(X.shape[1])]
    elif kernel == "matern52":
        if hyper_prior_params:
            ls = Gamma("ls", alpha=2, beta=2, mu=hyper_prior_params["mean"], sigma=hyper_prior_params["sigma"])
            eta = HalfCauchy("eta", beta=2)
            base_kernels = [eta ** 2 * Matern52(input_dim=len(X.T), ls=ls) for item in range(X.shape[1])]
        else:
            base_kernels = [Matern52(input_dim=len(X.T), ls=1) for item in range(X.shape[1])]
    return base_kernels

def get_additive_kernel(*kernels, mode="LR"):
    if any([isinstance(kernel, Covariance) for kernel in kernels]):
        raise ValueError("Cannot add kernels that are already additive")
    return Add(*kernels)

def additive_kernel_permutation(items, k=3):
    import itertools
    import time
    permutations = [list(p) for p in itertools.combinations(items, r=k)]
    print(f"Start building additive kernel. \n Calculated {len(permutations)}.")
    start = time.time()
    additive_kernel = Add([Prod([combination[0], combination[1]]) 
                      for p, permutation in enumerate(permutations) 
                      for c, combination in enumerate(itertools.combinations(permutation, 2))])
    end = time.time()
    print(f"Finished building additive kernel. It took {end - start:.2f}s.")
    return additive_kernel

def get_additive_lr_kernel(X, root_mean, root_std):
    """
    only with standard normal prior mean and std
    """
    return get_constant_kernel(c=root_mean) + get_linear_kernel(X) + get_white_noise_kernel(sigma=root_std)

def get_standard_lr_kernel(X):
    """
    only with standard normal prior mean and std
    """
    return get_constant_kernel(c=1) + get_linear_kernel(X) + get_white_noise_kernel(sigma=1)


# regularized kernel construction

# coregionalized kernel construction
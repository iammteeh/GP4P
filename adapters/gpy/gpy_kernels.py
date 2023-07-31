from GPy.core.parameterization.priors import Gaussian, Gamma, StudentT, HalfT
#from GPy.likelihoods import Gaussian, StudentT, Gamma, LogLogistic, MixedNoise, Poisson, Weibull
from GPy.kern import Hierarchical, RBF, Coregionalize, Linear, Matern52

def get_kernel(kernel_name, input_dim, active_dims=None, variances=None, ARD=False):
    if kernel_name == "rbf":
        return RBF(input_dim, ARD=ARD)
    elif kernel_name == "matern":
        kernel = Matern52(input_dim, ARD=ARD)
        if variances is not None:
            kernel.variance.set_prior(Gaussian(variances[0], variances[1]))
        kernel.lengthscale.set_prior(Gamma(2, 1))
        return kernel
    elif kernel_name == "linear":
        return Linear(input_dim, ARD=ARD)
    elif kernel_name == "hierarchical":
        return Hierarchical(input_dim, ARD=ARD)
    
def get_linear_kernel(X, active_dims=None, ARD=False):
    return Linear(input_dim=X.shape[1], active_dims=active_dims, ARD=ARD)
    
def get_matern52_kernel(X, active_dims=None, ARD=True, hyper_priors=False, **hyper_prior_params):    
    kernel = Matern52(input_dim=X.shape[1], active_dims=active_dims, ARD=ARD)
    if hyper_priors:
        kernel.lengthscale.set_prior(Gamma(hyper_prior_params["gamma_a"], hyper_prior_params["gamma_b"]))
        eta = HalfT(A=hyper_prior_params["A"], nu=hyper_prior_params["nu"])
        return eta ** 2 * kernel
    return kernel

def get_base_kernels(X, kernel="linear", ARD=True, active_dims=None, hyper_priors=True, **hyper_prior_params):
    if kernel == "linear":
        base_kernels = [Linear(input_dim=X.shape[1], ARD=ARD) for item in range(X.shape[1])]
        return base_kernels
    elif kernel == "matern52":
        base_kernels = [Matern52(input_dim=X.shape[1], ARD=ARD) for item in range(X.shape[1])]
        if hyper_priors:
            base_kernels = [kernel.lengthscale.set_prior(Gamma(hyper_prior_params["gamma_a"], hyper_prior_params["gamma_b"])) for kernel in base_kernels]
            # add eta prior
        return base_kernels

        
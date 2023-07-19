from GPy.core.parameterization.priors import Gaussian, Gamma, StudentT
from GPy.kern import Hierarchical, RBF, Coregionalize, Linear, Matern52
from GPyOpt.acquisitions import HierarchicalExpectedImprovement

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
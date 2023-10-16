import gpytorch.kernels as gk
from gpytorch.utils import grid
from torch import tensor
from gpytorch.kernels import RBFKernel, GridInterpolationKernel, ScaleKernel, ProductKernel, AdditiveStructureKernel, LinearKernel, PeriodicKernel, MaternKernel
from gpytorch.priors import GammaPrior, HalfCauchyPrior
import numpy as np
import time

# Linear Kernel
def get_linear_kernel(X, **hyper_prior_params):
    return LinearKernel()

# Periodic Kernel
def get_periodic_kernel(X, **hyper_prior_params):
    return PeriodicKernel()

# Matern Kernels
def get_matern12_kernel(X, **hyper_prior_params):
    if hyper_prior_params:
        alpha = np.square(hyper_prior_params["mean"]) / np.square(hyper_prior_params["sigma"])
        beta = np.square(hyper_prior_params["mean"]) / np.square(hyper_prior_params["sigma"])
        lengthscale_prior = GammaPrior(alpha, beta)
    else:
        lengthscale_prior = None
    return MaternKernel(nu=0.5, lengthscale_prior=lengthscale_prior, ard_num_dims=len(X.T))

def get_matern32_kernel(X, **hyper_prior_params):
    if hyper_prior_params:
        alpha = tensor(hyper_prior_params["mean"]).float()
        beta = tensor(hyper_prior_params["sigma"]).float()
        lengthscale_prior = GammaPrior(alpha, beta)
    else:
        lengthscale_prior = None
    return MaternKernel(nu=1.5, lengthscale_prior=lengthscale_prior, ard_num_dims=len(X.T))

def get_matern52_kernel(X, **hyper_prior_params):
    if hyper_prior_params:
        alpha = tensor(np.square(hyper_prior_params["mean"]) / np.square(hyper_prior_params["sigma"])).float()
        beta = tensor(hyper_prior_params["mean"] / np.square(hyper_prior_params["sigma"])).float()
        lengthscale_prior = GammaPrior(2, 2)
    else:
        lengthscale_prior = None
    return MaternKernel(nu=2.5, lengthscale_prior=lengthscale_prior)

# Squared Exponential Kernel
def get_squared_exponential_kernel(X, interpolation=False, **hyper_prior_params):
    # You can set hyper-prior with prior object in GPyTorch
    if hyper_prior_params:
        alpha = np.square(hyper_prior_params["mean"]) / np.square(hyper_prior_params["sigma"])
        beta = np.square(hyper_prior_params["mean"]) / np.square(hyper_prior_params["sigma"])
        lengthscale_prior = GammaPrior(alpha, beta)
    else:
        lengthscale_prior = None
    return RBFKernel(lengthscale_prior=lengthscale_prior)

# Base Kernels
def get_base_kernels(X, kernel="linear", ARD=True, **hyper_prior_params):
    if kernel == "linear":
        base_kernels = [LinearKernel() for item in range(X.shape[1])]
    elif kernel == "matern32":
        base_kernels = [get_matern32_kernel(X, **hyper_prior_params) for item in range(X.shape[1])]
    elif kernel == "matern52":
        base_kernels = [get_matern52_kernel(X, **hyper_prior_params) for item in range(X.shape[1])]
    return base_kernels

def wrap_scale_kernel(base_kernel, **scale_prior_params):
    outscale_prior = HalfCauchyPrior(scale=1) if scale_prior_params else None
    return ScaleKernel(base_kernel=base_kernel, outputscale_prior=outscale_prior)

# Additive Kernel
def get_additive_kernel(*kernels):
    return sum(kernels)

# Additive Kernel Permutation 2nd order structure
def additive_kernel_permutation(items, k=3):
    import itertools
    permutations = [list(p) for p in itertools.combinations(items, r=k)]
    print(f"Start building additive kernel. \n Calculated {len(permutations)}.")
    start = time.time()
    additive_kernel = sum([kl * kr for p, permutation in enumerate(permutations) for kl, kr in itertools.combinations(permutation, 2)])
    end = time.time()
    print(f"Finished building additive kernel. \n Time elapsed: {end - start:.2f}s")
    return additive_kernel

def additive_structure_kernel(X, base_kernels, interpolation=False, **scale_prior_params):
    import itertools
    kernel_triple = [list(p) for p in itertools.combinations(base_kernels, r=3)] # (n over k) in size
    outscale_prior = HalfCauchyPrior(scale=1) if scale_prior_params else None
    d_kernels = [ScaleKernel(ProductKernel(kernels[0], kernels[1]), outputscale_prior=outscale_prior) for p, permutation in enumerate(kernel_triple) for c, kernels in itertools.combinations(permutation, 2)] # k * (n over k) in size
    dim_tuples = ((kl, kr) for p, permutation in enumerate(kernel_triple) for c, (kl, kr) in itertools.combinations(permutation, 2))
    if interpolation:
        grid_size = grid.choose_grid_size(X, kronecker_structure=False)
        wrapper_kernels = ScaleKernel(GridInterpolationKernel(base_kernel=d_kernels, grid_size=2**len(X)), outputscale_prior=outscale_prior)
        return AdditiveStructureKernel(base_kernel=wrapper_kernels, num_dims=len(X))
    else:
        return AdditiveStructureKernel(base_kernel=d_kernels, num_dims=len(X))
    
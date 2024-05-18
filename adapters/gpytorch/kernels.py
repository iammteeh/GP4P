import torch
import gpytorch.kernels as gk
from gpytorch.utils import grid
from torch import tensor
from domain.env import POLY_DEGREE
from gpytorch.kernels import (
    Kernel, GridInterpolationKernel, ScaleKernel, ProductKernel, AdditiveStructureKernel, LinearKernel, RBFKernel, PeriodicKernel, MaternKernel,
    PolynomialKernel, PiecewisePolynomialKernel, SpectralMixtureKernel, RFFKernel)
from gpytorch.priors import Prior
from torch.distributions import constraints, HalfCauchy
from gpytorch.priors import LogNormalPrior, GammaPrior, HalfCauchyPrior
import numpy as np
import time

# Linear Kernel
def get_linear_kernel(X):
    return LinearKernel()

# Periodic Kernel
def get_periodic_kernel(X, ARD=True):
    periodic_legthscale_prior = HalfCauchyPrior(scale=0.1)
    if ARD:
        return PeriodicKernel(period_length_prior=periodic_legthscale_prior, ard_num_dims=len(X.T))
    else:
        return PeriodicKernel(period_length_prior=periodic_legthscale_prior)

def get_polynomial_kernel(X, ARD=True):
    offset_prior = LogNormalPrior(0, 1)
    if ARD:
        return PolynomialKernel(power=POLY_DEGREE, offset_prior=offset_prior, ard_num_dims=len(X.T))
    else:
        return PolynomialKernel(power=POLY_DEGREE, offset_prior=offset_prior)

def get_piecewise_polynomial_kernel(X, ARD=True):
    if ARD:
        return PiecewisePolynomialKernel(power=POLY_DEGREE, ard_num_dims=len(X.T))
    else:
        return PiecewisePolynomialKernel(power=POLY_DEGREE)

# Squared Exponential Kernel
def get_squared_exponential_kernel(X, ARD=True):
    # You can set hyper-prior with prior object in GPyTorch
    lengthscale_prior = HalfCauchyPrior(scale=0.1)
    if ARD:
        return RBFKernel(lengthscale_prior=lengthscale_prior, ard_num_dims=len(X.T))
    else:
        return RBFKernel(lengthscale_prior=lengthscale_prior)

def get_matern32_kernel(X, ARD=True):
    lengthscale_prior = HalfCauchyPrior(scale=0.1)
    if ARD:
        return MaternKernel(nu=1.5, lengthscale_prior=lengthscale_prior, ard_num_dims=len(X.T))
    else:
        return MaternKernel(nu=1.5, lengthscale_prior=lengthscale_prior)

def get_matern52_kernel(X, ARD=True):
    lengthscale_prior = HalfCauchyPrior(scale=0.1)
    if ARD:
        return MaternKernel(nu=2.5, lengthscale_prior=lengthscale_prior, ard_num_dims=len(X.T))
    else:    
        return MaternKernel(nu=2.5, lengthscale_prior=lengthscale_prior)
    
def get_rff_kernel(X, ARD=True):
    lengthscale_prior = HalfCauchyPrior(scale=0.1)
    if ARD:
        return RFFKernel(num_samples=len(X.T), lengthscale_prior=lengthscale_prior, ard_num_dims=len(X.T))
    else:
        return RFFKernel(num_samples=len(X.T), lengthscale_prior=lengthscale_prior)

def get_spectral_mixture_kernel(X, ARD=True):
    if ARD:
        return SpectralMixtureKernel(num_mixtures=len(X.T), ard_num_dims=len(X.T))
    else:
        return SpectralMixtureKernel(num_mixtures=len(X.T))


# Base Kernels
def get_base_kernels(X, kernel="linear", ARD=False):
    if kernel == "polynomial":
        base_kernels = [get_polynomial_kernel(X, ARD=ARD) for item in range(X.shape[1])]
    elif kernel == "piecewise_polynomial":
        base_kernels = [get_piecewise_polynomial_kernel(X, ARD=ARD) for item in range(X.shape[1])]
    elif kernel == "RBF":
        base_kernels = [get_squared_exponential_kernel(X, ARD=ARD) for item in range(X.shape[1])]
    elif kernel == "matern32":
        base_kernels = [get_matern32_kernel(X, ARD=ARD) for item in range(X.shape[1])]
    elif kernel == "matern52":
        base_kernels = [get_matern52_kernel(X, ARD=ARD) for item in range(X.shape[1])]
    elif kernel == "RFF":
        base_kernels = [get_rff_kernel(X, ARD=ARD) for item in range(X.shape[1])]
    elif kernel == "spectral_mixture":
        base_kernels = [get_spectral_mixture_kernel(X, ARD=ARD) for item in range(X.shape[1])]
    return base_kernels

def wrap_scale_kernel(base_kernel):
    outscale_prior = GammaPrior(2, 0.15)
    return ScaleKernel(base_kernel=base_kernel, outputscale_prior=outscale_prior)

# Additive Kernel
def get_additive_kernel(kernels):
    """
    takes the first out of a list of kernels and sums over it
    """
    additive_kernel = kernels[0]
    for kernel in kernels[1:]:
        additive_kernel += kernel
    return len(kernels), additive_kernel


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
    outscale_prior = HalfCauchyPrior(scale=1) if scale_prior_params else None # gimmick
    if interpolation:
        grid_size = grid.choose_grid_size(X, kronecker_structure=False)
        d_kernels = [ScaleKernel(GridInterpolationKernel(ProductKernel(k1,k2), grid_size=int(grid_size), num_dims=1, active_dims=[i,j]), outputscale_prior=outscale_prior, num_dims=2, ard_num_dims=2) for (i,k1),(j,k2)  in itertools.combinations(enumerate(base_kernels), 2)] # k * (n over k) in size
    else:
        d_kernels = [ScaleKernel(ProductKernel(k1,k2), outputscale_prior=outscale_prior, num_dims=1, active_dims=[i,j], ard_num_dims=2) for (i,k1),(j,k2)  in itertools.combinations(enumerate(base_kernels), 2)] # k * (n over k) in size
    num_dims, d_kernels = get_additive_kernel(d_kernels)
    return AdditiveStructureKernel(base_kernel=d_kernels, num_dims=num_dims)
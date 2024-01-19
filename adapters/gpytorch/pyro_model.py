from typing import Dict, Tuple
from itertools import combinations
from gpytorch.constraints import GreaterThan
from botorch.models.fully_bayesian import SaasPyroModel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.kernels import Kernel, ScaleKernel, ProductKernel, PiecewisePolynomialKernel, PolynomialKernel, SpectralMixtureKernel, MaternKernel, PeriodicKernel, RBFKernel, InducingPointKernel, AdditiveStructureKernel
from adapters.gpytorch.kernels import get_base_kernels, get_additive_kernel
from domain.env import KERNEL_TYPE, KERNEL_STRUCTURE ,POLY_DEGREE
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means.mean import Mean
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from torch import Tensor, Size

def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    # Detach and clone the new_value tensor
    detached_value = new_value.detach().clone()

    # Check if the new_value is a scalar or has a single element
    if detached_value.numel() == 1:
        # If target is also a scalar, use the scalar value directly
        if target.numel() == 1:
            return detached_value.item()
        else:
            # Otherwise, expand the scalar to match the target shape
            return detached_value.expand(target.shape)
    else:
        # For non-scalar values, ensure the shape matches the target
        return detached_value.view(target.shape).to(target.device)

class SaasPyroModel(SaasPyroModel):
    def load_covar_module(self, kernel_type=KERNEL_TYPE, kernel_structure=KERNEL_STRUCTURE, **kwargs):
        self.kernel_type = kernel_type
        self.kernel_structure = kernel_structure

        if kernel_structure == "additive":
            base_kernels = get_base_kernels(self.train_X, kernel=kernel_type)
            d_kernels = [ScaleKernel(ProductKernel(k1,k2), num_dims=1, active_dims=[i,j], ard_num_dims=2, batch_shape=kwargs["batch_shape"]) for (i,k1),(j,k2) in combinations(enumerate(base_kernels), 2)] # k * (n over k) in size
            num_dims, d_kernels = get_additive_kernel(d_kernels)
            return AdditiveStructureKernel(num_dims=num_dims, base_kernel=d_kernels)
        
        # SpectralMixtureKernel not supported for now
        #if kernel_type == "spectral_mixture":
        #    return SpectralMixtureKernel(num_mixtures=len(self.train_X.T), ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"])
        
        elif kernel_type == "piecewise_polynomial":
            return ScaleKernel(
                base_kernel=PiecewisePolynomialKernel(q=POLY_DEGREE, ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif kernel_type == "matern52": # candidate kernel
            return ScaleKernel(
                base_kernel=MaternKernel(nu=2.5, ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif kernel_type == "periodic":
            return ScaleKernel(
                base_kernel=PeriodicKernel(ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif kernel_type == "rbf":
            return ScaleKernel(
                base_kernel=RBFKernel(ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        else:
            raise NotImplementedError(f"Unknown kernel type: {kernel_type}" )
        
    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module (PiecewisePolynomial), and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = self.load_covar_module(kernel_type=KERNEL_TYPE, ard_num_dims=self.ard_num_dims, batch_shape=batch_shape)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        if self.kernel_structure == "additive":
            lengthscale_squeezed = mcmc_samples['lengthscale'].squeeze(-1)
            # Iterate over all pairs of dimensions (d over 2)
            dimension_pairs = list(combinations(range(len(self.train_X.T)), 2))

            for i, scale_kernel in enumerate(covar_module.base_kernel.kernels):
                # Get the indices for the current pair of dimensions
                dim1, dim2 = dimension_pairs[i]

                for j, base_kernel in enumerate(scale_kernel.base_kernel.kernels):
                    # Select the appropriate lengthscale value
                    lengthscale_value = lengthscale_squeezed[dim1, dim2] if j == 0 else lengthscale_squeezed[dim2, dim1]
                    base_kernel.lengthscale = reshape_and_detach(
                        target=base_kernel.lengthscale,
                        new_value=lengthscale_value,
                    )

                # Update outputscale for each ScaleKernel, if needed
                if 'outputscale' in mcmc_samples:
                    scale_kernel.outputscale = reshape_and_detach(
                        target=scale_kernel.outputscale,
                        new_value=mcmc_samples['outputscale'],
                    )
        else:
            covar_module.base_kernel.lengthscale = reshape_and_detach(
                target=covar_module.base_kernel.lengthscale,
                new_value=mcmc_samples["lengthscale"],
            )
            covar_module.outputscale = reshape_and_detach(
                target=covar_module.outputscale,
                new_value=mcmc_samples["outputscale"],
            )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood
from typing import Dict, Tuple
from gpytorch.constraints import GreaterThan
from botorch.models.fully_bayesian import SaasPyroModel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.kernels import Kernel, ScaleKernel, PiecewisePolynomialKernel, SpectralMixtureKernel
from domain.env import KERNEL_TYPE
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means.mean import Mean
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from torch import Tensor, Size

def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    """Detach and reshape `new_value` to match `target`."""
    return new_value.detach().clone().view(target.shape).to(target)

class PolySaasPyroModel(SaasPyroModel):
    def load_covar_module(kernel_type=KERNEL_TYPE, **kwargs):
        if kernel_type == "spectral_mixture":
            return SpectralMixtureKernel()
        
        elif kernel_type == "piecewise_polynomial":
            return ScaleKernel(
                base_kernel=PiecewisePolynomialKernel(**kwargs),
            )
        else:
            raise NotImplementedError(f"Unknown kernel type: {kernel_type}" )
        
    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module (PiecewisePolynomial), and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=PiecewisePolynomialKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
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
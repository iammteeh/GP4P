from typing import Dict, Tuple, Union
from itertools import combinations
import torch
import jax.numpy as jnp
import numpy as np
from gpytorch.constraints import GreaterThan
from botorch.models.fully_bayesian import SaasPyroModel
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
import pyro
from pyro.infer.mcmc import MCMC, NUTS
from pyro.contrib.gp.kernels import RBF, RationalQuadratic, Exponential, Matern32, Matern52, Periodic, Polynomial, Sum, Product
from adapters.pyro.kernels import polynomial_kernel, rbf_kernel, matern_kernel, AdditivePyroKernel
from gpytorch.means.linear_mean import LinearMean
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.kernels import Kernel, ScaleKernel, ProductKernel, PiecewisePolynomialKernel, SpectralMixtureKernel, MaternKernel, PeriodicKernel, RBFKernel, RFFKernel ,InducingPointKernel, AdditiveStructureKernel
from botorch.models.fully_bayesian import matern52_kernel, compute_dists
from adapters.gpytorch.polynomial_kernel import PolynomialKernel
from adapters.gpytorch.kernels import get_base_kernels, get_additive_kernel
from domain.env import MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE ,POLY_DEGREE
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means.mean import Mean
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from torch import Tensor, Size
from pyro.infer.autoguide import init_to_feasible, init_to_median, init_to_sample

def convert_to_tensors(data_dict):
    tensor_dict = {}
    for key, value in data_dict.items():
        # Check if the value is array-like
        if isinstance(value, (list, tuple, np.ndarray, jnp.ndarray)):
            # Convert to NumPy array if it's not already one
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            tensor_dict[key] = torch.tensor(value)
        else:
            # If it's a scalar, convert directly
            tensor_dict[key] = torch.tensor(value)
    return tensor_dict

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
    
def fit_fully_bayesian_model_nuts(
    model: Union[SaasFullyBayesianSingleTaskGP, SaasFullyBayesianMultiTaskGP],
    max_tree_depth: int = 10,
    warmup_steps: int = 512,
    num_samples: int = 256,
    thinning: int = 16,
    disable_progbar: bool = False,
    jit_compile: bool = True,
) -> None:
    r"""Fit a fully Bayesian model using the No-U-Turn-Sampler (NUTS)


    Args:
        model: SaasFullyBayesianSingleTaskGP to be fitted.
        max_tree_depth: Maximum tree depth for NUTS
        warmup_steps: The number of burn-in steps for NUTS.
        num_samples:  The number of MCMC samples. Note that with thinning,
            num_samples / thinning samples are retained.
        thinning: The amount of thinning. Every nth sample is retained.
        disable_progbar: A boolean indicating whether to print the progress
            bar and diagnostics during MCMC.
        jit_compile: Whether to use jit. Using jit may be ~2X faster (rough estimate),
            but it will also increase the memory usage and sometimes result in runtime
            errors, e.g., https://github.com/pyro-ppl/pyro/issues/3136.

    Example:
        >>> gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
    """
    model.train()

    # Do inference with NUTS
    nuts = NUTS(
        model.pyro_model.sample,
        jit_compile=jit_compile,
        step_size=1,
        full_mass=True,
        ignore_jit_warnings=True,
        max_tree_depth=max_tree_depth,
        target_accept_prob= 0.8,
        init_strategy=init_to_feasible,
    )
    mcmc = MCMC(
        nuts,
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        disable_progbar=disable_progbar,
        disable_validation=False
    )
    mcmc.run()

    # Get final MCMC samples from the Pyro model
    mcmc_samples = model.pyro_model.postprocess_mcmc_samples(
        mcmc_samples=mcmc.get_samples()
    )
    for k, v in mcmc_samples.items():
        mcmc_samples[k] = v[::thinning]

    # Load the MCMC samples back into the BoTorch model
    model.load_mcmc_samples(mcmc_samples)
    model.eval()

    return mcmc_samples


class SaasPyroModel(SaasPyroModel):
    def __init__(self, mean_func=MEAN_FUNC, kernel_structure=KERNEL_STRUCTURE, kernel_type=KERNEL_TYPE):
        self.mean_func = mean_func
        self.kernel_type = kernel_type
        self.kernel_structure = kernel_structure

    def init_kernel(self, outputscale=None, lengthscale=None, inv_length_sq=None):
        r"""Initialize the kernel based on the kernel type and structure."""
        if "poly" in self.kernel_type:
            if self.kernel_type == "poly2" or self.kernel_type == "polynomial":
                    POLY_DEGREE = 2
            elif self.kernel_type == "poly3":
                    POLY_DEGREE = 3
            elif self.kernel_type == "poly4":
                    POLY_DEGREE = 4
            return polynomial_kernel(self.train_X, self.train_X, degree=POLY_DEGREE)
        elif self.kernel_type == "rbf":
            return rbf_kernel(self.train_X, self.train_X, lengthscale=lengthscale)
        elif self.kernel_type == "matern32":
            return matern_kernel(self.train_X, self.train_X, inv_length_sq, nu=1.5)
        elif self.kernel_type == "matern52":
            return matern52_kernel(self.train_X, lengthscale)
        elif self.kernel_type == "periodic":
            kernel = Periodic(input_dim=self.train_X.shape[1], lengthscale=lengthscale)
            return kernel.forward(self.train_X)
            # own implementations don't produce positive definite matrices yet
            return periodic_kernel(self.train_X, self.train_X, lengthscale=lengthscale, period=1.0)
            return _periodic_kernel(self.train_X, self.train_X, lengthscale=lengthscale, period=1.0)
        # not yet by GPyTorch supported kernels
        #elif self.kernel_type == "rational":
        #    kernel = RationalQuadratic(input_dim=self.train_X.shape[1], lengthscale=lengthscale)
        #    return kernel.forward(self.train_X)
        #elif self.kernel_type == "exponential":
        #    kernel = Exponential(input_dim=self.train_X.shape[1], lengthscale=lengthscale)
        #    return kernel.forward(self.train_X)

    def init_additive_base_kernel(self, i, j, active_dims, outputscale=None, lengthscale=None, inv_length_sq=None):
        r"""Initialize the kernel based on the kernel type and structure."""
        if "poly" in self.kernel_type:
            if self.kernel_type == "poly2" or self.kernel_type == "polynomial":
                    POLY_DEGREE = 2
            elif self.kernel_type == "poly3":
                    POLY_DEGREE = 3
            elif self.kernel_type == "poly4":
                    POLY_DEGREE = 4
            return polynomial_kernel(self.train_X[i, active_dims], self.train_X[j, active_dims], degree=POLY_DEGREE)
        elif self.kernel_type == "rbf":
            return rbf_kernel(self.train_X[i, active_dims], self.train_X[j, active_dims], lengthscale=lengthscale)
        elif self.kernel_type == "matern32":
            return matern_kernel(self.train_X[i, active_dims], self.train_X[j, active_dims], inv_length_sq, nu=1.5)
        elif self.kernel_type == "matern52":
            return matern_kernel(self.train_X[i, active_dims], self.train_X[j, active_dims], inv_length_sq, nu=2.5)
        elif self.kernel_type == "periodic":
            raise NotImplementedError("Periodic kernel not supported for additive structure.")

    def get_base_kernel(self, X, Z, kernel_type, dim=None, **kwargs):
        if dim is None:
            name_suffix = ""
        else:
            name_suffix = f"_{dim}"
        # sample hyperpriors for base_kernel
        if kernel_type == "polynomial" or kernel_type == "poly2":
            return polynomial_kernel(X, Z, degree=2)
        elif kernel_type == "poly3":
            return polynomial_kernel(X, Z, degree=3)
        elif kernel_type == "poly4":
            return polynomial_kernel(X, Z, degree=4)
        elif kernel_type == "rbf":
            inverse_lengthscale, lengthscale = SaasPyroModel.sample_lengthscale(dim=2)
            return rbf_kernel(X, Z, lengthscale=lengthscale)
        elif kernel_type == "matern32":
            inverse_lengthscale, lengthscale = SaasPyroModel.sample_lengthscale(dim=2)
            return matern_kernel(X, Z, inv_length_sq=inverse_lengthscale, nu=1.5)
        elif kernel_type == "matern52":
            inverse_lengthscale, lengthscale = SaasPyroModel.sample_lengthscale(dim=1, name_suffix=name_suffix)
            #return outputscale * matern_kernel(X, Z, inv_length_sq=inverse_lengthscale, nu=2.5)
            kernel = MaternKernel(nu=2.5)
            kernel.lengthscale = lengthscale
            return kernel
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """

        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        if self.kernel_structure != "additive":
            outputscale = SaasPyroModel.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
            mean = SaasPyroModel.sample_mean(**tkwargs)
            noise = self.sample_noise(**tkwargs)
            inverse_lengthscale, lengthscale = SaasPyroModel.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
            kernel = self.init_kernel(outputscale=outputscale, lengthscale=lengthscale, inv_length_sq=inverse_lengthscale)
            kernel = outputscale * kernel + noise * torch.eye(self.train_X.shape[0], **tkwargs)
            pyro.sample(
                "Y",
                pyro.distributions.MultivariateNormal(
                    loc=mean.view(-1).expand(self.train_X.shape[0]),
                    covariance_matrix=kernel,
                ),
                obs=self.train_Y.squeeze(-1),
            )
        else:
            base_kernels = [self.get_base_kernel(self.train_X, self.train_X, kernel_type=self.kernel_type, dim=item) for item in range(self.train_X.shape[1])]
            d_kernels = []
            print(f"Building additive kernel with {len(base_kernels)} base kernels.")
            for (i,k1),(j,k2) in combinations(enumerate(base_kernels), 2):
                name = f"outputscale_{i}_{j}"
                outputscale = SaasPyroModel.sample_outputscale(concentration=2.0, rate=0.15, name=name, **tkwargs)
                scaled_kernel = ScaleKernel(ProductKernel(k1,k2), num_dims=1, active_dims=[i,j], ard_num_dims=2)
                ScaleKernel.outputscale = outputscale
                d_kernels.append(scaled_kernel)

            print(f"Evaluating additive kernel with {len(d_kernels)} base kernels.")
            K = AdditivePyroKernel(base_kernels=d_kernels).forward(self.train_X, self.train_X)
            mean = self.sample_mean(**tkwargs)
            pyro.sample(
                "Y",
                pyro.distributions.MultivariateNormal(
                    loc=mean.view(-1).expand(self.train_X.shape[0]),
                    covariance_matrix=K,
                ),
                obs=self.train_Y.squeeze(-1),
            )

    @classmethod
    def sample_outputscale(
        cls, concentration: float = 2.0, rate: float = 0.15, name="outputscale", **tkwargs
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            name,
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    @classmethod
    def sample_mean(cls, **tkwargs) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return MIN_INFERRED_NOISE_LEVEL + pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(0.9, **tkwargs),
                    torch.tensor(10.0, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    @classmethod
    def sample_lengthscale(
        cls, dim: int, alpha: float = 0.1, name_suffix="", **tkwargs
    ) -> Tensor:
        r"""Sample the lengthscale."""
        tausq = pyro.sample(
            "kernel_tausq" + name_suffix,
            pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
        )
        inv_length_sq = pyro.sample(
            "_kernel_inv_length_sq" + name_suffix,
            pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
        )
        inv_length_sq = pyro.deterministic(
            "kernel_inv_length_sq" + name_suffix, tausq * inv_length_sq
        )
        lengthscale = pyro.deterministic(
            "lengthscale" + name_suffix,
            inv_length_sq.rsqrt(),
        )
        return inv_length_sq, lengthscale

    def load_covar_module(self, **kwargs):
        r"""Load the covariance module based on the kernel type and structure."""
        if self.kernel_structure == "additive":
            base_kernels = get_base_kernels(self.train_X, kernel=self.kernel_type)
            d_kernels = [ScaleKernel(ProductKernel(k1,k2), num_dims=1, active_dims=[i,j], ard_num_dims=2, batch_shape=kwargs["batch_shape"]) for (i,k1),(j,k2) in combinations(enumerate(base_kernels), 2)] # k * (n over k) in size
            num_dims, d_kernels = get_additive_kernel(d_kernels)
            return AdditiveStructureKernel(num_dims=num_dims, base_kernel=d_kernels)
        
        # SpectralMixtureKernel not supported for now
        #if kernel_type == "spectral_mixture":
        #    return SpectralMixtureKernel(num_mixtures=len(self.train_X.T), ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"])
        
        elif self.kernel_type == "piecewise_polynomial":
            return ScaleKernel(
                base_kernel=PiecewisePolynomialKernel(q=POLY_DEGREE, ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif self.kernel_type == "polynomial" or self.kernel_type == "poly2" or self.kernel_type == "poly3" or self.kernel_type == "poly4":
            # workaround for polynomial kernel
            if self.kernel_type == "poly2" or self.kernel_type == "polynomial":
                POLY_DEGREE = 2
            elif self.kernel_type == "poly3":
                POLY_DEGREE = 3
            elif self.kernel_type == "poly4":
                POLY_DEGREE = 4
            return ScaleKernel(
                base_kernel=PolynomialKernel(power=POLY_DEGREE, ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif self.kernel_type == "matern32":
            return ScaleKernel(
                base_kernel=MaternKernel(nu=1.5, ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif self.kernel_type == "matern52": # candidate kernel
            return ScaleKernel(
                base_kernel=MaternKernel(nu=2.5, ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif self.kernel_type == "periodic":
            return ScaleKernel(
                base_kernel=PeriodicKernel(ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif self.kernel_type == "rbf" or self.kernel_type == "RBF":
            return ScaleKernel(
                base_kernel=RBFKernel(ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        elif self.kernel_type == "RFF":
            return ScaleKernel(
                base_kernel=RFFKernel(num_samples=kwargs["ard_num_dims"], ard_num_dims=kwargs["ard_num_dims"], batch_shape=kwargs["batch_shape"]),
            batch_shape=kwargs["batch_shape"]
            )
        else:
            raise NotImplementedError(f"Unknown kernel type: {self.kernel_type}" )
        
    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module (PiecewisePolynomial), and likelihood."""
        # check if the samples are tensors
        if not all([isinstance(v, Tensor) for v in mcmc_samples.values()]):
            mcmc_samples = convert_to_tensors(mcmc_samples)
        tkwargs = {"device": torch.tensor(np.array(self.train_X)).device, "dtype": torch.tensor(np.array(self.train_X)).dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = Size([num_mcmc_samples])
        #TODO: Implement sampling the linear weighted mean
        #if self.mean_func == "linear_weighted":
        #    mean_module = LinearMean(input_size=len(self.train_X.T), batch_shape=batch_shape).to(**tkwargs)
        #elif self.mean_func == "constant":
        #    mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        #else:
        #    raise NotImplementedError(f"Mean has to be constant.")
        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = self.load_covar_module(ard_num_dims=self.ard_num_dims, batch_shape=batch_shape)
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
            lengthscales_squeezed = [mcmc_samples[key].squeeze(-1) for i, key in enumerate(mcmc_samples) if f'lengthscale_{i}' in mcmc_samples.keys()]
            #outputscales_squeezed = [mcmc_samples[key].squeeze(-1) for (i, j), key in zip(mcmc_samples, mcmc_samples) if f'outputscale' in mcmc_samples.keys()]
            # dict for outputscale
            outputscales_squeezed = {key: mcmc_samples[key].squeeze(-1) for key in mcmc_samples if 'outputscale' in key}
            # Iterate over all pairs of dimensions (d over 2)
            if len(self.train_X.T) <= mcmc_samples['lengthscale_0'].shape[0]:
                dims = len(self.train_X.T)
            elif len(self.train_X.T) > mcmc_samples['lengthscale_0'].shape[0]:
                dims = mcmc_samples['lengthscale_0'].shape[0]
            dimension_pairs = list(combinations(range(dims), 2))

            #for i, scale_kernel in enumerate(covar_module.base_kernel.kernels):
            for (i, j) in dimension_pairs:
                # Get the indices for the current pair of dimensions
                if i == len(dimension_pairs):
                    break
                dim1, dim2 = dimension_pairs[i]

                #for j, base_kernel in enumerate(scale_kernel.base_kernel.kernels):
                for scale_kernel in covar_module.base_kernel.kernels:
                    # Select the appropriate lengthscale value
                    for k, base_kernel in enumerate(scale_kernel.base_kernel.kernels):
                        lengthscale_value = lengthscales_squeezed[dim1] if k == 0 else lengthscales_squeezed[dim2]

                        base_kernel.lengthscale = reshape_and_detach(
                            target=base_kernel.lengthscale,
                            new_value=lengthscale_value.median()
                        )
                    # Select the according outputscale value
                    outputscale_value = outputscales_squeezed[f"outputscale_{i}_{j}"]
                    scale_kernel.outputscale = reshape_and_detach(
                        target=scale_kernel.outputscale,
                        new_value=outputscale_value
                    )

        elif "poly" in self.kernel_type:
            covar_module.outputscale = reshape_and_detach(
                target=covar_module.outputscale,
                new_value=mcmc_samples["outputscale"],
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

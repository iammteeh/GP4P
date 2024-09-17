import time
from itertools import combinations
import jax.numpy as jnp
import jax.random as random
from jax.numpy import clip
from jax import jit, vmap
from jax.lax import clamp
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from numpyro.diagnostics import summary
from gpytorch.constraints import GreaterThan

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

from adapters.pyro.pyro_model import SaasPyroModel
from gpytorch.means.constant_mean import ConstantMean
from adapters.pyro.jax_kernels import rbf_kernel, matern_kernel, polynomial_kernel, AdditiveJAXKernel
from numpyro.infer import MCMC, NUTS

from typing import Dict, Tuple, Union
from jax.typing import ArrayLike
from gpytorch.means.mean import Mean
from gpytorch.likelihoods import Likelihood, FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.kernels import Kernel

MIN_INFERRED_NOISE_LEVEL = 1e-4

def reshape_and_detach(target, new_value) -> None:
    # Detach and clone the new_value tensor
    detached_value = new_value.copy()

    # Check if the new_value is a scalar or has a single element
    if detached_value.shape == 1:
        # If target is also a scalar, use the scalar value directly
        if target.shape == 1:
            return detached_value.item()
        else:
            # Otherwise, expand the scalar to match the target shape
            return jnp.broadcast_to(detached_value, target.shape)
    else:
        # For non-scalar values, ensure the shape matches the target
        return jnp.reshape(detached_value, target.shape)

def fit_fully_bayesian_model_nuts(
    model: Union[SaasFullyBayesianSingleTaskGP, SaasFullyBayesianMultiTaskGP],
    max_tree_depth: int = 10,
    warmup_steps: int = 512,
    num_samples: int = 256,
    thinning: int = 16,
    progress_bar: bool = True,
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
    rng_key_hmc, rng_key_predict = random.split(random.PRNGKey(42), 2)

    # Do inference with NUTS
    nuts = NUTS(
        model.pyro_model.sample,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=warmup_steps,
        num_samples=num_samples,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key_hmc)

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


class SaasPyroModelJAX(SaasPyroModel):
    """
    This class contains the necessary modeling and inference code to fit a gaussian process with a SAAS prior.

    See below for arguments.
    """

    def __init__(
        self,
        mean_func="constant",
        kernel_type="matern",
        kernel_structure="simple",
        alpha=0.1,  # controls sparsity
        observation_variance=0.0,  # observation variance to use; this scalar value is inferred if observation_variance==0.0
    ):
        if alpha <= 0.0:
            raise ValueError("The hyperparameter alpha should be positive.")
        if observation_variance < 0.0:
            raise ValueError("The hyperparameter observation_variance should be non-negative.")
        
        self.mean_func = mean_func
        self.kernel_type = kernel_type
        self.kernel_structure = kernel_structure
        self.alpha = alpha
        self.kernel = rbf_kernel if kernel_type == "rbf" else matern_kernel
        self.observation_variance = observation_variance
        self.learn_noise = observation_variance == 0.0
        self.Ls = None

    def set_inputs(
        self, train_X, train_Y, train_Yvar = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        # transform x and y to jnp arrays
        self.train_X = jnp.array(self.train_X)
        self.train_Y = jnp.array(self.train_Y).reshape(-1, 1)
        self.ard_num_dims = self.train_X.shape[-1]

    def init_kernel(self, outputscale=None, lengthscale=None, inv_length_sq=None, noise=None):
        r"""Initialize the kernel based on the kernel type and structure."""
        if self.kernel_type == "poly2" or self.kernel_type == "polynomial":
            return polynomial_kernel(self.train_X, self.train_X, degree=2, var=outputscale, noise=noise)
        elif self.kernel_type == "poly3":
            return polynomial_kernel(self.train_X, self.train_X, degree=3, var=outputscale, noise=noise)
        elif self.kernel_type == "poly4":
            return polynomial_kernel(self.train_X, self.train_X, degree=4, var=outputscale, noise=noise)
        elif self.kernel_type == "rbf" or self.kernel_type == "RBF":
            return rbf_kernel(self.train_X, self.train_X, inv_length_sq=inv_length_sq, var=outputscale, noise=noise)
        elif self.kernel_type == "matern32":
            return matern_kernel(self.train_X, self.train_X, inv_length_sq=inv_length_sq, var=outputscale, noise=noise, nu=1.5)
        elif self.kernel_type == "matern52":
            return matern_kernel(self.train_X, self.train_X, inv_length_sq=inv_length_sq, var=outputscale, noise=noise, nu=2.5)

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
        elif kernel_type == "rbf" or kernel_type == "RBF":
            inverse_lengthscale, lengthscale = self.sample_lengthscale(dim=1, name_suffix=name_suffix)
            return rbf_kernel(X, Z, var=1.0, inv_length_sq=inverse_lengthscale)
        elif kernel_type == "matern32":
            inverse_lengthscale, lengthscale = self.sample_lengthscale(dim=1, name_suffix=name_suffix)
            return matern_kernel(X, Z, inv_length_sq=inverse_lengthscale, nu=1.5)
        elif kernel_type == "matern52":
            inverse_lengthscale, lengthscale = self.sample_lengthscale(dim=1, name_suffix=name_suffix)
            return matern_kernel(X, Z, inv_length_sq=inverse_lengthscale, nu=2.5)
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")
    # define the surrogate model. users who want to modify e.g. the prior on the kernel variance
    # should make their modifications here.
    def sample(self):
        if self.kernel_structure != "additive":
            N, P = self.train_X.shape

            variance = self.sample_outputscale()
            mean = self.sample_mean()
            noise = (
                self.sample_noise() if self.learn_noise else self.observation_variance
            )
            inv_length_sq, lengthscale = self.sample_lengthscale(P)
            k = self.init_kernel(outputscale=variance, lengthscale=lengthscale, inv_length_sq=inv_length_sq, noise=noise)
            numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.broadcast_to(mean, self.train_X.shape[0]), covariance_matrix=k), obs=self.train_Y)
        else:
            base_kernels = [self.get_base_kernel(self.train_X, self.train_X, self.kernel_type, dim=item) for item in range(self.ard_num_dims)]
            d_kernels = []
            for (i, k1),(j, k2) in combinations(enumerate(base_kernels), 2): # iterate over all pairs of dimensions (d over 2)
                name = f"{i}_{j}"
                outputscale = self.sample_outputscale(name_suffix=f"_{name}")
                d_kernels.append(base_kernels[i] * base_kernels[j] * outputscale) # scaled product kernel
            K = AdditiveJAXKernel(base_kernels=d_kernels).forward(self.train_X, self.train_X)
            noise = self.sample_noise()
            K += jnp.eye(self.train_X.shape[0]) * noise
            mean = jnp.broadcast_to(self.sample_mean(), self.train_X.shape[0])
            numpyro.sample(f"Y", dist.MultivariateNormal(loc=mean, covariance_matrix=K), obs=self.train_Y)
    def sample_hyperparameters(self, rng_key, num_samples=1):
        pass

    def sample_outputscale(self, concentration: float = 2.0, rate: float = 0.15, name_suffix=""):
        return numpyro.sample("outputscale" + name_suffix, dist.Gamma(concentration, rate))
    
    def sample_mean(self):
        return numpyro.sample("mean", dist.Normal(0.0, 1.0))
    
    def sample_noise(self, name_suffix=""):
        return numpyro.sample("noise" + name_suffix, dist.Gamma(0.9, 10.0))
    
    def sample_lengthscale(self, dim: int, name_suffix=""):
        tausq = numpyro.sample("kernel_tausq" + name_suffix, dist.HalfCauchy(self.alpha)) # shrinkage prior

        # note we use deterministic to reparameterize the geometry
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq" + name_suffix, dist.HalfCauchy(jnp.ones(dim)))
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq" + name_suffix, tausq * inv_length_sq)

        lengthscale = numpyro.deterministic("lengthscale" + name_suffix, 1.0 / jnp.sqrt(inv_length_sq))

        return inv_length_sq, lengthscale
    
    def postprocess_mcmc_samples(self, mcmc_samples) -> dict:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        if self.kernel_structure == "simple":
            inv_length_sq = (
                jnp.expand_dims(mcmc_samples["kernel_tausq"], axis=-1)
                * mcmc_samples["_kernel_inv_length_sq"]
            )
            mcmc_samples["lengthscale"] = jnp.sqrt(1.0 / inv_length_sq)
            # Delete `kernel_tausq` and `_kernel_inv_length_sq` since they aren't loaded
            # into the final model.
            del mcmc_samples["kernel_tausq"], mcmc_samples["_kernel_inv_length_sq"]
        elif self.kernel_structure == "additive" and "poly" not in self.kernel_type:
            for i in range(self.ard_num_dims):
                inv_length_sq = (
                    jnp.expand_dims(mcmc_samples[f"kernel_tausq_{i}"], axis=-1)
                    * mcmc_samples[f"_kernel_inv_length_sq_{i}"]
                )
                mcmc_samples[f"lengthscale_{i}"] = jnp.sqrt(1.0 / inv_length_sq)
                del mcmc_samples[f"kernel_tausq_{i}"], mcmc_samples[f"_kernel_inv_length_sq_{i}"]
        else:
            pass
        return mcmc_samples
    
    def load_jax_mcmc_samples(self, mcmc_samples: Dict[str, ArrayLike]) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module (PiecewisePolynomial), and likelihood."""
        num_mcmc_samples = len(mcmc_samples["mean"])
        # get batch_shape of num_mcmc_samples
        batch_shape = [num_mcmc_samples]
        #TODO: Implement sampling the linear weighted mean
        #if self.mean_func == "linear_weighted":
        #    mean_module = LinearMean(input_size=len(self.train_X.T), batch_shape=batch_shape).to(**tkwargs)
        #elif self.mean_func == "constant":
        #    mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        #else:
        #    raise NotImplementedError(f"Mean has to be constant.")
        mean_module = ConstantMean(batch_shape=batch_shape)
        covar_module = self.load_covar_module(ard_num_dims=self.ard_num_dims, batch_shape=batch_shape)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            )
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            )
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"],
            )
        if self.kernel_structure == "additive":
            lengthscale_squeezed = mcmc_samples['lengthscale'].squeeze(-1)
            # Iterate over all pairs of dimensions (d over 2)
            if len(self.train_X.T) <= mcmc_samples['lengthscale'].shape[0]:
                dims = len(self.train_X.T)
            elif len(self.train_X.T) > mcmc_samples['lengthscale'].shape[0]:
                dims = mcmc_samples['lengthscale'].shape[0]
            dimension_pairs = list(combinations(range(dims), 2))

            for i, scale_kernel in enumerate(covar_module.base_kernel.kernels):
                # Get the indices for the current pair of dimensions
                if i == len(dimension_pairs):
                    break
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

    # run gradient-based NUTS MCMC inference
    def run_inference(self, rng_key, X, Y):
        start = time.time()
        kernel = NUTS(self.sample, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.verbose,
        )
        mcmc.run(rng_key, X, Y)

        flat_samples = mcmc.get_samples(group_by_chain=False)
        chain_samples = mcmc.get_samples(group_by_chain=True)
        flat_summary = summary(flat_samples, prob=0.90, group_by_chain=False)

        if self.verbose:
            rhat = flat_summary["kernel_inv_length_sq"]["r_hat"]
            print(
                "[kernel_inv_length_sq] r_hat min/max/median:  {:.3f}  {:.3f}  {:.3f}".format(
                    jnp.min(rhat), jnp.max(rhat), jnp.median(rhat)
                )
            )

            mcmc.print_summary(exclude_deterministic=False)
            print("\nMCMC elapsed time:", time.time() - start)

        return chain_samples, flat_samples, flat_summary
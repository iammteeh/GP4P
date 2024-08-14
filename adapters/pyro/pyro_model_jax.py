import math
import time
from functools import partial
from typing import Union
from itertools import combinations
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from adapters.pyro.pyro_model import SaasPyroModel
from adapters.pyro.jax_kernels import rbf_kernel, matern_kernel
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS


def reshape_and_detach(target, new_value) -> None:
    # Detach and clone the new_value tensor
    detached_value = new_value.copy()

    # Check if the new_value is a scalar or has a single element
    if detached_value.size() == 1:
        # If target is also a scalar, use the scalar value directly
        if target.size() == 1:
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

    # define the surrogate model. users who want to modify e.g. the prior on the kernel variance
    # should make their modifications here.
    def sample(self):
        N, P = self.train_X.shape

        outputscale = self.sample_outputscale()
        mean = self.sample_mean()
        noise = (
            self.sample_noise() if self.learn_noise else self.observation_variance
        )
        inv_length_sq, lengthscale = self.sample_lengthscale(P)

        k = self.kernel(self.train_X, self.train_X, outputscale, inv_length_sq, noise, True)
        numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.broadcast_to(mean, self.train_X.shape[0]), covariance_matrix=k), obs=self.train_Y)

    def sample_hyperparameters(self, rng_key, num_samples=1):
        pass

    def sample_outputscale(self, concentration: float = 2.0, rate: float = 0.15):
        return numpyro.sample("outputscale", dist.Gamma(concentration, rate))
    
    def sample_mean(self):
        return numpyro.sample("mean", dist.Normal(0.0, 1.0))
    
    def sample_noise(self):
        return numpyro.sample("noise", dist.Gamma(0.9, 10.0))
    
    def sample_lengthscale(self, dim: int):
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(self.alpha)) # shrinkage prior

        # note we use deterministic to reparameterize the geometry
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(dim)))
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)

        lengthscale = numpyro.deterministic("lengthscale", 1.0 / jnp.sqrt(inv_length_sq))

        return inv_length_sq, lengthscale
    
    def postprocess_mcmc_samples(self, mcmc_samples) -> dict:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        inv_length_sq = (
            jnp.expand_dims(mcmc_samples["kernel_tausq"], axis=-1)
            * mcmc_samples["_kernel_inv_length_sq"]
        )
        mcmc_samples["lengthscale"] = jnp.sqrt(1.0 / inv_length_sq)
        # Delete `kernel_tausq` and `_kernel_inv_length_sq` since they aren't loaded
        # into the final model.
        del mcmc_samples["kernel_tausq"], mcmc_samples["_kernel_inv_length_sq"]
        return mcmc_samples


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
                    np.min(rhat), np.max(rhat), np.median(rhat)
                )
            )

            mcmc.print_summary(exclude_deterministic=False)
            print("\nMCMC elapsed time:", time.time() - start)

        return chain_samples, flat_samples, flat_summary
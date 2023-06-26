from sklearn.base import BaseEstimator
from bayesify.pairwise import PyroMCMCRegressor, get_n_words, assert_ci
import arviz as az
import inspect
import numpy as np
from numpyro import sample, plate
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from pandas import DataFrame
from pprint import pprint
import jax.numpy as jnp
from sklearn.svm import SVC

class PyroMCMCRegressor(PyroMCMCRegressor, BaseEstimator):
    def __init__(self,
                mcmc_samples: int = 1000, 
                mcmc_tune: int = 1000, 
                n_chains=8,
                base_prior_dist="Normal",
                base_prior_loc=0,
                base_prior_scale=1,
                base_prior_concentration=1,
                base_prior_concentration1=1e+6,
                base_prior_concentration0=1e-6,
                base_prior_rate=1,
                base_prior_probs=0.5,
                infl_prior_dist="Normal",
                infl_prior_loc=0,
                infl_prior_scale=1,
                infl_prior_concentration=1,
                infl_prior_rate=1,
                infl_prior_probs=0.5,
                error_prior_dist="Normal",
                error_prior_loc=0,
                error_prior_scale=1,
                error_prior_concentration=1,
                error_prior_rate=1,
                error_prior_probs=0.5,
                #weight_errs_per_sample=None,
                #weighted_rel_errs_per_sample=None,
        ):

        self.base_prior_dist = base_prior_dist
        self.base_prior_loc = base_prior_loc
        self.base_prior_scale = base_prior_scale
        self.base_prior_concentration = base_prior_concentration
        self.base_prior_concentration1 = base_prior_concentration1
        self.base_prior_concentration0 = base_prior_concentration0
        self.base_prior_rate = base_prior_rate
        self.base_prior_probs = base_prior_probs
        self.infl_prior_dist = infl_prior_dist
        self.infl_prior_loc = infl_prior_loc
        self.infl_prior_scale = infl_prior_scale
        self.infl_prior_concentration = infl_prior_concentration
        self.infl_prior_rate = infl_prior_rate
        self.infl_prior_probs = infl_prior_probs
        self.error_prior_dist = error_prior_dist
        self.error_prior_loc = error_prior_loc
        self.error_prior_scale = error_prior_scale
        self.error_prior_concentration = error_prior_concentration
        self.error_prior_rate = error_prior_rate
        self.error_prior_probs = error_prior_probs
        #self.weight_errs_per_sample = weight_errs_per_sample
        #self.weighted_rel_errs_per_sample = weighted_rel_errs_per_sample

        super().__init__(mcmc_samples, mcmc_tune, n_chains)

        #self.base_prior, self.infl_prior, self.error_prior = self.generate_priors()
        #self.base_prior = dist.Normal(0,1)
        #self.infl_prior = dist.Normal(0,1)
        #self.error_prior = dist.Normal(0,1)

    def generate_priors(self):
        base_prior_method = self.base_prior_dist if self.base_prior_dist else random.choice(["Normal", "Gamma", "HalfCauchy", "Bernoulli", "Exponential", "Gumbel", "Laplace", "LogNormal", "StudentT", "Beta", "Weibull"])
        infl_prior_method = self.infl_prior_dist if self.infl_prior_dist else random.choice(["Normal", "Dirichlet", "Multinomial", "Binomial", "Poisson", "Gamma"])
        error_prior_method = self.error_prior_dist if self.error_prior_dist else random.choice(["Normal", "Gamma"])

        if base_prior_method == "Normal":
            base_prior = dist.Normal(self.base_prior_loc, self.base_prior_scale)
        elif base_prior_method == "Gamma":
            base_prior = dist.Gamma(self.base_prior_concentration, self.base_prior_rate)
        elif base_prior_method == "HalfCauchy":
            base_prior = dist.HalfCauchy(self.base_prior_scale)
        elif base_prior_method == "Bernoulli":
            base_prior = dist.Bernoulli(self.base_prior_probs)
        elif base_prior_method == "Exponential":
            base_prior = dist.Exponential(self.base_prior_rate)
        elif base_prior_method == "Gumbel":
            base_prior = dist.Gumbel(self.base_prior)
        elif base_prior_method == "Laplace":
            base_prior = dist.Laplace(self.base_prior_loc, self.base_prior_scale)
        elif base_prior_method == "LogNormal":
            base_prior = dist.LogNormal(self.base_prior_loc, self.base_prior_scale)
        elif base_prior_method == "Weibull":
            base_prior = dist.Weibull(self.base_prior_concentration, self.base_prior_rate)
        elif base_prior_method == "Beta":
            base_prior = dist.Beta(self.base_prior_concentration1, self.base_prior_concentration0)

        if infl_prior_method == "Normal":
            infl_prior = dist.Normal(self.infl_prior_loc, self.infl_prior_scale)
        elif infl_prior_method == "Dirichlet":
            infl_prior = dist.Dirichlet(self.infl_prior_probs)
        elif infl_prior_method == "Multinomial":
            infl_prior = dist.Multinomial(self.infl_prior_total_count, self.infl_prior_probs)
        elif infl_prior_method == "Binomial":
            infl_prior = dist.Binomial(self.infl_prior_total_count, self.infl_prior_probs)
        elif infl_prior_method == "Poisson":
            infl_prior = dist.Poisson(self.infl_prior_rate)
        elif infl_prior_method == "Gamma":
            infl_prior = dist.Gamma(self.infl_prior_concentration, self.infl_prior_rate)

        if error_prior_method == "Normal":
            error_prior = dist.Normal(self.error_prior_loc, self.error_prior_scale)
        elif error_prior_method == "Gamma":
            error_prior = dist.Gamma(self.error_prior_concentration, self.error_prior_rate)

        return base_prior, infl_prior, error_prior

    def apply_search_space(self):
        """
        set parameters to generate priors from search space and data
        """
    # visualizing the model
    #def model_plot(self, model):

    def model(
        self,
        data,
        y=None,
        base_prior=None,
        infl_prior=None,
        error_prior=None,
    ):
        base_prior = self.base_prior if base_prior is None else base_prior
        self.base_prior = base_prior
        infl_prior = self.infl_prior if infl_prior is None else infl_prior
        self.infl_prior = infl_prior
        error_prior = self.error_prior if error_prior is None else error_prior
        self.error_prior = error_prior
        print(f'base_prior: {base_prior}')
        print(f'infl_prior: {infl_prior}')
        print(f'error_prior: {error_prior}')
        if y is not None:
            y = jnp.array(y)
        data = jnp.array(data)
        print(f'data shape: {data.shape}')
        print(f'data: {data}')
        base = sample(
            "base",
            # dist.Normal(0, 1)  # dist.Normal(self.prior_root_mean, self.prior_root_std)
            base_prior
            # "base",
            # base_prior,
        )
        rnd_influences = sample(
            "coefs",
            infl_prior
            # dist.Normal(jnp.zeros(data.shape[1]), jnp.ones(data.shape[1]))
            # dist.Normal(
            #     self.prior_coef_means,
            #     self.prior_coef_stdvs,
            # ),
        )
        print(f'rnd_influences shape: {rnd_influences.shape}')
        print(f'data.shape[1] == rnd_influences.shape[0]: {data.shape[1]} == {rnd_influences.shape[0]}')
        mat_infl = rnd_influences.reshape(-1, 1)
        print(f'mat_infl after reshaping rnd_influences: {mat_infl.shape}')
        product = jnp.matmul(data, mat_infl).reshape(-1)
        result = product + base
        error_var = sample(
            # "error", dist.Gamma(self.gamma_alpha, self.gamma_beta)
            "error",
            # dist.Exponential(1)
            error_prior,
        )
        with plate("data_vectorized", len(result)):
            obs = sample("measurements", dist.Normal(result, error_var), obs=y) # samples from the likelihood
            print(f'obs shape: {obs.shape}')
        return obs


    def fit(self, X, y, random_key=0, verbose=True, feature_names=None, pos_map=None, mcmc_tune=None, mcmc_cores=None, mcmc_samples=None):
        self.rv_names = (
            get_n_words(X.shape[1])
            if feature_names is None
            else ["&".join(option for option in feature) for feature in feature_names]
        )

        (
            coef_prior,
            base_prior,
            error_prior,
            self.weighted_errs_per_sample,
            self.weighted_rel_errs_per_sample,
        ) = self.get_prior_weighted_normal(X, y, self.rv_names, gamma=3)
        rng_key = random.PRNGKey(random_key)
        nuts_kernel = NUTS(self.model, adapt_step_size=True)
        n_samples = mcmc_samples if mcmc_samples else self.mcmc_samples
        n_tune = mcmc_tune if mcmc_tune else self.mcmc_tune
        n_chains = mcmc_cores if mcmc_cores else self.n_chains
        mcmc = MCMC(
            nuts_kernel,
            num_samples=n_samples,
            num_warmup=n_tune,
            num_chains=n_chains,
            progress_bar=False, # when running on a TPU, this needs to be False
        )
        #X = DataFrame(X, columns=self.rv_names).to_numpy()
        #y = DataFrame(y, columns=["y"]).to_numpy()
        print(f'X shape: {X.shape}')
        print(f'y shape: {y.shape}')
        print(f'base_prior: {self.base_prior if self.base_prior else base_prior}')
        print(f'infl_prior: {self.infl_prior if self.infl_prior else coef_prior}')
        print(f'error_prior: {self.error_prior if self.error_prior else error_prior}')
        mcmc.run(
            rng_key,
            X,
            y,
            base_prior=self.base_prior if self.base_prior else base_prior,
            infl_prior=self.infl_prior if self.infl_prior else coef_prior,
            error_prior=self.error_prior if self.error_prior else error_prior,
        )
        self.samples = mcmc.get_samples()
        if verbose:
            pprint(self.samples)
            mcmc.print_summary()
        self.mcmc = mcmc
        self.update_coefs()
    
    def score(self):
        return self.loo()
    
    def predict(self, X, n_samples: int = None, ci: float = None):
        """
        Performs a prediction conforming to the sklearn interface.

        Parameters
        ----------
        X : Array-like data
        n_samples : number of posterior predictive samples to return for each prediction
        ci : value between 0 and 1 representing the desired confidence of returned confidence intervals. E.g., ci= 0.8 will generate 80%-confidence intervals

        Returns
        -------
        - a scalar if only x is specified
        - a set of posterior predictive samples of size n_samples if is given and n_samples > 0
        - a set of pairs, representing lower and upper bounds of confidence intervals for each prediction if ci is given

        """
        if n_samples is None:
            n_samples = 500

        y_samples = self._predict_samples(X, n_samples=n_samples)

        # Making sure y_samples is a numpy array
        if not isinstance(y_samples, np.ndarray):
            y_samples = np.array(y_samples)

        # If y_samples is 1D, add an extra dimension
        if y_samples.ndim == 1:
            y_samples = y_samples[:, None]

        if ci is not None:
            assert_ci(ci)
            y_pred = az.hdi(y_samples, hdi_prob=ci)
        else:
            y_pred = np.mean(y_samples, axis=0)

        return y_pred

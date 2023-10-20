import torch
import numpy as np
from domain.GP_Prior import GP_Prior
from gpytorch.models import ExactGP
from gpytorch.likelihoods import Likelihood, GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from adapters.gpytorch.means import LinearMean
from gpytorch.means import ConstantMean
from adapters.gpytorch.kernels import get_linear_kernel, get_squared_exponential_kernel, get_matern32_kernel, get_matern52_kernel, get_base_kernels, wrap_scale_kernel, additive_structure_kernel

class GPyT_Prior(GP_Prior):
    def __init__(self, X, y, feature_names):
        super().__init__(X, y, feature_names)
        
class GPRegressionModel(GP_Prior, ExactGP):
    def __init__(self, X, y, feature_names, likelihood=None, kernel="linear", mean_func="linear_weighted", structure="simple", interpolate=False):
        
        GP_Prior.__init__(self, X, y, feature_names)
        
        # transform x and y to tensors
        self.X = torch.tensor(self.X).float()
        self.y = torch.tensor(self.y).float()

        # select prior knowledge parameter values and adjust dimensionality
        self.weighted_mean = self.means_weighted[0]
        self.weighted_std = self.stds_weighted[0]

        if likelihood == "gaussian":
            ExactGP.__init__(self, self.X, self.y, GaussianLikelihood())
        elif likelihood == "fixed_noise_gaussian":
            ExactGP.__init__(self, self.X, self.y, FixedNoiseGaussianLikelihood(noise=torch.tensor(np.full((len(self.X),),self.noise_sd_over_all_regs)).float(), learn_additional_noise=True))
        elif isinstance(likelihood, Likelihood):
            ExactGP.__init__(self, self.X, self.y, likelihood=likelihood)
        else:
            raise NotImplementedError("Only Gaussian likelihood is supported for now")

        self.mean_func = self.get_mean(mean_func=mean_func)
        self.kernel = self.get_kernel(type=kernel, structure=structure)

        # init kernel hyperparameters
        if structure == "simple":
            hyper_parameter_init_values = {
                #'likelihood.noise_covar.noise': self.noise_sd_over_all_regs,
                #'mean_func.beta': self.weighted_mean,
                #'mean_func.intercept': self.root_mean,
                'kernel.base_kernel.lengthscale': torch.tensor(0.5),
                'kernel.outputscale': torch.tensor(1.),
            }
        elif structure == "additive":
            hyper_parameter_init_values = {}
            for i in range(120):
                if interpolate:
                    hyper_parameter_init_values[f'kernel.base_kernel.kernels.{i}.base_kernel.base_kernel.kernels.1.lengthscale'] = torch.tensor(0.5)    
                hyper_parameter_init_values[f'kernel.base_kernel.kernels.{i}.base_kernel.kernels.1.lengthscale'] = torch.tensor(0.5)
                hyper_parameter_init_values[f'kernel.base_kernel.kernels.{i}.outputscale'] = torch.tensor(1.)
        self.initialize(**hyper_parameter_init_values)
    
    def get_mean(self, mean_func="linear"):
        if mean_func == "constant":
            return ConstantMean()
        elif mean_func == "linear_weighted":
            return LinearMean(beta=self.means_weighted, intercept=self.root_mean)
        else:
            raise NotImplementedError("Only linear weighted mean function is supported for now")
    
    def get_kernel(self, type="linear", structure="simple", ARD=False):
        hyper_prior_params = {}
        hyper_prior_params["mean"] = self.weighted_mean
        hyper_prior_params["sigma"] = self.weighted_std
        if structure == "simple":
            if type == "linear":
                base_kernel = get_linear_kernel(self.X)
            elif type == "RBF":
                base_kernel = get_squared_exponential_kernel(self.X, **hyper_prior_params)
            elif type == "matern32":
                base_kernel = get_matern32_kernel(self.X, **hyper_prior_params)
            elif type == "matern52":
                base_kernel = get_matern52_kernel(self.X, **hyper_prior_params)
            return wrap_scale_kernel(base_kernel, **hyper_prior_params)
        elif structure == "additive":
            if type == "linear":
                base_kernels = get_base_kernels(self.X, kernel="linear", ARD=ARD)
            elif type == "RBF":
                base_kernels = get_base_kernels(self.X, kernel="RBF", ARD=ARD)
            elif type == "matern32":
                base_kernels = get_base_kernels(self.X, kernel="matern32", ARD=ARD)
            elif type == "matern52":
                base_kernels = get_base_kernels(self.X, kernel="matern52", ARD=ARD, **hyper_prior_params)
            return additive_structure_kernel(self.X, base_kernels, **hyper_prior_params)
        
    def define_kernels(self, type="linear", structure="simple", ARD=False):
        pass

    def forward(self, x):
        kernel = self.kernel(x).evaluate()
        kernel += torch.eye(kernel.shape[0]) * 1e-2 # add jitter
        output = MultivariateNormal(self.mean_func(x), kernel) # evaluate() is necessary to get the covariance/kernel matrix directly as output with a shape attribute when kernel tensors being evaluated lazily
        if not torch.isfinite(output.mean).all() or not torch.isfinite(output.variance).all():
            raise ValueError("Model output is NaN or inf")
        return output
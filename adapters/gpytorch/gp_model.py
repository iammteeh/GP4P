import numpy as np
import torch
import gpytorch
from typing import Optional
from botorch.models.utils import validate_input_scaling
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from domain.GP_Prior import GP_Prior
from gpytorch.models import ExactGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from adapters.gpytorch.pyro_model import SaasPyroModel
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.distributions import MultivariateNormal
from adapters.gpytorch.means import LinearMean
from gpytorch.means import ConstantMean
from adapters.gpytorch.kernels import get_linear_kernel, get_squared_exponential_kernel, get_matern32_kernel, get_matern52_kernel, get_base_kernels, wrap_scale_kernel, additive_structure_kernel
from gpytorch.beta_features import default_preconditioner

class GPyT_Prior(GP_Prior):
    def __init__(self, X, y, feature_names):
        super().__init__(X, y, feature_names)
        
class GPRegressionModel(GP_Prior, ExactGP, BatchedMultiOutputGPyTorchModel):
    def __init__(self, train_X, train_y, feature_names, likelihood=None, kernel="linear", mean_func="linear_weighted", structure="simple",             
            outcome_transform: Optional[OutcomeTransform] = None,
            input_transform: Optional[InputTransform] = None
            ):
            
        GP_Prior.__init__(self, train_X, train_y, feature_names)
        
        # transform x to tensor
        self.X = torch.tensor(self.X).float()
        # transform y to tensor and make it 2D
        self.y = torch.tensor(self.y).float().unsqueeze(-1)

        # preprocess data
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=self.X, input_transform=input_transform
            )
        if outcome_transform is not None:
            self.y = outcome_transform(self.y)
        self._validate_tensor_args(X=transformed_X, Y=self.y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=self.y
        )
        self._set_dimensions(train_X=self.X, train_Y=self.y)
        self.X, self.y, self.yvar = self._transform_tensor_args(X=self.X, Y=self.y)

        # select prior knowledge parameter values and adjust dimensionality
        self.weighted_mean = self.means_weighted[0]
        self.weighted_std = self.stds_weighted[0]

        if likelihood == "gaussian":
            ExactGP.__init__(self, self.X, self.y, GaussianLikelihood())
        elif isinstance(likelihood, Likelihood):
            ExactGP.__init__(self, self.X, self.y, likelihood=likelihood)
        else:
            raise NotImplementedError("Only Gaussian likelihood is supported for now")

        self.mean_func = self.get_mean(mean_func=mean_func)
        self.kernel = self.get_kernel(type=kernel, structure=structure)

        # init kernel hyperparameters
        hyper_parameter_init_values = {
            #'likelihood.noise_covar.noise': self.noise_sd_over_all_regs,
            #'mean_func.beta': self.weighted_mean,
            #'mean_func.intercept': self.root_mean,
            'kernel.base_kernel.lengthscale': torch.tensor(0.5),
            'kernel.outputscale': torch.tensor(1.),
        }
        #self.initialize(**hyper_parameter_init_values)
    
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
    
class SAASGP(GP_Prior, SaasFullyBayesianSingleTaskGP):
    def __init__(self, X, y, feature_names):
        GP_Prior.__init__(self, X, y, feature_names)
        # transform x and y to tensors
        self.X = torch.tensor(self.X).double()
        self.y = torch.tensor(self.y).double().unsqueeze(-1)
        self.noised_y = self.y + torch.randn_like(self.y) * 1e-6 # add jitter || torch.full_like(self.y, 1e-6)
        SaasFullyBayesianSingleTaskGP.__init__(self, self.X, self.y, self.noised_y, Standardize(m=1), pyro_model=SaasPyroModel())
    
    @property
    def median_lengthscale(self) -> torch.Tensor:
        r"""Median lengthscales across the MCMC samples."""
        if type(self.covar_module) is gpytorch.kernels.AdditiveStructureKernel:
            self._check_if_fitted()
            lengthscales = []    
            # Extract unique lengthscales from each ScaleKernel
            for scale_kernel in self.covar_module.base_kernel.kernels:
                # Collect lengthscales from MaternKernels within each ScaleKernel
                scale_kernel_lengthscales = [base_kernel.lengthscale.clone() for base_kernel in scale_kernel.base_kernel.kernels]
                
                # Concatenate lengthscales from both MaternKernels and compute the median
                combined_lengthscale = torch.cat(scale_kernel_lengthscales, dim=0)
                median_scale_kernel_lengthscale = combined_lengthscale.median(0).values.squeeze(0)
                
                lengthscales.append(median_scale_kernel_lengthscale)

            return torch.stack(lengthscales)
        else:
            self._check_if_fitted()
            lengthscale = self.covar_module.base_kernel.lengthscale.clone()
            return lengthscale.median(0).values.squeeze(0)

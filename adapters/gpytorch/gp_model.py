import numpy as np
import torch
import gpytorch
from typing import Optional
from botorch.models.utils import validate_input_scaling
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from domain.GP_Prior import GP_Prior
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution, MeanFieldVariationalDistribution, DeltaVariationalDistribution
from gpytorch.variational import VariationalStrategy, CiqVariationalStrategy, AdditiveGridInterpolationVariationalStrategy, NNVariationalStrategy, OrthogonallyDecoupledVariationalStrategy
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from adapters.gpytorch.pyro_model import SaasPyroModel
from botorch.models.transforms import Standardize
from gpytorch.utils import grid
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.distributions import MultivariateNormal
from adapters.gpytorch.means import LinearMean
from gpytorch.means import ConstantMean
from adapters.gpytorch.kernels import get_linear_kernel, get_squared_exponential_kernel, get_matern32_kernel, get_matern52_kernel, get_base_kernels, wrap_scale_kernel, additive_structure_kernel
from gpytorch.beta_features import default_preconditioner

class MyExactGP(GP_Prior, ExactGP, BatchedMultiOutputGPyTorchModel):
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
        # take the first out of list as they're identical for each feature
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
        hyper_parameter_init_values = {}
        if kernel == "RBF" or kernel == "matern32" or kernel == "matern52" or kernel == "RFF":
            hyper_parameter_init_values["kernel.base_kernel.lengthscale"] = torch.tensor(0.5)
        elif kernel == "polynomial":
            hyper_parameter_init_values["kernel.base_kernel.offset"] = torch.tensor(0.5)
        if structure != "additive":
            hyper_parameter_init_values["kernel.outputscale"] = torch.tensor(1.)
        self.initialize(**hyper_parameter_init_values)

    def forward(self, x):
        kernel = self.kernel(x).evaluate()
        kernel += torch.eye(kernel.shape[0]) * 1e-2 # add jitter
        output = MultivariateNormal(self.mean_func(x), kernel) # evaluate() is necessary to get the covariance/kernel matrix directly as output with a shape attribute when kernel tensors being evaluated lazily
        if not torch.isfinite(output.mean).all() or not torch.isfinite(output.variance).all():
            raise ValueError("Model output is NaN or inf")
        return output

class MyApproximateGP(GP_Prior, ApproximateGP, BatchedMultiOutputGPyTorchModel):
    def __init__(self, train_X, train_y, feature_names, kernel="linear", mean_func="linear_weighted", structure="simple", inducing_points=None):
        
        GP_Prior.__init__(self, train_X, train_y, feature_names)
        # transform x and y to tensors
        self.X = torch.tensor(self.X).float()
        self.y = torch.tensor(self.y).float().unsqueeze(-1)
        # select prior knowledge parameter values and adjust dimensionality
        self.weighted_mean = self.means_weighted[0]
        self.weighted_std = self.stds_weighted[0]
        # draw inducing points
        if inducing_points is None:
            inducing_points = torch.tensor(train_X[torch.randperm(self.X.size(0))[:len(self.X)//2]]).float()
        #self.variational_strategy.register_preconditioner("cij")
        #self.variational_strategy.register_preconditioner("cii")
        #self.variational_strategy.register_preconditioner("cjj")
        #variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        #variational_distribution = MeanFieldVariationalDistribution(num_inducing_points=inducing_points.size(0))
        variational_distribution = DeltaVariationalDistribution(inducing_points.size(0))
        #variational_strategy = CiqVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        grid_size = int(grid.choose_grid_size(self.X, kronecker_structure=False))
        variational_strategy = AdditiveGridInterpolationVariationalStrategy(self, grid_size=grid_size, grid_bounds=[(0.0, 1.0)*len(self.X.T)], num_dim=len(self.X), variational_distribution=variational_distribution, mixing_params=True)
        #variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        #variational_strategy = NNVariationalStrategy(self, inducing_points, variational_distribution,k=len(self.X.T), training_batch_size=len(self.X.T))
        ApproximateGP.__init__(self, variational_strategy)
        self.mean_module = self.get_mean(mean_func=mean_func)
        self.covar_module = self.get_kernel(type=kernel, structure=structure)
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x).evaluate()
        covar_x += torch.eye(covar_x.shape[0]) * 1e-2 # add jitter
        return MultivariateNormal(mean_x, covar_x)


class SAASGP(GP_Prior, SaasFullyBayesianSingleTaskGP):
    def __init__(self, X, y, feature_names, mean_func="linear", kernel_structure="simple", kernel_type="linear"):
        GP_Prior.__init__(self, X, y, feature_names)
        # transform x and y to tensors
        self.X = torch.tensor(self.X).double()
        self.y = torch.tensor(self.y).double().unsqueeze(-1)
        #self.noised_y = self.y + torch.randn_like(self.y) * 1e-6 # add jitter || torch.full_like(self.y, 1e-6)
        #SaasFullyBayesianSingleTaskGP.__init__(self, self.X, self.y, self.noised_y, Standardize(m=1), pyro_model=SaasPyroModel())
        pyro_model = SaasPyroModel(mean_func=mean_func, kernel_structure=kernel_structure, kernel_type=kernel_type)
        SaasFullyBayesianSingleTaskGP.__init__(self, self.X, self.y, pyro_model=pyro_model)

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

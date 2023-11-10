from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.test_functions import DTLZ1, DTLZ2
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from adapters.botorch.aquisitions import LinearMCObjective
from botorch.acquisition.objective import MCAcquisitionObjective, ConstrainedMCObjective
import torch
from torch import Tensor
from domain.GP_Prior import GP_Prior
from sklearn.linear_model import RidgeCV
import numpy as np
from numpy import ndarray
from domain.env import ALPHAS
from datetime import datetime

class LinearPredictionUtility(BaseTestProblem, GP_Prior, LinearMCObjective):
    def __init__(self, X, y, feature_names):
        GP_Prior.__init__(self, X, y, feature_names)
        self.noise_std = self.noise_sd_over_all_regs
        LinearMCObjective.__init__(self, weights=self.means_weighted)
        # normalize X
        self.input_transform = Normalize(d=self.X.shape[-1])
        # standardize y
        self.outcome_transform = Standardize(m=self.y.shape[-1])

    def evaluate_true(self, X: ndarray) -> Tensor:
        """
        perform a linear regression over the training data
        and return the predicted values for X
        """
        alphas = np.logspace(*ALPHAS)
        model = RidgeCV(alphas, store_cv_values=True)
        model.fit(self.X, self.y)
        # store CV values in file with timestamp
        cv_values = model.cv_values_
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        np.savetxt(f"regression_cv_values_{timestamp}.csv", cv_values, delimiter=",")
        if not isinstance(X, ndarray):
            X = np.array(X).reshape(1, -1)
        y_hat = model.predict(X)
        return y_hat
    
    def forward(self, X: Tensor, noise: bool = True, samples=None) -> Tensor:
        r"""Evaluate the function on a set of points.
        """
        f = Tensor(self.evaluate_true(X=X)).float()
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        #if self.negate:
        #    f = -f
        return f.norm(dim=-1, keepdim=True)
    
    def apply_constraints(self, X: Tensor) -> Tensor:
        outcome_constraints = self.outcome_constraint(X)
        return outcome_constraints
    
    def outcome_constraint(self, X: Tensor) -> Tensor:
        return X.sum(dim=-1) - 3
    
    def get_objective_weights(self) -> Tensor:
        return self.means_weighted
    
    def get_objective_thresholds(self) -> Tensor:
        return self.thresholds
    
    
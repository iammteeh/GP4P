import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from operator import itemgetter
from typing import ArrayLike, Int, MatrixLike
from sklearn.base import clone
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.gaussian_process import GaussianProcessRegressor, check_random_state
from scipy.stats import multivariate_normal, norm

def compare_predictions(pre_mean, pre_cov, gpr_mean, gpr_cov):
    # compute log-likelihood for both models
    pre_ll = multivariate_normal.logpdf(pre_mean, mean=pre_mean, cov=pre_cov)
    gpr_ll = multivariate_normal.logpdf(gpr_mean, mean=pre_mean, cov=pre_cov)

    if pre_ll > gpr_ll:
        print("Prior has higher log-likelihood. Returning prior.")
        return pre_mean, pre_cov
    elif gpr_ll > pre_ll:
        print("GPR has higher log-likelihood. Returning GPR.")
        return gpr_mean, gpr_cov
    else:
        print("Both models have the same log-likelihood. Returning prior.")
        return pre_mean, pre_cov

class GPR(GaussianProcessRegressor):
    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):
        """
        for details, see original implementation
        """
        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, copy_X_train=copy_X_train, random_state=random_state)

    def sample_y(self, X, pre_mean: ArrayLike = None, pre_cov: MatrixLike = None, n_samples: Int = 1000, random_state: RandomState = 0) -> ndarray:
        """
        for details, see original implementation
        """
        rng = check_random_state(random_state)
        if pre_mean is None or pre_cov is None:
            pre_mean, pre_cov = self.predict(X, return_cov=True)
        else:
            gpr_mean, gpr_cov = self.predict(X, return_cov=True)
            y_mean, y_cov = compare_predictions(pre_mean, pre_cov, gpr_mean, gpr_cov)
        
        # sample from multivariate normal distribution
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [
                rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)
        return y_samples

        
        

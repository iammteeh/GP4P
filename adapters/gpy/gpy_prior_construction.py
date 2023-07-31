from adapters.pymc.prior_construction import GP_Prior
from adapters.gpy.gpy_kernels import get_kernel, get_base_kernels, get_linear_kernel, get_matern52_kernel
from domain.feature_model.feature_modeling import additive_kernel_permutation
import GPy as gp
import numpy as np


class GPy_Prior(GP_Prior):
    def __init__(self, X, y, feature_names, mean_func="linear", kernel_type="linear", kernel_structure="simple", ARD=True):
        super().__init__(X, y, feature_names, mean_func=mean_func, kernel=kernel_type)
        self.mean_func = self.get_mean(mean_func=mean_func)
        self.kernel = self.get_kernel(type=kernel_type, structure=kernel_structure, ARD=ARD)
    def get_mean(self, mean_func="linear_weighted"):
        if mean_func == "standard":
            mf = gp.mappings.Linear(input_dim=self.X.shape[1], output_dim=1, name="mean_func")
            return mf
        elif mean_func == "linear_weighted":
            mf = gp.mappings.Linear(input_dim=self.X.shape[1], output_dim=1, name="mean_func")
            # apply weights to mean function
            #weights = [self.means_weighted, self.root_mean]
            mf[:] = np.array(self.means_weighted)
            return mf
        
    def get_kernel(self, type="linear", structure="simple", ARD=True):
        if structure == "simple":
            if type == "linear":
                return get_linear_kernel(self.X, ARD=ARD)
            elif type == "matern52":
                return get_matern52_kernel(self.X, ARD=ARD)
        elif structure == "additive":
            if type == "linear":
                base_kernels = get_base_kernels(self.X, kernel="linear", ARD=ARD)
                return additive_kernel_permutation(base_kernels, k=3)

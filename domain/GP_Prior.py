from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV, LassoCV
from sklearn.kernel_ridge import KernelRidge
from adapters.gpytorch.means import LinearMean
from gpytorch.means import ConstantMean
from adapters.gpytorch.kernels import get_linear_kernel, get_squared_exponential_kernel, get_matern32_kernel, get_matern52_kernel, get_spectral_mixture_kernel, get_rff_kernel, get_polynomial_d2_kernel, get_polynomial_d3_kernel, get_polynomial_d4_kernel, get_piecewise_polynomial_kernel, get_base_kernels, wrap_scale_kernel, additive_structure_kernel
from domain.env import KERNEL_TYPE
from adapters.util import get_feature_names_from_rv_id, print_scores, get_err_dict
import math
from scipy.special import binom
import jax.numpy as jnp
import numpy as np
import time
import copy
from abc import abstractmethod

def weighted_avg_and_std(values, weights, gamma=1):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if gamma != 1:
        weights = np.power(weights, gamma)
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    if variance <= 0:
        sqr_var = 0.0
    else:
        sqr_var = math.sqrt(variance)
    return average, sqr_var

def is_positive_semi_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)


class Priors:
    def __init__(self, X, y, feature_names):
        # TODO: check if X and y is a dataframe or a numpy array
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.pos_map = {ft: pos for pos, ft in enumerate(feature_names)}

    def transform_data_to_candidate_features(self, candidate, train_x):
        mapped_features = []
        for term in candidate:
            idx = [self.pos_map[ft] for ft in term]
            selected_cols = np.array(train_x)[:, idx]
            if len(idx) > 1:
                mapped_feature = np.product(selected_cols, axis=1).ravel()
            else:
                mapped_feature = selected_cols.ravel()
            mapped_features.append(list(mapped_feature))
        reshaped_mapped_x = np.atleast_2d(mapped_features).T
        return reshaped_mapped_x

    def fit_and_eval_lin_reg(
        self, reg_proto=None, verbose=True
    ):
        if not reg_proto:
            reg_proto = Ridge()
        inters = [
            get_feature_names_from_rv_id(ft_inter_Str)
            for ft_inter_Str in self.feature_names
        ]
        x_mapped = self.X  # self.transform_data_to_candidate_features(inters, X)
        lr = copy.deepcopy(reg_proto)
        lr.fit(x_mapped, self.y)
        if verbose:
            print(f"{lr.__class__.__name__}")
            print_scores("analogue LR", lr, "train set", x_mapped, self.y)
        errs = get_err_dict(lr, x_mapped, self.y)
        return lr, errs

    def get_regression_spectrum(
        self, n_steps=50, cv=3, n_jobs=-1
    ):
        """
        Sample coefficients of an ElasticNetCV regressor with different l1_ratios
        and do Ridge and Lasso regression for L1 and L2 norm respectively.
        """
        start = time.time()
        regs = []
        step_list = np.linspace(0, 1, n_steps)
        print(f"fitting {n_steps} regressors")
        for l1_ratio in step_list:
            if 0 < l1_ratio < 1:
                reg_prototype = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, n_jobs=n_jobs)
                reg, err = self.fit_and_eval_lin_reg(
                    reg_proto=reg_prototype, verbose=False
                )
                regs.append((reg, err))
        ridge = RidgeCV(cv=cv)
        lasso = LassoCV(cv=cv, n_jobs=n_jobs)
        for reg in [ridge, lasso]:
            fitted_reg, err = self.fit_and_eval_lin_reg(
                reg_proto=reg, verbose=True # only show L1 und L2 norm results
            )
            regs.append((fitted_reg, err))

        reg_dict = {l1_ratio: tup[0] for tup, l1_ratio in zip(regs, step_list)}
        err_dict = {l1_ratio: tup[1] for tup, l1_ratio in zip(regs, step_list)}

        end = time.time()
        cost = end - start
        print("Prior Spectrum Computation took", cost)

        return reg_dict, err_dict
    
    def get_weighted_normal_params(self, gamma=1, stddev_multiplier=3):
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum()
        all_raw_errs = [errs["raw"] for errs in list(err_dict.values())]
        all_abs_errs = np.array(
            [abs(err["y_pred"] - err["y_true"]) for err in all_raw_errs]
        )
        mean_abs_errs = all_abs_errs.mean(axis=1)
        all_rel_errs = np.array(
            [
                abs((err["y_pred"] - err["y_true"]) / err["y_true"])
                for err in all_raw_errs
            ]
        )
        mean_rel_errs = all_rel_errs.mean(axis=1)
        reg_list = list(reg_dict_final.values())

        means_weighted = []
        stds_weighted = []
        # calculate weights
        weights = (
            1 - MinMaxScaler().fit_transform(np.atleast_2d(mean_abs_errs).T).ravel()
        )
        # extract errors
        err_mean, err_std = weighted_avg_and_std(mean_abs_errs, weights, gamma=gamma)
        noise_sd_over_all_regs = err_mean + 3 * err_std
        # extract roots from intercepts
        root_candidates = np.array([reg.intercept_ for reg in reg_list])
        root_mean, root_std = weighted_avg_and_std(
            root_candidates, weights, gamma=gamma
        )
        # extract means and variances/stds from coefficients
        coef_matrix = []
        for coef_id, coef in enumerate(self.feature_names):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list]) # vector of coefficients
            coef_matrix.append(coef_candidates)
            mean_weighted, std_weighted = weighted_avg_and_std(
                coef_candidates, weights, gamma=gamma
            )
            means_weighted.append(mean_weighted)
            stds_weighted.append(stddev_multiplier * std_weighted)

        coef_matrix = np.array(coef_matrix).T # transpose to have colvars
        ########################
        weighted_errs_per_sample = np.average(
            all_abs_errs, axis=0, weights=mean_abs_errs
        )
        weighted_rel_errs_per_sample = np.average(
            all_rel_errs, axis=0, weights=mean_rel_errs
        )
        #########################
        return root_mean, root_std, means_weighted, stds_weighted, coef_matrix, noise_sd_over_all_regs
    
    def exploit_kernel_ridge_regression(self):
        kernel_type = "rbf" if KERNEL_TYPE == "RBF" else "laplacian" 
        regs = []
        step_size = int(binom(len(self.X.T), 3))
        print(f"fit {step_size} kernel ridge regressors out of {len(self.X.T)} features")
        step_list = np.linspace(0.1, 0.9, step_size) # leaving 0 < alpha < 1
        start_time = time.time()
        for alpha in step_list:
            alpha = 1/(2*alpha)
            kernel_ridge = KernelRidge(alpha=alpha, kernel=kernel_type)
            reg, err = self.fit_and_eval_lin_reg(reg_proto=kernel_ridge, verbose=False)
            regs.append((reg, err))

        reg_dict = {l1_ratio: tup[0] for tup, l1_ratio in zip(regs, step_list)}
        err_dict = {l1_ratio: tup[1] for tup, l1_ratio in zip(regs, step_list)}

        end_time = time.time()
        print(f"fitting {len(regs)} kernel ridge regressors took {end_time - start_time:.2f} seconds")

        all_raw_errs = [errs["raw"] for errs in list(err_dict.values())]
        all_abs_errs = np.array(
            [abs(err["y_pred"] - err["y_true"]) for err in all_raw_errs]
        )
        mean_abs_errs = all_abs_errs.mean(axis=1)
        all_rel_errs = np.array(
            [
                abs((err["y_pred"] - err["y_true"]) / err["y_true"])
                for err in all_raw_errs
            ]
        )
        mean_rel_errs = all_rel_errs.mean(axis=1)
        reg_list = list(reg_dict.values())

        means_weighted = []
        stds_weighted = []
        weights = (
            1 - RobustScaler().fit_transform(np.atleast_2d(mean_abs_errs).T).ravel()
        )
        err_mean, err_std = weighted_avg_and_std(mean_abs_errs, weights, gamma=1)
        noise_sd_over_all_regs = err_mean + 3 * err_std
        # extract mean and root of variance
        for coef_id, coef in enumerate(self.feature_names):
            means = np.array([reg.dual_coef_.mean() for reg in reg_list]) # means of the vector of coefficients in kernel space
            variances = np.array([reg.dual_coef_.var() for reg in reg_list]) # variances of the vector of coefficients in kernel space
            mean_weighted = np.average(means, weights=weights)
            std_weighted = math.sqrt(np.average(variances, weights=weights))
            means_weighted.append(mean_weighted)
            stds_weighted.append(3 * std_weighted)

        ########################
        weighted_errs_per_sample = np.average(
            all_abs_errs, axis=0, weights=mean_abs_errs
        )
        weighted_rel_errs_per_sample = np.average(
            all_rel_errs, axis=0, weights=mean_rel_errs
        )
        #########################
        return means_weighted, stds_weighted, noise_sd_over_all_regs


class GP_Prior(Priors):
    def __init__(self, X, y, feature_names):
        super().__init__(X, y, feature_names)
        # compute empirical prior parameters to avoid improper priors
        (self.root_mean, 
        self.root_std, 
        self.means_weighted, 
        self.stds_weighted, 
        coef_matrix, 
        self.noise_sd_over_all_regs ) = self.get_weighted_normal_params(gamma=1, stddev_multiplier=3)

    def get_mean(self, mean_func="linear"):
        if mean_func == "constant":
            return ConstantMean()
        elif mean_func == "linear_weighted":
            return LinearMean(beta=self.means_weighted, intercept=self.root_mean)
        else:
            raise NotImplementedError("Only linear weighted mean function is supported for now")
    
    def get_kernel(self, type="linear", structure="simple"):
        hyper_prior_params = {}
        hyper_prior_params["mean"] = self.weighted_mean
        hyper_prior_params["sigma"] = self.weighted_std
        if structure == "simple":
            if type == "linear":
                base_kernel = get_linear_kernel(self.X)
            elif type == "polynomial" or type == "poly2":
                base_kernel = get_polynomial_d2_kernel(self.X)
            elif type == "poly3":
                base_kernel = get_polynomial_d3_kernel(self.X)
            elif type == "poly4":
                base_kernel = get_polynomial_d4_kernel(self.X)
            elif type == "piecewise_polynomial":
                base_kernel = get_piecewise_polynomial_kernel(self.X)
            elif type == "RBF":
                base_kernel = get_squared_exponential_kernel(self.X)
            elif type == "matern32":
                base_kernel = get_matern32_kernel(self.X)
            elif type == "matern52":
                base_kernel = get_matern52_kernel(self.X)
            elif type == "RFF":
                base_kernel = get_rff_kernel(self.X)
            elif type == "spectral_mixture":
                base_kernel = get_spectral_mixture_kernel(self.X)
            else:
                raise NotImplementedError("Only linear, polynomial, RBF, matern32, matern52, RFF and spectral_mixture kernels are supported for now")
            return wrap_scale_kernel(base_kernel)
        elif structure == "additive":
            if type == "polynomial":
                base_kernels = get_base_kernels(self.X, kernel="polynomial")
            elif type == "piecewise_polynomial":
                base_kernels = get_base_kernels(self.X, kernel="piecewise_polynomial")
            elif type == "RBF":
                base_kernels = get_base_kernels(self.X, kernel="RBF")
            elif type == "matern32":
                base_kernels = get_base_kernels(self.X, kernel="matern32")
            elif type == "matern52":
                base_kernels = get_base_kernels(self.X, kernel="matern52")
            elif type == "RFF":
                base_kernels = get_base_kernels(self.X, kernel="RFF")
            elif type == "spectral_mixture":
                base_kernels = get_base_kernels(self.X, kernel="spectral_mixture")
            else:
                raise NotImplementedError("Only linear, polynomial, RBF, matern32, matern52, RFF and spectral_mixture kernels are supported for now")
            return additive_structure_kernel(self.X, base_kernels)
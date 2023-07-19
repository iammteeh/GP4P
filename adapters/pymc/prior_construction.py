from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV, LassoCV
from adapters.PyroMCMCRegressor import PyroMCMCRegressor
from bayesify.pairwise import get_feature_names_from_rv_id, print_scores, get_err_dict
import pymc3 as pm
from sklearn import kernel_approximation, metrics
from adapters.pymc.kernel_construction import get_linear_kernel, get_additive_lr_kernel
from numpy.linalg import eigvalsh
from scipy.linalg import sqrtm
import math
import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist
import time
import copy

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
            print_scores("analogue LR", lr, "train set", x_mapped, self.y)
        errs = get_err_dict(lr, x_mapped, self.y)
        return lr, errs

    def get_regression_spectrum(
        self, n_steps=50, cv=3, n_jobs=-1
    ):
        start = time.time()
        regs = []
        step_list = np.linspace(0, 1, n_steps)
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
                reg_proto=reg, verbose=False
            )
            regs.append((fitted_reg, err))

        reg_dict = {l1_ratio: tup[0] for tup, l1_ratio in zip(regs, step_list)}
        err_dict = {l1_ratio: tup[1] for tup, l1_ratio in zip(regs, step_list)}

        end = time.time()
        cost = end - start
        print("Prior Spectrum Computation took", cost)

        return reg_dict, err_dict
    
    def get_weighted_mvnormal_params(self, gamma=1, stddev_multiplier=3):
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
        print(f"fitting {len(reg_list)} regressors")

        means_weighted = []
        stds_weighted = []
        weights = (
            1 - MinMaxScaler().fit_transform(np.atleast_2d(mean_abs_errs).T).ravel()
        )
        err_mean, err_std = weighted_avg_and_std(mean_abs_errs, weights, gamma=gamma)
        noise_sd_over_all_regs = err_mean + 3 * err_std
        root_candidates = np.array([reg.intercept_ for reg in reg_list])
        root_mean, root_std = weighted_avg_and_std(
            root_candidates, weights, gamma=gamma
        )

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

        weighted_errs_per_sample = np.average(
            all_abs_errs, axis=0, weights=mean_abs_errs
        )
        weighted_rel_errs_per_sample = np.average(
            all_rel_errs, axis=0, weights=mean_rel_errs
        )

        return root_mean, root_std, means_weighted, stds_weighted, coef_matrix, noise_sd_over_all_regs
    
class GP_Prior(Priors):
    def __init__(self, X, y, feature_names, mean_func="linear", kernel="linear"):
        super().__init__(X, y, feature_names)
        # compute empirical prior parameters to avoid improper priors
        (self.root_mean, 
        self.root_std, 
        self.means_weighted, 
        self.stds_weighted, 
        self.coef_matrix, 
        self.noise_sd_over_all_regs ) = self.get_weighted_mvnormal_params(gamma=1, stddev_multiplier=3)
        self.mean_func = self.get_mean(mean_func=mean_func)
        self.kernel = self.get_kernel(kernel=kernel)

    def get_mean(self, mean_func="linear"):
        if mean_func == "linear":
            mean_func = pm.gp.mean.Linear(coeffs=self.means_weighted, intercept=self.root_mean)
        elif mean_func == "constant":
            mean_func = pm.gp.mean.Constant(c=np.mean(self.means_weighted))
        
        return mean_func
    
    def get_kernel(self, kernel="linear"):
        if kernel == "linear":
            return get_linear_kernel(self.X)
        elif kernel == "additive_lr":
            return get_additive_lr_kernel(self.X, self.root_mean, self.root_std)
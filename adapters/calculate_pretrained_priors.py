from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV, LassoCV
from adapters.PyroMCMCRegressor import PyroMCMCRegressor
from bayesify.pairwise import weighted_avg_and_std, get_feature_names_from_rv_id, print_scores, get_err_dict
import pymc3 as pm
from sklearn import kernel_approximation, metrics
from numpy.linalg import eigvalsh
from scipy.linalg import sqrtm
import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist
import time
import copy

class Priors:
    def __init__(self, X, y, feature_names, dummy=True):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.dummy = dummy
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
        self, X, y, lin_reg_features, reg_proto=None, verbose=True
    ):
        if not reg_proto:
            reg_proto = Ridge()
        inters = [
            get_feature_names_from_rv_id(ft_inter_Str)
            for ft_inter_Str in lin_reg_features
        ]
        x_mapped = X  # self.transform_data_to_candidate_features(inters, X)
        lr = copy.deepcopy(reg_proto)
        lr.fit(x_mapped, y)
        if verbose:
            print_scores("analogue LR", lr, "train set", x_mapped, y)
        errs = get_err_dict(lr, x_mapped, y)
        return lr, errs

    def get_regression_spectrum(
        self, X, y, lin_reg_features, n_steps=50, cv=3, n_jobs=-1
    ):
        start = time.time()
        regs = []
        step_list = np.linspace(0, 1, n_steps)
        for l1_ratio in step_list:
            if 0 < l1_ratio < 1:
                reg_prototype = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, n_jobs=n_jobs)
                reg, err = self.fit_and_eval_lin_reg(
                    X, y, lin_reg_features, reg_proto=reg_prototype, verbose=False
                )
                regs.append((reg, err))
        ridge = RidgeCV(cv=cv)
        lasso = LassoCV(cv=cv, n_jobs=n_jobs)
        for reg in [ridge, lasso]:
            fitted_reg, err = self.fit_and_eval_lin_reg(
                X, y, lin_reg_features, reg_proto=reg, verbose=False
            )
            regs.append((fitted_reg, err))

        reg_dict = {l1_ratio: tup[0] for tup, l1_ratio in zip(regs, step_list)}
        err_dict = {l1_ratio: tup[1] for tup, l1_ratio in zip(regs, step_list)}

        end = time.time()
        cost = end - start
        print("Prior Spectrum Computation took", cost)

        return reg_dict, err_dict
    
    def get_prior_weighted_normal(self, X, y, feature_names, gamma=1, stddev_multiplier=3, kernel="expquad"):
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum(X, y, feature_names)
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
        print(f"reg_list: {reg_list}")

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
        for coef_id, coef in enumerate(feature_names):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list]) # vector of coefficients
            print(f"coef_candidates with shape {len(coef_candidates)} {coef_candidates}")
            coef_matrix.append(coef_candidates)
            mean_weighted, std_weighted = weighted_avg_and_std(
                coef_candidates, weights, gamma=gamma
            )
            means_weighted.append(mean_weighted)
            stds_weighted.append(stddev_multiplier * std_weighted)

        print(f"coef matrix: {coef_matrix}")
        print(f"means weighted: {means_weighted}")
        print(f"stds weighted: {stds_weighted}")

        print(f"construct cov matrix")
        coef_matrix = np.array(coef_matrix).T # convert to numpy array and transpose
        coef_matrix = np.corrcoef(coef_matrix) # perturb the covariance matrix with the correlation matrix
        if kernel == "linear":
            cov_matrix = np.cov(coef_matrix, rowvar=False)  # Compute the covariance matrix with a linear kernel (i.e. no kernel) as its derived from the linear regression coefficients
        elif kernel == "expquad":
            cov_matrix = pm.gp.cov.ExpQuad(input_dim=len(X.T), ls=len(means_weighted)).full(X) # Compute the covariance matrix with a squared exponential kernel
        elif kernel == "rbf":
            cov_matrix = pm.gp.cov.RBF(1, coef_matrix.shape[1]).full(X) # Compute the covariance matrix with a radial basis function kernel
        elif kernel == "matern":
            cov_matrix = pm.gp.cov.Matern52(1, coef_matrix.shape[1]).full(X)
        elif kernel == "X2":
            cov_matrix = metrics.polynomial_kernel(coef_matrix, coef_matrix, degree=2)
        #elif kernel == "X3":
        #    cov_matrix = sklearn.metrics.pairwise.polynomial_kernel(coef_matrix, coef_matrix, degree=3)
        #elif kernel == "chi2":
        #    cov_matrix = sklearn.metrics.pairwise.chi2_kernel(coef_matrix, coef_matrix)
        #elif kernel == "additive_chi2":
        #    cov_matrix = sklearn.metrics.additive_chi2_kernel(coef_matrix, coef_matrix)
        #elif kernel == "laplacian":
        #    cov_matrix = sklearn.metrics.pairwise.laplacian_kernel(coef_matrix, coef_matrix)
        print(f"cov matrix: {cov_matrix}")

        # check iff any NaN values occur, replace them with 0
        if np.isnan(means_weighted).any():
            means_weighted = np.nan_to_num(means_weighted)
        # check if covariance matrix is positive semi-definite
        def is_positive_semi_definite(matrix):
            return np.all(np.linalg.eigvals(matrix) >= 0)

        #if not is_positive_semi_definite(cov_matrix) and not isinstance(cov_matrix, pm.gp.cov.ExpQuad):
        #    print("Covariance matrix is not positive semi-definite. Adding noise.")
        #    cov_matrix += np.eye(cov_matrix.shape[0]) * noise_sd_over_all_regs
        #    if not is_positive_semi_definite(cov_matrix):
        #        print("Covariance matrix is still not positive semi-definite. Compute PSD approximation.")
        #        cov_matrix = sqrtm(cov_matrix)
        #        cov_matrix = cov_matrix @ cov_matrix.T
        #        if not is_positive_semi_definite(cov_matrix):
        #            raise ValueError("Covariance matrix is not positive semi-definite.")
        
        # Use the multivariate normal distribution for the coefficients
        #coef_prior = dist.MultivariateNormal(jnp.array(means_weighted), covariance_matrix=jnp.array(cov_matrix))  
        weighted_errs_per_sample = np.average(
            all_abs_errs, axis=0, weights=mean_abs_errs
        )
        weighted_rel_errs_per_sample = np.average(
            all_rel_errs, axis=0, weights=mean_rel_errs
        )

        base_prior = dist.Normal(jnp.array(root_mean), jnp.array(root_std))
        error_prior = dist.Exponential(jnp.array(err_mean))

        return (
            means_weighted,
            cov_matrix,
            #coef_prior,
            base_prior,
            error_prior,
            weighted_errs_per_sample,
            weighted_rel_errs_per_sample,
        )

    def get_weighted_normal_params(self, X, y, feature_names, gamma=1, stddev_multiplier=3, kernel="expquad"):
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum(X, y, feature_names)
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
        print(f"reg_list: {reg_list}")

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
        for coef_id, coef in enumerate(feature_names):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list]) # vector of coefficients
            print(f"coef_candidates with shape {len(coef_candidates)} {coef_candidates}")
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
    
    def get_gp_prior(self, X, y, feature_names, µ_vector, sigma, noise, kernel="expquad", coef_matrix=None):
        def is_positive_semi_definite(matrix):
            return np.all(np.linalg.eigvals(matrix) >= 0)
        
        if kernel == "linear":
            cov_matrix = np.corrcoef(coef_matrix) # perturb coef_matrix to get cov_matrix
            cov_matrix = np.cov(coef_matrix, rowvar=False)

            if np.isnan(µ_vector).any():
                µ_vector = np.nan_to_num(µ_vector)
            if np.isnan(cov_matrix).any() and np.isinf(cov_matrix).any():
                print("cov_matrix contains nan or inf")
                cov_matrix = np.eye(len(µ_vector))

            if not is_positive_semi_definite(cov_matrix):
                print("Covariance Matrix is not positive semi definite. Adding noise.")
                cov_matrix += np.eye(len(µ_vector)) * noise
                if not is_positive_semi_definite(cov_matrix):
                    print("Covariance Matrix is still not positive semi definite. Compute PSD approximation.")
                    cov_matrix = sqrtm(cov_matrix)
                    cov_matrix = np.dot(cov_matrix, cov_matrix.T)
                    if not is_positive_semi_definite(cov_matrix):
                        raise ValueError("Covariance Matrix is still not positive semi definite. Cannot compute GP prior.")
            
            #coef_prior = dist.MultivariateNormal(jnp.array(µ_vector), covariance_matrix=jnp.array(cov_matrix)
            return pm.gp.cov.Linear(len(µ_vector), cov_matrix)
                    
        elif kernel == "expquad":
            return pm.gp.cov.ExpQuad(len(µ_vector), ls=1.0)

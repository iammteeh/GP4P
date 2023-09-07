
from types import TracebackType
from typing import Any, Optional, Type
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn import linear_model, svm, tree, ensemble, neighbors, naive_bayes, discriminant_analysis, gaussian_process
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, roc_curve
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import pymc as pm
from scipy.stats import norm

class Regression:
    def __init__(self, X_train, X_test, y_train, feature_names, method=None, alphas=None):
        self.X_train = X_train
        self.feature_names = feature_names
        self.X_test = X_test
        self.y_train = y_train
        self.method = self.set_method(method, alphas)
    
    def set_method(self, method, alphas):
        if method == "linear":
            return linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        elif method == "lasso":
            return linear_model.Lasso(alpha=alphas[0], max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "ridge":
            return linear_model.Ridge(alpha=alphas[0], max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "elastic":
            return linear_model.ElasticNet(alpha=alphas, max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "lars":
            return linear_model.Lars(copy_X=True, fit_intercept=True, normalize=False, verbose=1)
        elif method == "LarsCV":
            return linear_model.LarsCV(copy_X=True, fit_intercept=True, normalize=True, verbose=1)
        elif method == "LassoCV":
            return linear_model.LassoCV(alphas=alphas, max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "RidgeCV":
            return linear_model.RidgeCV(alphas=alphas, fit_intercept=True)
        elif method == "ElasticNetCV":
            return linear_model.ElasticNetCV(alphas=alphas, max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "BayesianRidge":
            return linear_model.BayesianRidge(copy_X=True, fit_intercept=True, verbose=True)
        elif method == "ARDRegression":
            return linear_model.ARDRegression(copy_X=True, fit_intercept=True, normalize=False)
        elif method == "SGDRegressor":
            return linear_model.SGDRegressor(max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "PLS":
            return PLSRegression(n_components=16, scale=True, max_iter=9999, tol=1e-06, copy=True)
        elif method == "symbolic":
            self.X_train = self.X_train.astype(float)
            self.X_test = self.X_test.astype(float)
            return SymbolicRegressor(population_size=1000,
                            generations=20, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient='auto', random_state=42, n_jobs=14, function_set=('add', 'sub'))
        elif method == "symbolic_transformer":
            self.X_train = self.X_train.astype(float)
            self.X_test = self.X_test.astype(float)
            return SymbolicTransformer(population_size=5000,
                            generations=75, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=42, n_jobs=8)
        else:
            raise ValueError("no such method")

    def __enter__(self):
        return self
    
    def __exit__(self,
                exc_type: Optional[Type[BaseException]],
                exc_value: Optional[BaseException],
                exc_traceback: Optional[TracebackType]
                ) -> bool:
        return False
    
    def fit(self):
        self.method.fit(self.X_train, self.y_train)
    
    def predict(self):
        return self.method.predict(self.X_test)
    
    def get_coef(self):
        return self.method.coef_
    
    def get_significant_coef(self, threshold=0.01):
        significant_coefficients_indices = np.where(np.abs(self.method.coef_) > threshold)[0]
        return self.method.coef_[significant_coefficients_indices]
    
    def get_intercept(self):
        return self.method.intercept_
    
    def get_feature_coefficients(self):
        coef = self.get_coef()
        return dict(zip(self.feature_names, coef))
    
    def get_significant_features(self, threshold=0.01):
        significant_coefs = self.get_significant_coef(threshold)
        significant_feature_names = [self.feature_names[i] for i in np.where(np.abs(self.method.coef_) > threshold)[0]]
        return dict(zip(significant_feature_names, significant_coefs))
  
    def get_program(self):
        return self.method._program
    
    #def get_regression_spectrum(self, X, y, feature_names, n_steps=50, cv=5, n_jobs=5)
    
    #def weighted_avg_and_std(values, weights, gamma)

    #def get_prior_weighted_normal(self, X, y, rv_names, gamma, stddev_multiplier)
    
class P4extended():
    pass



class ProbitRegressor(LogisticRegression):
    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=1,
        warm_start=False,
        n_jobs=10,
        l1_ratio=None,
    ):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

    def probit_link(self, x):
        from scipy.special import expit
        return expit(x)
    
    def probit_regression(self, X, y):
        n_features = X.shape[1]
        with pm.Model() as self.model:
            # Define priors for the intercept and features
            intercept = pm.Normal('intercept', mu=0, sd=10)
            coeffs = pm.Normal('coeffs', mu=0, sd=10, shape=n_features)

            # Calculate the linear predictor
            linpred = intercept + pm.math.dot(X, coeffs)

            # Define the likelihood using the Probit link function
            y_obs = pm.Bernoulli('y_obs', p=pm.math.invlogit(linpred), observed=y)
        return self.model
    
    def fit(self, X_train_scaled, y_train):
        self.model = self.probit_regression(X_train_scaled, y_train)
        with self.model:
            self.trace = pm.sample(2000, tune=1000, start={'alpha': np.array([0.]), 'tau': np.array([0.]), 'sigma_c': np.array([0.]), 'sigma_t': np.array([0.])})
        return self.trace

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        return self.probit_link(proba)
    
    def predict(self, X, use_proba=False):
        if use_proba:
            return self.predict_proba(X)
        else:
            linpred = np.dot(X, self.trace['coeffs'].T) + self.trace['intercept']
            return norm.cdf(linpred)
        
    def get_coef(self):
        return np.mean(self.trace['coeffs'], axis=0)
    
    def get_intercept(self):
        return np.mean(self.trace['intercept'])
    
    def get_feature_coefficients(self):
        coef = self.get_coef()
        feature_names = self.X_train.columns
        return dict(zip(feature_names, coef))
    
    def get_significant_features(self, threshold=0.01):
        significant_coefs = self.get_significant_coef(threshold)
        feature_names = self.X_train.columns
        significant_feature_names = [feature_names[i] for i in np.where(np.abs(self.method.coef_) > threshold)[0]]
        return dict(zip(significant_feature_names, significant_coefs))
    
    def get_significant_coef(self, threshold=0.01):
        significant_coefficients_indices = np.where(np.abs(self.method.coef_) > threshold)[0]
        return self.method.coef_[significant_coefficients_indices]
    
    def plot_posterior(self):
        pm.plot_posterior(self.trace)
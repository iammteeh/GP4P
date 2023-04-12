
from types import TracebackType
from typing import Any, Optional, Type

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, roc_curve
from gplearn.genetic import SymbolicRegressor

class Regression:
    def __init__(self, X_train, X_test, y_train, method=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.method = self.set_method(method)
    
    def set_method(self, method):
        if method == "linear":
            return linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        elif method == "lasso":
            return linear_model.Lasso(alpha=0.01, max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "ridge":
            return linear_model.Ridge(alpha=0.01, max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "elastic":
            return linear_model.ElasticNet(alpha=0.01, max_iter=9999, copy_X=True, fit_intercept=True, normalize=False)
        elif method == "symbolic":
            return SymbolicRegressor(population_size=5000,
                            generations=20, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=42)

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
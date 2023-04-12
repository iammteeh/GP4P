
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
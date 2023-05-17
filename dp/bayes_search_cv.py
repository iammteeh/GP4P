import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from adapters.PyroMCMCRegressor import PyroMCMCRegressor
from domain.regression import ProbitRegressor
from dp.search_space_parametrization import get_model_search_space
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error


# Initialize a Random Forest Regressor
model = PyroMCMCRegressor()

search_space = get_model_search_space(type(model))

#model_pipeline = lambda model: Pipeline([('scaler', RobustScaler()), ('model', model)])
model_pipeline = Pipeline([('scaler', RobustScaler()), ('model', model)])

# Initialize the BayesSearchCV
bayes_search_cv = BayesSearchCV(PyroMCMCRegressor(), 
                    search_space, 
                    n_iter=50, 
                    n_jobs=8, 
                    cv=5, 
                    verbose=1, 
                    scoring='roc_auc', 
                    random_state=1)

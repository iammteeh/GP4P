import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from bayesify.pairwise import P4Preprocessing
from adapters.PyroMCMCRegressor import PyroMCMCRegressor
from application.bayesify_pipeline import bayesify_pipeline
from domain.regression import ProbitRegressor
from dp.search_space_parametrization import get_model_search_space
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score


# Initialize a Random Forest Regressor
preprocessing = P4Preprocessing()
model = PyroMCMCRegressor()
search_space = get_model_search_space(type(model))
scoring = make_scorer(r2_score, greater_is_better=True)

#model_pipeline = lambda model: Pipeline([('scaler', RobustScaler()), ('model', model)])
model_pipeline = Pipeline([('preprocessing', preprocessing), ('model', model)])

# Initialize the BayesSearchCV
bayes_search_cv = BayesSearchCV(model_pipeline, 
                    search_space, 
                    n_iter=50,
                    cv=5, 
                    scoring=scoring, 
                    random_state=1)

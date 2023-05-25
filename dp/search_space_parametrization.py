from sklearn.ensemble import RandomForestRegressor
from adapters.PyroMCMCRegressor import PyroMCMCRegressor
from domain.regression import ProbitRegressor
from skopt.space import Real, Integer, Categorical
from numpyro import distributions as dist
import random

base_prior_dist = Categorical(categories=['Normal', 'Gamma', 'Bernoulli', 'Exponential', 'Gumbel', 'Laplace', 'LogNormal', 'Weibull', 'Beta'])
infl_prior_dist = Categorical(categories=['Normal', 'Dirichlet', 'Multinomial', 'Binomial', 'Gamma'])
error_prior_dist = Categorical(categories=['Normal', 'Gamma'])
test_base_prior_dist = Categorical(categories=['Normal', 'HalfNormal'])
test_infl_prior_dist = Categorical(categories=['Normal', 'HalfNormal'])
test_error_prior_dist = Categorical(categories=['Normal', 'HalfNormal'])


def get_model_search_space(model, *args, **kwargs):
    search_space = {}
    print(f"Model: {model}")
    if issubclass(model, PyroMCMCRegressor):
        search_space = {
            'model__base_prior_dist': test_base_prior_dist,
            'model__base_prior_loc': Real(low=-1, high=1), # leave out later 
            'model__base_prior_scale': Real(low=0.1, high=1),
            'model__base_prior_concentration': Real(low=0.1, high=1),
            'model__base_prior_concentration0': Real(low=0.1, high=1),
            'model__base_prior_concentration1': Real(low=0.1, high=1),
            'model__base_prior_rate': Real(low=0.1, high=1),
            'model__base_prior_probs': Real(low=0.1, high=1),
            'model__infl_prior_dist': test_infl_prior_dist,
            'model__infl_prior_loc': Real(low=-1, high=1),
            'model__infl_prior_scale': Real(low=0.1, high=1),
            'model__infl_prior_concentration': Real(low=0.1, high=1),
            'model__infl_prior_rate': Real(low=0.1, high=1),
            'model__infl_prior_probs': Real(low=0.1, high=1),
            'model__error_prior_dist': test_error_prior_dist,
            'model__error_prior_loc': Real(low=-1, high=1),
            'model__error_prior_scale': Real(low=0.1, high=1),
            'model__error_prior_concentration': Real(low=0.1, high=1),
            'model__error_prior_rate': Real(low=0.1, high=1),
            'model__error_prior_probs': Real(low=0.1, high=1),
           # 'weighted_errs_per_sample': Categorical(categories=[True, False]),
           # 'weighted_rel_errs_per_sample': Categorical(categories=[True, False]),
        }
    elif isinstance(model, RandomForestRegressor):
        search_space = {'n_estimators': (10, 1000),
               'max_depth': (3, 50),
               'min_samples_split': (2, 10),
               'min_samples_leaf': (1, 10)
               }
        
    elif isinstance(model, ProbitRegressor):
        search_space = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'dual': [True, False],
            'tol': Real(1e-6, 1e-2, prior='log-uniform'),
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'fit_intercept': [True, False],
            'random_state': [random.randint(0, 10000)],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'class_weight': ['balanced', None],
            'max_iter': Integer(100, 1000),
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'warm_start': [True, False],
            'l1_ratio': Real(0, 1)
            }
    return search_space
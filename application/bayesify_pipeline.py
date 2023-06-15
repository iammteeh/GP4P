from domain.env import DUMMY_DATA
from adapters.preprocessing import prepare_dataset, preprocessing
from adapters.calculate_priors import Priors
from adapters.visualization import plot_coefs, scatter_plot, plot_feature_importance, plot_dist
from bayesify.pairwise import P4Preprocessing, get_feature_names_from_rv_id, print_baseline_perf, print_scores, get_err_dict, get_snapshot_dict
from adapters.PyroMCMCRegressor import PyroMCMCRegressor
from bayesify.pairwise import print_baseline_perf
from sklearn.pipeline import Pipeline
import seaborn as sns
import pandas as pd
import numpy as np
import time

def get_X_y():
    tips = sns.load_dataset("tips")
    tips = pd.get_dummies(tips)
    y = tips["tip"].to_numpy()
    feature_names = ["total_bill", "sex_Male", "smoker_Yes", "size"]
    X = tips[feature_names].to_numpy()
    return X, feature_names, y

#seed = np.random.randint(np.iinfo(np.int32).max)
seed = 0

ds = prepare_dataset(DUMMY_DATA)
feature_names = ds.get_feature_names() if not DUMMY_DATA else ds["feature_names"]
print(f"initial feature set length: {len(feature_names)}")
feature_names, X_train, X_test, y_train, y_test = preprocessing(ds, "2_poly", "none")    
# use ndarrays of X and y
X_train = X_train[1]
X_test = X_test[1]
y_train = y_train[1]
y_test = y_test[1]
print(f"X_train shape: {X_train.shape}")

def train_quick_model(X_train, feature_names, y_train, dummy=True):
    if dummy:
        X_train, feature_names, y_train = get_X_y()
    elif not dummy and X_train is not None and y_train.all() is not None:
        pass
    reg = PyroMCMCRegressor()
    mcmc_cores = 1
    reg.fit(
        X_train,
        y_train,
        mcmc_samples=100,
        mcmc_tune=200,
        feature_names=feature_names,
        mcmc_cores=mcmc_cores,
        random_key=seed,
        verbose=True,
    )
    return reg

def bayesify_pipeline():
    return Pipeline([('preprocessing', P4Preprocessing()), ('model', PyroMCMCRegressor())])

pipeline = bayesify_pipeline()
print_baseline_perf(X_train, y_train, X_test, y_test)
start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

#print(ds.all_configs)
#print(ds.position_map)
#print(ds.prototype_config)
#print(f"redundant feature: {ds.redundant_ft}")
#print(f"redundant feature names: {ds.redundant_ft_names}")
#print(f"alternative feature: {ds.alternative_ft}")
#print(f"alternative feature names: {ds.alternative_ft_names}")

# print_scores(model_name, type, sample_set_id, X, y)

base = ["MultiNormal, MultiStudentT", "Gumbel", "Beta"]
def new_baseline_performance(X, y, **model):
    # get last entry
    pass

print(pipeline.named_steps['model'].coef_ci(0.95))

# calculate prior weighted normal
priors = Priors(X_train, y_train, feature_names)
coef_prior, base_prior, error_prior, weighted_errs_per_sample, weighted_rel_errs_per_sample = priors.get_prior_weighted_normal(X_train, y_train, feature_names)
print(weighted_errs_per_sample)
print(weighted_rel_errs_per_sample)
# plot prior weighted normal
plot_dist(coef_prior, "coef_prior")
plot_dist(base_prior, "base_prior")
plot_dist(error_prior, "error_prior")
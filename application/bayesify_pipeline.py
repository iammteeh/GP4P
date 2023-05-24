from domain.env import DUMMY_DATA
from adapters.preprocessing import prepare_dataset, preprocessing
from bayesify.pairwise import P4Preprocessing, get_feature_names_from_rv_id, print_baseline_perf, print_scores, get_err_dict, get_snapshot_dict
from adapters.PyroMCMCRegressor import PyroMCMCRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns
import pandas as pd
import numpy as np

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
X_train, X_test, y_train, y_test = preprocessing(ds, "2_poly", "none")

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
pipeline.fit(X_train, y_train)

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
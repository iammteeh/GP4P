from adapters.preprocessing import prepare_dataset, preprocessing
from application.bayesify_pipeline import bayesify_pipeline
from sklearn.model_selection import train_test_split
from domain.model import Model
from domain.env import DUMMY_DATA
from domain.regression import Regression
from dp.bayes_search_cv import bayes_search_cv
from domain.metrics import get_metrics
import numpy as np
import time
try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass

ds = prepare_dataset(DUMMY_DATA)
feature_names = ds.get_feature_names() if not DUMMY_DATA else ds["feature_names"]
X_train, X_test, y_train, y_test = preprocessing(ds, "2_poly", "none")

print("Starting BayesSearchCV")
opt = bayes_search_cv

# plot the model

# Fit the BayesSearchCV to your data
opt.fit(X_train, y_train)

# Predict on the test set
start_time = time.time()
y_pred = opt.predict(X_test)
end_time = time.time()
print(f"Prediction time: {end_time - start_time} seconds")

# Evaluate the model
metrics = get_metrics(y_test, y_pred, type="regression")
for metric in metrics:
    print(f"{metric}: {metrics[metric]}")

# Get the best parameters
print('Best parameters found: ', opt.best_params_)
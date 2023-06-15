import numpy as np
from domain.env import REGRESSION, REGRESSION_PENALTY, COEFFICIENT_THRESHOLD, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, DUMMY_DATA
from domain.model import Model
from adapters.preprocessing import prepare_dataset, preprocessing
from adapters.plot_features import plot_train_test_errors, plot_regularization_path
from adapters.lasso_cv import lasso_cv
import time

ds = prepare_dataset(DUMMY_DATA)
#feature_names = ds.get_feature_names() if not DUMMY_DATA else ds["feature_names"]
feature_names, X_train, X_test, y_train, y_test = preprocessing(ds, "3_poly", "robust")

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

print("start training")
start = time.time()
train_errors, test_errors, beta_values = lasso_cv(X_train, X_test, y_train, y_test, X_train.shape[1], REGRESSION_PENALTY)
end = time.time()
print("end training")
print("time elapsed:", end - start, "s")

plot_train_test_errors(train_errors, test_errors, np.logspace(*REGRESSION_PENALTY))
plot_regularization_path(lambd_values=np.logspace(*REGRESSION_PENALTY), beta_values=beta_values)
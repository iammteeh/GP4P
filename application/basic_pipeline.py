import pandas as pd
import numpy as np
from itertools import combinations
from adapters.visualization import plot_lassoCV
from domain.env import REGRESSION, REGRESSION_PENALTY, ALPHAS, COEFFICIENT_THRESHOLD, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, USE_DUMMY_DATA
from domain.model import Model
from domain.regression import Regression
from adapters.preprocessing import prepare_dataset, preprocessing
import statsmodels.api as stats
from adapters.visualization import plot_train_test_errors, plot_regularization_path
import seaborn as sns 
import time

def init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="2_poly"):
    ds = prepare_dataset(use_dummy_data)
    #feature_names = ds.get_feature_names() if not DUMMY_DATA else ds["feature_names"]
    feature_names, X_train, X_test, y_train, y_test = preprocessing(ds, extra_features, "robust")
    print(f"len(feature_names): {len(feature_names)}")

    return ds, feature_names, X_train, X_test, y_train, y_test

def main():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=True)
    # use ndarrays of X and y
    X_train = X_train[1]
    X_test = X_test[1]
    y_train = y_train[1]
    y_test = y_test[1]
    model = Model(REGRESSION, ["mse_relative_to_mean", "mse_relative_to_variance", "mape", "r2"], ds, feature_names, y_test)
    #print(f"perform regression with {model.method}, {model.metrics} with {X_train.shape[1]} features: {X_train.columns}")
    lambda_values = np.logspace(*ALPHAS)
    with Regression(X_train, X_test, y_train, feature_names, model.method, lambda_values) as regression:
        # first do some linear regression to shrink the model
        print(f"fit with {regression.method}")
        start = time.time()
        
        regression.fit()
        
        end = time.time()
        print(f"Finished fitting in {end - start :.2f} seconds")

        y_pred = regression.predict()
        
        coef = regression.get_coef()
        #significant_coef = regression.get_significant_coef(COEFFICIENT_THRESHOLD)
        intercept = regression.get_intercept()
        #feature_coefficients = regression.get_feature_coefficients()
        #significant_features = regression.get_significant_features(COEFFICIENT_THRESHOLD)
        #new_features = significant_features.keys()
        model.coef = coef
        #print(f"regression coefficients: {coef} ({len(coef)} coefficients) \n")
        print(f"regression intercept: {intercept} \n")
        #print(f"regression feature coefficients: {feature_coefficients} \n")
        #print(f"regression significant coefficients: {significant_coef} ({len(significant_coef)} signicifant coefficient - {len(coef)-len(significant_coef)} filtered out) \n")
        #print(f"regression significant features: {significant_features} \n")

        print(f"regression coefficients: {coef} ({len(coef)} coefficients) \n")
        model.y_pred = y_pred
        print(model.eval())
        select_features = model.coef[model.coef != 0].index
        print(f"selected features: {select_features}")

        plot_lassoCV(model, regression)
if __name__ == "__main__":
    main()
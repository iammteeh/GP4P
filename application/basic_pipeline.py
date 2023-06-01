import pandas as pd
from itertools import combinations
from domain.env import REGRESSION, REGRESSION_PENALTY, COEFFICIENT_THRESHOLD, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, DUMMY_DATA
from domain.model import Model
from domain.regression import Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from adapters.preprocessing import prepare_dataset, preprocessing
import statsmodels.api as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, roc_curve
import decimal
from adapters.plot_features import plot_train_test_errors, plot_regularization_path
import seaborn as sns 
import time

ds = prepare_dataset(DUMMY_DATA)
feature_names = ds.get_feature_names()
X_train, X_test, y_train, y_test = preprocessing(ds, "3_poly", "robust")
model = Model(REGRESSION, ["mse_relative_to_mean", "mse_relative_to_variance", "mape", "r2"], ds, y_test)
#print(f"perform regression with {model.method}, {model.metrics} with {X_train.shape[1]} features: {X_train.columns}")
with Regression(X_train, X_test, y_train, feature_names, model.method, REGRESSION_PENALTY) as regression:
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


    model.y_pred = y_pred
    print(model.eval())

#plt.scatter(X_test.iloc[:,0], y_test)
#plt.plot(X_test.iloc[:,0], y_pred)
plt = sns.displot(y=y_pred)
plt = sns.displot(y=y_test)
#plt.xticks(())
#plt.yticks(())
# plot displot
#plt.show()
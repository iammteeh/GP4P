import pandas as pd
from itertools import combinations
from env import REGRESSION
from domain.model import Model
from domain.regression import Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from preprocessing import prepare_dataset, preprocessing
import statsmodels.api as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, roc_curve
import decimal
#import matplotlib.pyplot
import seaborn as sns 
import time


# see env.py for mode
# if MODE == "simple": panda dataframe
# if MODE != "simple": 
ds = prepare_dataset()
X_train, X_test, y_train, y_test = preprocessing(ds, extra_ft="polynomial")
model = Model(REGRESSION, ["mse", "mape", "r2"], ds, y_test)
#print(f"perform regression with {model.method}, {model.metrics} with {X_train.shape[1]} features: {X_train.columns}")
with Regression(X_train, X_test, y_train, model.method) as regression:
    # first do some linear regression to shrink the model
    print(f"fit with {regression.method}")
    start = time.time()
    
    regression.fit()
    
    end = time.time()
    print(f"Finished fitting in {end - start :.2f} seconds")

    y_pred = regression.predict()
    
    threshold = 1
    coef = regression.get_coef()
    significant_coef = regression.get_significant_coef(threshold)
    intercept = regression.get_intercept()
    feature_coefficients = regression.get_feature_coefficients()
    significant_features = regression.get_significant_features(threshold)
    new_features = significant_features.keys()
    model.coef = coef
    #print(f"regression coefficients: {coef} ({len(coef)} coefficients) \n")
    print(f"regression intercept: {intercept} \n")
    #print(f"regression feature coefficients: {feature_coefficients} \n")
    print(f"regression significant coefficients: {significant_coef} ({len(significant_coef)} signicifant coefficient - {len(coef)-len(significant_coef)} filtered out) \n")
    print(f"regression significant features: {significant_features} \n")


    model.y_pred = y_pred
    print(model.eval())

# then do symbolic regression to get a regression tree
REGRESSION = "symbolic"
X_train = X_train[new_features]
X_test = X_test[new_features]
with Regression(X_train, X_test, y_train, REGRESSION) as regression:
    regression.fit()
    program = regression.get_program()
    model.program = program
    print(f"regression program: {program}")
    print(model.plot_symbolic_program())


#plt.scatter(X_test.iloc[:,0], y_test)
#plt.plot(X_test.iloc[:,0], y_pred)
plt = sns.displot(y=y_pred)
plt = sns.displot(y=y_test)
#plt.xticks(())
#plt.yticks(())
# plot displot
#plt.show()
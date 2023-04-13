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
X_train, X_test, y_train, y_test = preprocessing(ds, extra_ft="none")
model = Model(REGRESSION, ["mse", "mape", "r2"], ds, y_test)

with Regression(X_train, X_test, y_train, model.method) as regression:
    print(f"fit with {regression.method}")
    start = time.time()
    
    regression.fit()
    
    end = time.time()
    print(f"Finished fitting in {end - start :.2f} seconds")

    y_pred = regression.predict()
    
    if REGRESSION == "symbolic":
        program = regression.get_program()
        model.program = program
        print(f"regression program: {program}")
    else:
        coef = regression.get_coef()
        model.coef = coef
        print(f"regression coefficients: {coef}")

    model.y_pred = y_pred

print(model.eval())
if REGRESSION == "symbolic":
    #features = domain.dependence_graph.get_features()
    print(model.plot_symbolic_program())


#plt.scatter(X_test.iloc[:,0], y_test)
#plt.plot(X_test.iloc[:,0], y_pred)
plt = sns.displot(y=y_pred)
plt = sns.displot(y=y_test)
#plt.xticks(())
#plt.yticks(())
# plot displot
#plt.show()
import pandas as pd
from itertools import combinations
from domain.model import Model
from regression import Regression
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
model = Model("linear", ["r2", "mape"], ds, y_test)

with Regression(X_train, X_test, y_train, model.method) as regression:
    print(f"fit with {regression.method}")
    start = time.time()
    regression.fit()
    end = time.time()
    print(f"Finished fitting in {end - start :.2f} seconds")

    y_pred = regression.predict()
    coef = regression.get_coef()

    model.y_pred = y_pred
    model.coef = coef

print(model.test_evaluation())


#plt.scatter(X_test.iloc[:,0], y_test)
#plt.plot(X_test.iloc[:,0], y_pred)
plt = sns.displot(y=y_pred)
plt = sns.displot(y=y_test)
#plt.xticks(())
#plt.yticks(())
# plot displot
#plt.show()
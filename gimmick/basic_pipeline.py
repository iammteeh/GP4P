import pandas as pd
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from preprocessing import prepare_dataset, preprocessing
import statsmodels.api as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, roc_curve
import decimal
#import matplotlib.pyplot
import seaborn as sns 
ds = prepare_dataset()
X_train, X_test, y_train, y_test = preprocessing(ds)

print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())

LR = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#Lasso = linear_model.Lasso(alpha=0.01, max_iter=1000, copy_X=True, fit_intercept=True, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
Lasso = linear_model.Lasso(alpha=0.01)
Lasso.fit(X_train, y_train)

y_pred = Lasso.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)

print(f"Coefficients: \n {Lasso.coef_}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)}")


#plt.scatter(X_test.iloc[:,0], y_test)
#plt.plot(X_test.iloc[:,0], y_pred)
plt = sns.displot(y=y_pred)
plt = sns.displot(y=y_test)
#plt.xticks(())
#plt.yticks(())
# plot displot
#plt.show()
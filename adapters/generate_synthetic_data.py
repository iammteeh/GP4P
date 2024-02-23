import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from domain.env import DATA_SLICE_AMOUNT
from pandas import DataFrame

def generate_synthetic_polynomial_data(n_samples, n_features, degree, noise):
    X = np.random.randint(2, size=(n_samples, n_features)) # binary features
    poly = PolynomialFeatures(degree, include_bias=False)
    X = poly.fit_transform(X)
    feature_names = [' x '.join(['{}'.format(pair[0],pair[1]) for pair in tuple if pair[1] != 0 ]) for tuple in [zip(range(X.shape[1]),p) for p in poly.powers_]]
    coefficients = np.random.rand(X.shape[1]) * 10
    y = X @ coefficients + np.random.normal(0, noise, n_samples)
    # ensure y is positive
    y = np.abs(y)
    # Calculating variance
    variance = np.var(y)
    print(f"Variance: {variance}")
    # Variance should be non-negative
    assert variance >= 0, "Calculated variance is negative, which should not happen."
    
    # concat X and y and return as DataFrame
    data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    columns = feature_names + ["y"]  # Adjusted to include the target variable
    return DataFrame(data, columns=columns)
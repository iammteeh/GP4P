import cvxpy as cp
import numpy as np
from sklearn.metrics import mean_squared_error as mse

def loss_fn(X, Y, beta):
    loss = cp.norm2(cp.matmul(X, beta) - Y)**2
    return loss

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse_(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def lasso_cv(X_train, X_test, Y_train, Y_test, n, lambda_values):
    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    lambda_values = np.logspace(*lambda_values)
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

    train_errors = []
    test_errors = []
    beta_values = []
    for v in lambda_values:
        print("fit for lambda:", v)
        lambd.value = v
        problem.solve(solver=cp.SCS)
        print("Problem status:", problem.status)
        train_errors.append(mse_(X_train, Y_train, beta))
        test_errors.append(mse_(X_test, Y_test, beta))
    return train_errors, test_errors, beta_values
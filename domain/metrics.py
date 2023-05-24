import sklearn.metrics
import numpy as np

def BIC(sigma_squared, n, k):
    return np.log(sigma_squared) + np.log(n) * k

def AIC(sigma_squared, n, k):
    return np.log(sigma_squared) + 2 * k

def VAIC(sigma_squared, n, k):
    return 1 - (np.log(sigma_squared) + 2 * k) / (np.log(sigma_squared) + np.log(n) * k)

def r2_adj(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def get_metrics(y_true, y_pred, type="regression"):
    if type == "regression":
        return {
            "mse": sklearn.metrics.mean_squared_error(y_true, y_pred),
            "mae": sklearn.metrics.mean_absolute_error(y_true, y_pred),
            "mape": sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred),
            "r2": sklearn.metrics.r2_score(y_true, y_pred),
            "r2_adj": r2_adj(sklearn.metrics.r2_score(y_true, y_pred), len(y_true), len(y_pred)),
            "BIC": BIC(sklearn.metrics.mean_squared_error(y_true, y_pred), len(y_true), len(y_pred)),
            "AIC": AIC(sklearn.metrics.mean_squared_error(y_true, y_pred), len(y_true), len(y_pred)),
            "explained_variance (ESS)": sklearn.metrics.explained_variance_score(y_true, y_pred),
        }
    elif type == "classification":
        return {
            "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
            "precision": sklearn.metrics.precision_score(y_true, y_pred),
            "recall": sklearn.metrics.recall_score(y_true, y_pred),
            "f1": sklearn.metrics.f1_score(y_true, y_pred),
            "auc": sklearn.metrics.roc_auc_score(y_true, y_pred),
            "mcc": sklearn.metrics.matthews_corrcoef(y_true, y_pred)
        }


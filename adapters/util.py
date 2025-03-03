import numpy as np
import math
# import sqrt
from sklearn.metrics import r2_score, mean_squared_error

def weighted_avg_and_std(values, weights, gamma=1):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if gamma != 1:
        weights = np.power(weights, gamma)
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    if variance <= 0:
        sqr_var = 0.0
    else:
        sqr_var = math.sqrt(variance)
    return average, sqr_var

def is_positive_semi_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)

def get_feature_names_from_rv_id(ft_inter):
    new_ft_inter = ft_inter.replace("_log__", "")
    new_ft_inter = new_ft_inter.replace("active_", "")
    new_ft_inter = new_ft_inter.replace("_scale", "")
    new_ft_inter = new_ft_inter.replace("influence_", "")
    result = new_ft_inter.split("&")
    return result

def score_mape(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    mape = np.mean(np.abs((y_true - y_predicted) / y_true)) * 100
    return mape

def score_rmse(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    rms = math.sqrt(mean_squared_error(y_true, y_predicted))
    return rms

def get_err_dict_from_predictions(y_pred, xs, ys):
    mape = score_mape(None, xs, ys, y_pred)
    rmse = score_rmse(None, xs, ys, y_pred)
    r2 = r2_score(ys, y_pred)
    errors = {
        "r2": r2,
        "mape": mape,
        "rmse": rmse,
        "raw": {"x": xs, "y_pred": y_pred, "y_true": ys},
    }
    return errors

def get_err_dict(reg, xs, ys):
    y_pred = reg.predict(xs)
    errors = get_err_dict_from_predictions(y_pred, xs, ys)
    return errors

def print_scores(model_name, reg, sample_set_id, xs, ys, print_raw=False):
    errors = get_err_dict(reg, xs, ys)
    for score_id, score in errors.items():
        if not print_raw and "raw" in score_id:
            continue
        print(
            "{} {} set {} score: {}".format(model_name, sample_set_id, score_id, score)
        )
    print()


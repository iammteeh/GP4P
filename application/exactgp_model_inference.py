from domain.env import MODELDIR
import numpy as np
import torch
from domain.env import SELECTED_FEATURES
from application.gpytorch_pipeline import get_data
from adapters.gpytorch.gp_model import MyExactGP
from domain.feature_model.feature_modeling import inverse_map
from adapters.gpytorch.util import decompose_matrix, get_beta, get_thetas, LFSR, get_PPAAs, map_inverse_to_sample_feature, get_groups, group_RATE, get_posterior_variations, interaction_distant, measure_subset
from adapters.sklearn.dimension_reduction import kernel_pca
from adapters.gpytorch.plotting import kde_plots, plot_combined_pdf, plot_density, plot_interaction_pdfs, mean_and_confidence_region
from domain.metrics import get_metrics, gaussian_log_likelihood
import random
from time import time

from scipy.stats import pointbiserialr

file_name = "GPY_linear_weighted_matern52_simple_100__20240324-110929"
model_file = f"{MODELDIR}/{file_name}.pth"

# get data that produced the model
#TODO: ensure that the input data is the same as the data in the model
data = get_data(get_ds=True)
ds, X_train, X_test, y_train, y_test, feature_names = data

# load the model
model = MyExactGP(X_train, y_train, feature_names, likelihood="gaussian", kernel="matern52", mean_func="linear_weighted") # has to be the same as the model that produced the model_file
model.load_state_dict(torch.load(model_file))

# get the posterior mean and variance
model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test)
    posterior_mean = posterior.mean
    posterior_variance = posterior.variance

# get the metrics
metrics = get_metrics(y_test, posterior_mean, posterior_variance)
print(metrics)
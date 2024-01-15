from domain.env import MODELDIR
import numpy as np
import torch
from domain.env import SELECTED_FEATURES
from application.fully_bayesian_gp import get_data, choose_model
from adapters.gpytorch.gp_model import GPRegressionModel, SAASGP
from domain.feature_model.feature_modeling import inverse_map
from adapters.gpytorch.util import decompose_matrix, beta_distances, get_alphas, get_betas, get_beta, get_thetas, LFSR, get_PPAAs, map_inverse_to_sample_feature, get_groups, group_RATE
from adapters.sklearn.dimension_reduction import kernel_pca
from adapters.gpytorch.plotting import plot_prior, plot_pairwise_posterior_mean_variances, mean_and_confidence_region, grid_plot, kde_plots
from domain.metrics import get_metrics, gaussian_log_likelihood
from time import sleep
import random

from scipy.stats import pointbiserialr

file_name = "SAASGP_linear_weighted_matern52_simple_ARD=False__20240111-165447" # after refactoring the preprocessing
model_file = f"{MODELDIR}/{file_name}.pth"


# get data that produced the model
#TODO: ensure that the data is the same as the data that produced the model
data = get_data()
X_train, X_test, y_train, y_test, feature_names = data
# slice X_test such that it has the same shape as X_train
if len(X_test) > len(X_train):
    X_test = X_test[:len(X_train)]
elif len(X_test) < len(X_train):
    X_train = X_train[:len(X_test)]

# init model and load state dict
model = choose_model(model="SAASGP", data=data)
model.load_state_dict(torch.load(model_file))

model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test)
    CONFIDENCE = get_metrics(posterior, y_test, posterior.mixture_mean.squeeze(), type="GP")["explained_variance"]
    confidence_region = posterior.mvn.confidence_region()
    # create dimensional model
    dims = len(model.X.T)
    dimensional_model = {}
    for dim in range(dims):
        dimensional_model[dim] = {}
        dimensional_model[dim]["X"] = X_train[:, dim]
        dimensional_model[dim]["y"] = y_train
        dimensional_model[dim]["X_test"] = X_test[:, dim]
        dimensional_model[dim]["mean"] = posterior.mean[dim]
        dimensional_model[dim]["variance"] = posterior.variance[dim]
        dimensional_model[dim]["covariance"] = posterior.covariance_matrix[dim]
        dimensional_model[dim]["lower"] = confidence_region[0][dim]
        dimensional_model[dim]["upper"] = confidence_region[1][dim]


U, lam, V = decompose_matrix(dimensional_model[0]["covariance"]) # example for first dimension

# compute group RATE according to Crawford et al. 2019
# j is the feature group that occurs in the rows of the validation set
j, groups = get_groups(X_test, SELECTED_FEATURES)
print(f"groups: {groups}")
group_rate = group_RATE(dimensional_model[0]["mean"], U, j)
print(f"group rate: {group_rate}")

# inference according to BAKR https://github.com/lorinanthony/BAKR/blob/master/Tutorial/BAKR_Tutorial.R
explained_var = np.cumsum(np.array(lam) / np.sum(np.array(lam)))
p = len(model.X.T)
# estimate, how much of the explained variance is explained by p components
p_explained_var = explained_var[p - 1]
print(f"{p_explained_var}.2f of the variance is explained by {p} components (the base features)")
q = np.where(explained_var >= CONFIDENCE)[0][0] + 1 # number of principal components to explain confidential proportion of variance
qq = next(x[0] for x in enumerate(explained_var) if x[1] > CONFIDENCE) + 1
qqq = next(i + 1 for i, var in enumerate(explained_var) if var >= CONFIDENCE)
Lambda = np.diag(np.sort(lam)[::-1])[:q] # diagonal matrix with first q eigenvalues 
full_latent_space = U @ lam @ V.T # full latent space
print(f"full latent space: {full_latent_space}")
print(f"full latent space shape: {full_latent_space.shape}")
U = U[:, :q] # first q columns
B = inverse_map(model.X.T, U)
Laplace_approximation = B @ B.T
thetas = get_thetas(dimensional_model[0]["covariance"], q)
betas = get_beta(B, thetas)
#alphas = get_alphas(dimensional_model[0]["covariance"], q)
beta_j, lambda_j = map_inverse_to_sample_feature(betas, B, thetas)



# calculate distances between betas
# the KLD between the mixture distribution and the posterior distribution of the feature combination in the validation set is the distance
# between a B[-j] and a B[j] (see BAKR)
#KLD = np.zeros(betas.shape[0])
#for i in range(len(betas)):
#    KLD[i] = np.linalg.norm(betas[i, :] - betas[i + 1, :]) # source: https://stats.stackexchange.com/questions/81691/kullback-leibler-divergence-between-two-multivariate-gaussians


print(f"betas: {betas}")
print(f"lambdas: {lambda_j}")
# threshold the lambdas and output a list of features that are influencial
t = 5
influencial_features = {}
for i in range(len(lambda_j)):
    dimensional_influence = []
    for j in range(len(lambda_j[i])):
        # Betrag von lambda
        if torch.abs(lambda_j[i][j]) > t:
            dimensional_influence.append((feature_names[j], lambda_j[i][j]))
    influencial_features[i] = dimensional_influence

print(f"influencial features: {influencial_features[0]}")

lfsr = LFSR(betas)
print(f"lfsr: {lfsr}")

print(f"look at influencial features on different significance levels")
# count from 0.8 to 0.995 in steps of 0.005
for s in range(970, 996, 5):
    feature_idx, PPAAs = get_PPAAs(betas, tuple(feature_names), sigval=s/1000)
    # fully print the PPAAs
    for i in range(1):
        print(f"s={s/1000}: PPAA {i} with {sum(PPAAs[i*random.randint(0, 200)])} influencial features")
        # output the names of the influencial features
        non_influentials = []
        for j in range(len(PPAAs[i])):
            if PPAAs[i][j] > 0:
                print(f"feature {feature_idx[j]}:{feature_names[feature_idx[j]]} on level {j} - influencial? {PPAAs[i][j]} with beta-value {betas[i][j]}")
            else:
                non_influentials.append((feature_names[feature_idx[j]], betas[i][j]))
        print(f"non influencial features: {non_influentials}")

# plot prior
#plot_prior(model, X_test, y_test)
# plot posterior mean and variance of dimensions 0 and 1
#plot_pairwise_posterior_mean_variances(0, 1, posterior, X_test, y_test)

# reduce dimensionality of X_train and X_test
#X_kpca, kernel_features, X_train_inverse_transform = kernel_pca(X_train, feature_names, kernel="poly")
#X_test_kpca, kernel_features, X_test_inverse_transform = kernel_pca(X_test, feature_names, kernel="poly")
#dims = len(X_train_inverse_transform.T)



#for dim in range(len(model.median_lengthscale)):
#    lengthscale_sorted = torch.sort(model.median_lengthscale)
##    lengthscale_argsorted = torch.argsort(model.median_lengthscale)
#    print(f"lengthscale sorted: {lengthscale_sorted[lengthscale_argsorted[dim-1]]}")
#dimensional_model = {}
#for dim in range(dims):
##    dim_group = (dim, random.randint(0, len(model.X.T)-1)) # choose a random dimension and a random feature in that dimension
#    dimensional_model[dim] = {}
#    dimensional_model[dim]["X"] = X_train[:, dim_group]
#    dimensional_model[dim]["y"] = y_train
#    dimensional_model[dim]["X_test"] = X_test[:, dim_group]
##    dimensional_model[dim]["mean"] = posterior.mean[:, dim_group]
#    dimensional_model[dim]["variance"] = posterior.variance[:, dim_group]
#    dimensional_model[dim]["covariance"] = posterior.covariance_matrix[:, dim_group]
#    dimensional_model[dim]["lower"] = confidence_region[0][:, dim_group]
#    dimensional_model[dim]["upper"] = confidence_region[1][:, dim_group]


# get metrics
print(f"get metrics...")
# check for shapes
print(f"posterior mean shape: {posterior.mixture_mean.shape}")
print(f"y_test shape: {y_test.shape}")
metrics = get_metrics(posterior, y_test, posterior.mixture_mean.squeeze(), type="regression")
print(f"metrics: {metrics}")
kernel_lengthscales = [(feature_names[i],model.median_lengthscale[i]) for i in range(len(model.median_lengthscale))]
# make list of tuples with feature name and lengthscale sorted by lengthscale
lengthscale_sorted = list(torch.sort(model.median_lengthscale)[0])
lengthscale_argsorted = list(torch.argsort(model.median_lengthscale))
feature_names = list(feature_names)
sorted_lengthscale = [(feature_names[lengthscale_argsorted[i]], lengthscale_sorted[i]) for i in range(len(model.median_lengthscale))]
for feature, lengthscale in sorted_lengthscale:
    print(f"lengthscale of {feature}: {lengthscale}")

#for dim in range(len(dimensional_model)):
#    mean_and_confidence_region(dimensional_model[dim]["X_test"], dimensional_model[dim]["X"], dimensional_model[dim]["y"], dimensional_model[dim]["mean"], dimensional_model[dim]["lower"], dimensional_model[dim]["upper"])

# plot feature wise
#grid_plot(X_train, y_train, X_test, posterior.mean, confidence_region)
kde_plots(X_train, y_train)
print(f"done.")
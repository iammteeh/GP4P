from domain.env import MODELDIR
import numpy as np
import torch
from scipy.stats import norm, multivariate_normal as mvn
from torch.distributions import MultivariateNormal, Normal
from copulae.core.linalg import cov2corr, corr2cov
from copulae.core.misc import rank_data
from domain.env import SELECTED_FEATURES
from application.fully_bayesian_gp import get_data
from adapters.gpytorch.gp_model import SAASGP
from domain.feature_model.feature_modeling import inverse_map
from adapters.gpytorch.util import decompose_matrix, get_beta, get_thetas, LFSR, get_PPAAs, map_inverse_to_sample_feature, get_groups, group_RATE, get_posterior_variations, interaction_distant, measure_subset
from adapters.sklearn.dimension_reduction import kernel_pca
from adapters.gpytorch.plotting import kde_plots, plot_combined_pdf, plot_density, plot_interaction_pdfs, mean_and_confidence_region
from domain.metrics import get_metrics, gaussian_log_likelihood
import random
from time import time

from scipy.stats import pointbiserialr

#file_name = "SAASGP_linear_weighted_matern52_simple_ARD=False__20240111-165447" # after refactoring the preprocessing n =1000
file_name = "SAASGP_linear_weighted_matern52_simple_ARD=False__20240329-195132" # n = 100
#file_name = "SAASGP_linear_weighted_matern52_additive_ARD=False__20240119-155321"
model_file = f"{MODELDIR}/{file_name}.pth"


# get data that produced the model
#TODO: ensure that the input data is the same as the data in the model
data = get_data(get_ds=True)
ds, X_train, X_test, y_train, y_test, feature_names = data

# init model and load state dict
model = SAASGP(X_train, y_train, feature_names)
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
        dimensional_model[dim]["feature_name"] = feature_names[dim]
        dimensional_model[dim]["y"] = y_train
        dimensional_model[dim]["X_test"] = X_test[:, dim]
        dimensional_model[dim]["mean"] = posterior.mean[dim]
        dimensional_model[dim]["variance"] = posterior.variance[dim]
        dimensional_model[dim]["std"] = torch.sqrt(posterior.variance[dim])
        dimensional_model[dim]["covariance"] = posterior.covariance_matrix[dim]
        dimensional_model[dim]["correlation"] = cov2corr(np.array(dimensional_model[dim]["covariance"]))
        dimensional_model[dim]["marginal"] = Normal(dimensional_model[dim]["mean"], torch.sqrt(posterior.variance[dim])).cdf(dimensional_model[dim]["mean"])
        dimensional_model[dim]["inverse_transform"] = Normal(dimensional_model[dim]["mean"], torch.sqrt(posterior.variance[dim])).icdf(dimensional_model[dim]["marginal"])
        dimensional_model[dim]["lower"] = confidence_region[0][dim]
        dimensional_model[dim]["upper"] = confidence_region[1][dim]

# co-generated content with github's copilot:
# for the copula or CDFs we need the eigendecomposition of the covariance matrix
# Co-variance matrix = U @ lam @ V^T
# Covariance matrix is symmetric and positive semi-definite, so we can decompose it into U @ lam @ V^T
# where U is the eigenvectors, lam is the eigenvalues and V is the transpose of the eigenvectors
# the eigenvectors are the directions of the latent space that are most important (the directions of the most variance) while the eigenvalues are the amount of variance in that direction
# we can use the eigendecomposition of the covariance matrix to get the betas from the inverse map
# the betas are the weights of the features in the latent space

# or the low rank pivoted cholesky decomposition of the covariance matrix
        
# copula
# math from https://copulae.readthedocs.io/en/latest/explainers/introduction.html
# compute CDF from marginals
# posterior predictive checks

U, lam, V = decompose_matrix(dimensional_model[0]["covariance"]) # example for first dimension

# compute group RATE according to Crawford et al. 2019
# j is the feature group that occurs in the rows of the validation set
j, groups = get_groups(X_test, SELECTED_FEATURES)
print(f"groups: {groups}")
group_rate = group_RATE(dimensional_model[0]["mean"], U, j) #TODO: How to visualize the group rate or KLD in general?
print(f"group rate: {group_rate}")
# use groups to calculate 2 posteriors, one with the group and one without the group, measure the distance and plot the PDFs


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

# instead of eigendecomposition of the covariance use low rank pivoted cholesky decomposition
L = np.linalg.cholesky(dimensional_model[0]["covariance"])
B = inverse_map(model.X.T, L)
thetas_ = torch.diag(lam) @ L.T
betas_ = get_beta(B, thetas_)

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
# take low rank approximation
betas = betas_
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
#plot_prior(model, X_test, y_test)dimensional_model
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
metrics = get_metrics(posterior, y_test, posterior.mixture_mean.squeeze(), type="GP")
print(f"metrics: {metrics}")
kernel_lengthscales = [(feature_names[i],model.median_lengthscale[i]) for i in range(len(model.median_lengthscale))]
# make list of tuples with feature name and lengthscale sorted by lengthscale
lengthscale_sorted = list(torch.sort(model.median_lengthscale)[0])
lengthscale_argsorted = list(torch.argsort(model.median_lengthscale))
feature_names = list(feature_names)
sorted_lengthscale = [(feature_names[lengthscale_argsorted[i]], lengthscale_sorted[i]) for i in range(len(model.median_lengthscale))]
sorted_features = [feature_names[lengthscale_argsorted[i]] for i in range(len(model.median_lengthscale))]
print(f"sorted features: {sorted_features} having lengthscales: {lengthscale_sorted}")
for feature, lengthscale in sorted_lengthscale:
    print(f"lengthscale of {feature}: {lengthscale}")

#for dim in range(len(dimensional_model)):
#    mean_and_confidence_region(dimensional_model[dim]["X_test"], dimensional_model[dim]["X"], dimensional_model[dim]["y"], dimensional_model[dim]["mean"], dimensional_model[dim]["lower"], dimensional_model[dim]["upper"])
exit(0)
## PLOTTING SECTION
# plot feature wise
#grid_plot(X_train, y_train, X_test, posterior.mean, confidence_region)
kde_plots(X_train, y_train)
# plot combined
selected_dimensions = [0,1,2]
for dim in selected_dimensions:
    mean = np.mean(dimensional_model[dim]["mean"].detach().numpy())
    variance = np.median(dimensional_model[dim]["variance"].detach().numpy())
    feature_name = dimensional_model[dim]["feature_name"]
    plot_density(mean, variance, feature_name)
dim_pairs = [
    (0,1,3), (0,1,4), (0,1,5),
]
for tuple in dim_pairs:
    dimensional_submodel = {}
    for dim in tuple:
        dimensional_submodel[dim] = dimensional_model[dim]
    plot_combined_pdf(dimensional_submodel)
plot_combined_pdf(dimensional_model)
opposites, interactions = get_posterior_variations(model, X_train, [(1,0),(3,1)])
measure_subset(model, ds, [(1,0),(3,1)])
dimensional_submodel = {}
dimensional_submodel["0"] = {}
dimensional_submodel["0"]["mean"] = opposites.mixture_mean.detach().numpy()
dimensional_submodel["0"]["std"] = np.sqrt(opposites.mixture_variance.detach().numpy())
dimensional_submodel["0"]["feature_name"] = "without interaction"
dimensional_submodel["1"] = {}
dimensional_submodel["1"]["mean"] = interactions.mixture_mean.detach().numpy()
dimensional_submodel["1"]["std"] = np.sqrt(interactions.mixture_variance.detach().numpy())
dimensional_submodel["1"]["feature_name"] = "with interaction"
#plot_combined_pdf(dimensional_submodel)
subset_that_lowers_y = [(1,0), (3,1)]
subset_that_increases_y = [(0,0),(1,1), (3,0), (5,1), (10,0)]
plot_interaction_pdfs(dimensional_submodel, subset_that_increases_y)
print(f"opposites and interactions diverge at {interaction_distant(model, X_test, subset_that_increases_y)}")
for dim in range(len(dimensional_model)):
    mean_and_confidence_region(dimensional_model[dim]["X_test"], dimensional_model[dim]["X"], dimensional_model[dim]["y"], dimensional_model[dim]["mean"], dimensional_model[dim]["lower"], dimensional_model[dim]["upper"])
print(f"done.")
from jax import numpy as jnp
from jax import vmap
import numpy as np
import torch
from domain.feature_model.feature_modeling import get_feature_model

def analyze_posterior(model, X_test, y_test, features):
    words, opposites, literals, interactions = get_feature_model(X_test, features)
    print(f"words: {words}")
    print(f"interactions: {interactions}")

    word_posterior = model.posterior(words)
    mean_words, variance_words = word_posterior.mixture_mean.numpy(), word_posterior.mixture_variance.numpy()
    mean_words = jnp.mean(mean_words, axis=0)
    variance_words = jnp.median(variance_words, axis=0)

    opposite_posterior = model.posterior(opposites)
    mean_opposites, variance_opposites = opposite_posterior.mixture_mean.numpy(), opposite_posterior.mixture_variance.numpy()
    mean_opposites = jnp.mean(mean_opposites, axis=0)
    variance_opposites = jnp.median(variance_opposites, axis=0)

    literals_posterior = model.posterior(literals)
    mean_literals, variance_literals = literals_posterior.mixture_mean.numpy() , literals_posterior.mixture_variance.numpy()
    mean_literals = jnp.mean(mean_literals, axis=0)
    variance_literals = jnp.median(variance_literals, axis=0)

    interactions_posterior = model.posterior(interactions)
    mean_interactions, variance_interactions = interactions_posterior.mixture_mean.numpy(), interactions_posterior.mixture_variance.numpy()
    mean_interactions = jnp.mean(mean_interactions, axis=0)
    variance_interactions = jnp.median(variance_interactions, axis=0)

    return (mean_opposites, variance_opposites), (mean_interactions, variance_interactions)

def explore_data(data):
    # if data is from get_data()
    print(f"shape of X_train: {data[0].shape}")
    print(f"shape of X_test: {data[1].shape}")
    # if data is any df, numpy array, tensor
    print(f"shape of data: {data.shape}")
    # return all the data
    print(f"{data[:]}")
    # slice for shape (d,)
    print(f"slice of data: {data[0]}")
    print(f"{data[1:]}") # all but first
    i = 6
    print(f"{data[i]}") # i-th
    print(f"{data[:i]}") # all but i-th
    print(f"{data[i:]}") # i-th and all after
    print(f"{data[:1]}") # first
    print(f"{data[:-1]}") # all but last
    print(f"{data[-1:]}") # last
    # for shape (n, d)
    print(f"shape of data: {data.shape[0]} rows, {data.shape[1]} columns")
    # slice of data
    print(f"slice of data: {data[0:5, 0]}") # first 5 rows of first column
    # convert tensor to numpy
    print(f"numpy: {data.numpy()}")

def analyze_model(model, active_dims):
    p, n, m = model.X.shape[1], model.X.shape[0], len(active_dims)
    num_coefficients = p + m * (m - 1) // 2

    probe = jnp.zeros((2 * p + 2 * m * (m - 1)), p)
    vec = jnp.zeros((num_coefficients, 2 * p + 2 * m * (m - 1)))

def analyze_mixture(model, X_test, y_test):
    # get mixture mean and variance
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        observed_pred = model.likelihood(model(X_test))
        mean = observed_pred.mean
        var = observed_pred.variance
        posterior = model.posterior(X_test)
        print(f"kernel lengthscales: {model.covar_module.base_kernel.lengthscale}")
        print(f"kernel outputscale: {model.covar_module.outputscale}")
        print(posterior.mean.shape)
        print(posterior.variance.shape)
        print(f"Ground truth:     {y_test.squeeze(-1)}")
        print(f"Mixture mean:     {posterior.mean.squeeze(-1)}")
        print(f"Mixture variance: {posterior.variance.squeeze(-1)}")

        # analyze mixture mean and variance

def analyze_dimensions(model, dimension, samples):
    vmap_args = (
        samples["mean"],
        samples["variance"],
        samples["lengthscale"],
        samples["outputscale"],
    )
    vmap_args = vmap(lambda x: x[:, dimension])(vmap_args) # get samples for dimension

    for i, arg in enumerate(vmap_args):
        print(f"sample {i}: {arg}")
    
def analyze_dimensionality(model, samples):
    vmap_args = (
        samples["mean"],
        samples["variance"],
        samples["lengthscale"],
        samples["outputscale"],
    )
    vmap_args = vmap(lambda x: x[:, 0])(vmap_args) # get samples for dimension

    for i, arg in enumerate(vmap_args):
        print(f"sample {i}: {arg}")

def analyze_dimensions_v2(model, samples):
    vmap_args = (
        samples["mean"],
        samples["variance"],
        samples["lengthscale"],
        samples["outputscale"],
    )
    # iterate over dimensions and get samples for pairs of dimensions
    for i in range(vmap_args[0].shape[1]):
        for j in range(i + 1, vmap_args[0].shape[1]):
            vmap_args_ij = vmap(lambda x: x[:, i])(vmap_args)
            vmap_args_ij = vmap(lambda x: x[:, j])(vmap_args_ij)
            for k, arg in enumerate(vmap_args_ij):
                print(f"sample {k}: {arg}")

def analyze_mixtures(bayesian_posterior, dim_tuple):
    """"
    get the mixture mean and variance for a given dimension tuple by adding
    the mean and variance of the individual dimensions and subtracting the
    covariance of the dimensions.
    """
    mean = bayesian_posterior.mixture_mean
    variance = bayesian_posterior.mixture_variance
    for i in dim_tuple:
        mean += bayesian_posterior.mean[:, i]
        variance += bayesian_posterior.variance[:, i]
    for i in range(len(dim_tuple)):
        for j in range(i + 1, len(dim_tuple)):
            mean -= bayesian_posterior.covariance[:, i, j]
            variance -= bayesian_posterior.covariance[:, i, j]
    return mean, variance


def waic(model, likelihood, X, Y):
    model.eval()
    with torch.no_grad():
        output = model(X)
        predictive_mean = output.mean
        predictive_var = output.variance
        error = Y - predictive_mean
        log_likelihoods = -0.5 * torch.log(2 * np.pi * predictive_var) - 0.5 * (error**2) / predictive_var
        lppd = torch.sum(torch.log(torch.mean(torch.exp(log_likelihoods), dim=0)))
        p_waic = torch.sum(torch.var(log_likelihoods, dim=0))
        waic = -2 * (lppd - p_waic)
    return waic.item()

def get_posterior_dimension(posterior):
    posterior_dimwise = []
    for dim in range(posterior.mean.shape[1]):
        beta_dict = {}
        beta_dict["mu_j"] = posterior.mean[dim]
        print(f"mu_j: {beta_dict['mu_j']}")
        print(f"mu_j shape: {beta_dict['mu_j'].shape}")
        #beta_dict['mu_minus_j'] = np.delete(posterior.mean.numpy(), dim, axis=0)
        # for beta_dict['mu_minus_j'] we take all but the dim-th row of the mean vector with torch
        beta_dict["mu_minus_j"] = torch.cat((posterior.mean[:dim], posterior.mean[dim + 1:]), dim=0)
        print(f"mu_minus_j: {beta_dict['mu_minus_j']}")
        print(f"mu_minus_j shape: {beta_dict['mu_minus_j'].shape}")
        beta_dict["mu_minus_j"] = np.delete(posterior.mean.numpy(), dim, axis=0)
        print(f"mu_minus_j: {beta_dict['mu_minus_j']}")
        print(f"mu_minus_j shape: {beta_dict['mu_minus_j'].shape}")
        Sigma = posterior.covariance_matrix
        print(f"Sigma: {Sigma}")
        print(f"Sigma shape: {Sigma.shape}")
        Sigma_j = Sigma[dim]
        print(f"Sigma_j: {Sigma_j}")
        print(f"Sigma_j shape: {Sigma_j.shape}")
        Sigma_minus_j = np.delete(Sigma.numpy(), dim, axis=0)
        print(f"Sigma_minus_j: {Sigma_minus_j}")
        print(f"Sigma_minus_j shape: {Sigma_minus_j.shape}")
        # Sigma_minus_j with torch
        Sigma_minus_j = torch.cat((posterior.covariance_matrix[:dim], posterior.covariance_matrix[dim + 1:]), dim=0)
        print(f"Sigma_minus_j: {Sigma_minus_j}")
        print(f"Sigma_minus_j shape: {Sigma_minus_j.shape}")
        #Sigma_minus_j = np.delete(np.delete(Sigma, dim, axis=0), dim, axis=1)
        #Sigma_j_minus_j = np.delete(Sigma[dim, :], dim, axis=0)
        #Sigma_minus_j_j = np.delete(Sigma[:, dim], dim, axis=0)
        beta_dict["beta_j"] = posterior.lazy_covariance_matrix[dim].cholesky(upper=False)
        print(f"beta_j: {beta_dict['beta_j']}")
        print(f"beta_j shape: {beta_dict['beta_j'].shape}")
        # for beta_dict['beta_minus_j'] we take all but the dim-th row and column of the covariance matrix with torch.cat
        beta_dict["beta_minus_j"] = Sigma_minus_j.cholesky(upper=False)
        #beta_dict["beta_minus_j"] = posterior.lazy_covariance_matrix[:dim].cholesky(upper=False)
        print(f"beta_minus_j: {beta_dict['beta_minus_j']}")
        print(f"beta_minus_j shape: {beta_dict['beta_minus_j'].shape}")
        posterior_dimwise.append(beta_dict)
    return posterior_dimwise

def get_posterior_dimension_v2(posterior):
    posterior_dimwise = []
    for dim in range(posterior.mean.shape[1]):
        beta_dict = {}
        beta_dict["mu_j"] = posterior.mean[dim] # has shape (n,1)
        beta_dict["mu_minus_j"] = torch.cat((posterior.mean[:dim], posterior.mean[dim + 1:]), dim=0) # has shape (d-1,n,1)
        # alt: beta_dict["mu_minus_j"] = np.delete(posterior.mean.numpy(), dim, axis=0)
        Sigma = posterior.covariance_matrix # has shape (d,n,n)
        Sigma_j = Sigma[dim] # has shape (n,n)
        Sigma_minus_j = torch.cat((posterior.covariance_matrix[:dim], posterior.covariance_matrix[dim + 1:]), dim=0) # has shape (d-1,n,n)
        Sigma_minus_j = np.delete(Sigma.numpy(), dim, axis=0)
        beta_dict["beta_j"] = posterior.lazy_covariance_matrix[dim].cholesky(upper=False) # is TriangularLinearOperator with shape (n,n)
        #beta_dict["beta_minus_j"] = Sigma_minus_j.cholesky(upper=False) # is Tensor with shape (d-1,n,n)
        beta_dict["beta_minus_j"] = np.linalg.cholesky(Sigma_minus_j) # is Tensor with shape (d-1,n,n)
        #beta_dict["Lambda_j"] = posterior.lazy_covariance_matrix[dim].inv_matmul(torch.eye(posterior.lazy_covariance_matrix[dim].size(-1))) # is Tensor with shape (n,n) and represents the precision matrix
        beta_dict["Lambda"] = torch.linalg.inv(Sigma)
        print(f"Lambda: {beta_dict['Lambda']}")
        print(f"Lambda shape: {beta_dict['Lambda'].shape}")
        beta_dict["Lambda_j"] = beta_dict["Lambda"][dim] # is Tensor with shape (n,n) and represents the precision matrix
        print(f"Lambda_j: {beta_dict['Lambda_j']}")
        print(f"Lambda_j shape: {beta_dict['Lambda_j'].shape}")
        beta_dict["Lambda_j"] = torch.linalg.inv(posterior.lazy_covariance_matrix[dim].to_dense()) # is Tensor with shape (n,n) and represents the precision matrix
        print(f"Lambda_j: {beta_dict['Lambda_j']}")
        print(f"Lambda_j shape: {beta_dict['Lambda_j'].shape}")
        #beta_dict["Lambda_minus_j"] = torch.tensor(Sigma_minus_j).inv_matmul(torch.eye(Sigma_minus_j.size(-1))) # is Tensor with shape (d-1,n,n)
        beta_dict["Lambda_minus_j"] = torch.linalg.inv(torch.tensor(Sigma_minus_j)) # is Tensor with shape (d-1,n,n)
        print(f"Lambda_minus_j: {beta_dict['Lambda_minus_j']}")
        print(f"Lambda_minus_j shape: {beta_dict['Lambda_minus_j'].shape}")
        #beta_dict["lambda_j"] = posterior.lazy_covariance_matrix[dim].inv_quad(torch.eye(posterior.lazy_covariance_matrix[dim].size(-1))) # is Tensor with shape (n,) and represents the precision vector (diagonal of precision matrix)
        # for beta_dict["lambda_j"] is the diagonal of the precision matrix
        beta_dict["lambda_j"] = torch.diagonal(torch.linalg.inv(posterior.lazy_covariance_matrix[dim].to_dense())) # is Tensor with shape (n,) and represents the precision vector (diagonal of precision matrix)
        # now we take the first element of the diagonal

        print(f"lambda_j: {beta_dict['lambda_j']}")
        print(f"lambda_j shape: {beta_dict['lambda_j'].shape}")
        #beta_dict["lambda_minus_j"] = torch.tensor(Sigma_minus_j).inv_quad(torch.eye(Sigma_minus_j.size(-1))) # is Tensor with shape (d-1,n)
        #print(f"lambda_minus_j: {beta_dict['lambda_minus_j']}")
        #print(f"lambda_minus_j shape: {beta_dict['lambda_minus_j'].shape}")
        lambda_matrix = torch.inverse(Sigma)
        # Extract Lambda_minus_j for each batch
        Lambda_minus_j = torch.cat([lambda_matrix[:, :dim, :dim], lambda_matrix[:, :dim, dim+1:]], dim=2)
        Lambda_minus_j = torch.cat([Lambda_minus_j, torch.cat([lambda_matrix[:, dim+1:, :dim], lambda_matrix[:, dim+1:, dim+1:]], dim=2)], dim=1)
        Lambda_minus_j = lambda_matrix[:, :dim-1, :dim-1]
        print(f"Lambda_minus_j: {Lambda_minus_j}")
        print(f"Lambda_minus_j shape: {Lambda_minus_j.shape}")
        # Extract lambda_minus_j for each batch
        lambda_minus_j = torch.cat([lambda_matrix[:dim], lambda_matrix[dim+1:]], dim=0)
        print(f"lambda_minus_j: {lambda_minus_j}")
        print(f"lambda_minus_j shape: {lambda_minus_j.shape}")
        posterior_dimwise.append(beta_dict)
    return posterior_dimwise

def get_precision_matrix(posterior):
    precision_matrix = np.linalg.inv(posterior.covariance_matrix)
    return precision_matrix

def calculate_conditional_expectation(sigma_minus_j, sigma_minus_j_j, mu_minus_j, beta_j, mu_j):
    theta_j = np.linalg.inv(sigma_minus_j) @ sigma_minus_j_j
    E_beta_minus_j_given_beta_j = mu_minus_j + theta_j @ (beta_j - mu_j) # E(beta_{-j} | beta_j)
    V_beta_minus_j_given_beta_j = np.linalg.inv(sigma_minus_j) # V(beta_{-j} | beta_j)


def decompose_covariance(cov):
    # convert covariance matrix to numpy
    if not isinstance(cov, np.ndarray):
        cov = cov.numpy()
    # decompose covariance matrix into correlation matrix and standard deviations
    if cov.shape[0] > 1:
        L = np.array()
        for i in range(cov.shape[0]):
            L[i] = np.linalg.cholesky(cov[i])
        return L
    std = np.sqrt(np.diag(cov))
    #corr = cov / np.outer(std, std)
    # apply cholesky decomposition to covariance matrix
    L = np.linalg.cholesky(cov)
    # apply cholesky decomposition to correlation matrix
    #L_corr = np.linalg.cholesky(corr)
    # also return the inverse of the cholesky decomposition
    L_inv = np.linalg.inv(L)
    
    return L


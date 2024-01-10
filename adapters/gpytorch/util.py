import torch
import numpy as np
from scipy.stats import median_abs_deviation, norm

def decompose_matrix(X):
    # if rank(X)) < n_features, then X = U * S * V.T
    X = torch.tensor(X)
    rank_X = torch.linalg.matrix_rank(X)
    U, S, V = torch.linalg.svd(X)
    if rank_X < X.shape[1]:
        print(f"X is not full rank. Rank(X) = {rank_X}")
        return U, S, V
    elif rank_X == X.shape[1] and X.shape[0] > X.shape[1]:
        print(f"X is full rank, but not square. Rank(X) = {rank_X}")
        return U, S, V
    else:
        print(f"X is full rank and square. Rank(X) = {rank_X}")
        return U, S, V
    
def compute_principal_components(X, top_k=2):
    U, S, V = decompose_matrix(X)
    # get top k singular values
    top_k_singular_values = S[:top_k]
    # get top k singular vectors
    top_k_singular_vectors = V[:top_k]
    # compute principal components
    principal_components = top_k_singular_vectors * top_k_singular_values
    return principal_components

def compute_pcs(X, top_k=10):
    """
    from https://github.com/lorinanthony/BAKR/blob/master/Rcpp/BAKRGibbs.cpp
    """
    # check if X is square
    if X.shape[0] != X.shape[1]:
        raise ValueError(f"X is not square. X.shape = {X.shape}")
    U, S, V = torch.linalg.svd()
    PCs = U @ torch.diag(S) # multiply the eigenvalues by the eigenvectors
    return PCs[:, :top_k]
        
def eigenvalue_decomposition(K):
    """
    Decompose a kernel matrix into its eigenvalues and eigenvectors.
    """


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

def get_thetas(K, q):
    U, S, V = decompose_matrix(K)
    #thetamat = U @ np.diag(S)
    thetas = torch.diag(S) @ U.T
    return thetas[:, :q] # return first q columns
    #theta_q = theta[:, q]
    #theta_minus_q = theta[:, q:]
    #return theta_q, theta_minus_q

def get_alphas(K, q):
    thetas = get_thetas(K, q)
    U, S, V = decompose_matrix(K)
    alpha_q = thetas.T @ torch.diag(S)[q:] @ thetas


def get_alphas_(mvn, X, index_map=None, low_rank=False):
    """
    calculate alpha values as (inter)activity rates according to
    Crawford 2019 et al. (https://arxiv.org/pdf/1905.05435.pdf)
    """
    #if isinstance(cov, torch.Tensor):
    #    cov = cov.detach().numpy()
    n, p = X.shape
    alpha_values = np.zeros(p)
    X_pinv = np.linalg.pinv(X)
    #Sigma = X_pinv @ cov @ X_pinv.T
    Sigma = mvn.covariance_matrix.detach().numpy()
    #Lambda = X @ np.linalg.pinv(Sigma) @ X.T
    #Lambda = np.linalg.pinv(Sigma)
    Lambda = mvn.precision_matrix.detach().numpy()
    L = mvn.scale_tril.detach().numpy()
    #L = scipy.linalg.sqrtm(np.asmatrix(Lambda)) # L = Lambda^1/2 = X (Sigma)^-1 X^T = X Lambda^-1 X^T = X Lambda^-1/2 Lambda^-1/2 X^T = X Lambda^-1/2 X^T with dimension p x n
    
    if not index_map:
        for j in range(L.shape[0]):
            Lj = np.delete(L, j, axis=1) # delete j-th column
            Lambda_j_pinv = np.linalg.pinv(Lj @ Lj.T)
            lambda_j = L[:, j] # get j-th column
            lambda_minus_j = np.delete(Lambda[:, j], j, axis=0) # delete j-th element
            Q, R = np.linalg.qr(Lj)
            b = np.linalg.solve(R, Q.T @ lambda_minus_j)  # b = L^+ -j lambda_minus_j
            alpha_j = lambda_minus_j.T @ Lambda_j_pinv @ lambda_minus_j
            alpha_values[j] = alpha_j
        
        return alpha_values
    else:
        raise NotImplementedError("TODO: implement index_map")

def get_beta(B, thetamat):
    """
    translated from https://github.com/lorinanthony/BAKR/blob/master/Rcpp/BAKRGibbs.cpp
    
    """
    betamat = np.zeros((B.shape[0], thetamat.shape[0]))
    for i in range(betamat.shape[1]):
        if i % 100 == 0:
            print(f"i = {i}")
            print(f"B.shape = {B.shape}")
            print(f"thetamat.shape = {thetamat.shape}")
            print(f"betamat[:, i] = {betamat[:, i]}")
            print(f"betamat[:, i].shape = {betamat[:, i].shape}")
            print(f"thetamat[i, :] = {thetamat[i, :]}")
            print(f"thetamat[i, :].shape = {thetamat[i, :].shape}")
        betamat[:, i] = B @ thetamat[i, :].T
    return betamat.T


def get_betas(cov_matrix, Xs, index_map=None):
    X_pinv = torch.pinverse(Xs)
    betas = X_pinv @ cov_matrix
    print(f"betas: {betas}")
    print(f"betas shape: {betas.shape}")
    return betas

def beta_distances(model, X_test, alphas=None, betas=None):
    posterior = model.posterior(X_test)
    mean = model.posterior(X_test).mean
    covariance = model.posterior(X_test).covariance_matrix
    mvn = posterior.mvn
    if not alphas:
        alphas = get_alphas(mvn, X_test)
    if not betas:
        betas = get_betas(covariance, X_test)
    
    # calculate distances between betas
    KLD = np.zeros(betas)
    #for i in range(len(betas)):
    #    for j in range(len(betas)):
    #        KLD[i, j] = scipy.stats.entropy(betas[i], betas[j])
    #KLD = scipy.spatial.distance.pdist(betas, metric="jensenshannon")
    for j in range(len(betas)):
        KLD[j] = alphas[j] * (betas[j] - mean[j])**2 / 2

def LFSR(betamat):
    betamat = torch.tensor(betamat).double()
    M = betamat.shape[0]
    P = betamat.shape[1]
    LFSR = torch.zeros(P).double()

    for i in range(P):
        m = torch.mean(betamat[:, i])
        if m > 0:
            LFSR[i] = torch.sum(betamat[:, i] < 0) / M
        else:
            LFSR[i] = torch.sum(betamat[:, i] > 0) / M
    
    return LFSR

def get_PPAAs(betamat, feature_names, sigval=0.1):
    """
    Posterior Probability of Association (PPAA)
    The idea of PPAA is to calculate the probability that a feature is associated with the response
    by comparing the posterior distribution of the feature with the posterior distribution of the response
    sigval is the desired significance level (e.g. 0.995)
    translated from https://github.com/lorinanthony/BAKR/blob/master/Rcpp/BAKRGibbs.cpp
    """
    betamat = torch.tensor(betamat).double()
    p = betamat.shape[1] # number of features
    PPAA = torch.zeros(betamat.shape[0], p)
    qval = torch.zeros(p)

    for j in range(p):
        zz = j + 1
        qval[j] = sigval * zz / p # bonferroni correction
    
    qval = torch.tensor([sigval * (j+1) / p for j in range(p)])

    for i in range(betamat.shape[0]):
        PIP = torch.zeros(p) # posterior inclusion probability
        pval = torch.zeros(p)
        pvalsort = torch.zeros(p)
        beta = betamat[i, :]
        sigmahat_ = median_abs_deviation(torch.abs(beta))
        sigmahat = median_abs_deviation(torch.abs(beta))/0.6745 # 0.6745 is the median of the standard normal distribution
        #pval = 2 * (1 - torch.distributions.Normal(0, sigmahat).cdf(torch.abs(beta)))
        pval = 2 * (1 - norm.cdf(torch.abs(beta / sigmahat))) # creates a tensor of p-values 
        # sort p-values and preserve feature name mapping
        pvalsort= np.sort(pval)
        feature_idx = np.argsort(pval)
        # dictionary of p-values and corresponding featuresalar
        pvaldict = {feature_idx[j]: pval[j] for j in range(p)}
        if np.any(pvalsort < np.array(qval)): # torch version throws TypError, use numpy instead
            pvalmax = np.max(pvalsort[pvalsort < np.array(qval)])
            t = sigmahat * norm.ppf(1 - pvalmax / 2, 0.0, 1.0) # threshold
            PIP[torch.abs(beta) > t] = 1
            PPAA[i, :] = PIP
    
    return feature_idx, PPAA
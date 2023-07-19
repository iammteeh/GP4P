from sklearn.covariance import EmpiricalCovariance, MinCovDet

def get_covariance_matrix(X, y, method="empirical", **kwargs):
    if method == "empirical":
        cov = EmpiricalCovariance().fit(X)
    elif method == "mcd":
        cov = MinCovDet().fit(X)
    else:
        raise ValueError("method must be either empirical or mcd")
    return cov.covariance_
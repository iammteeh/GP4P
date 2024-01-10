# la place rule
# ungewissheit = indifferenzprinzip (black box)
# unsicherheit = risikoäquivalent (co-varianz-std) <%>

from sklearn.decomposition import PCA, KernelPCA

def kernel_pca(X, feature_names, kernel="poly", n_components=None):
    kpca = KernelPCA(kernel=kernel, n_components=n_components, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X)
    kernel_features = kpca.get_feature_names_out(feature_names)
    inverse_transform = kpca.inverse_transform(X_kpca)
    return X_kpca, kernel_features, inverse_transform
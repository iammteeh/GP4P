from application.basic_pipeline import init_pipeline
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA
import time
import matplotlib.pyplot as plt

def kernel_pca(X, y, kernel="rbf", **kernel_params):
    pca = KernelPCA(n_components=len(X.T), kernel=kernel, n_jobs=8, fit_inverse_transform=True, **kernel_params)
    result = pca.fit_transform(X, y)
    return result

def main():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline()
    # use df of x and y
    X_train = X_train[0]
    X_test = X_test[0]
    y_train = y_train[0]
    y_test = y_test[0]

    modes = ["PCA", "KernelPCA", "SparsePCA"]
    print(f"run {modes}")
    data = {"X": X_train, "y": y_train, "svd_solver": "full", "kernel": "poly", "sparsity_alpha": 0.1}
    print(f" {modes[0]} with {data['svd_solver']}, {modes[1]} with {data['kernel']}, {modes[2]} with {data['sparsity_alpha']}")
    #results = get_pca(*modes, **data)
    pca = KernelPCA(n_components=len(data["X"].columns), kernel=data["kernel"], degree=3, gamma=0.03, fit_inverse_transform=True, random_state=42)
    print(f"start fitting {pca.__class__.__name__}")
    start = time.time()
    result = pca.fit_transform(data["X"], data["y"])
    end = time.time()
    print(f"Finished fitting in {end - start :.2f} seconds")

    # print results of the pca
    #print(f"explained variance ratio: {pca.explained_variance_ratio_}")
    #print(f"sum of explained variance ratio: {sum(pca.explained_variance_ratio_)}")
    #print(f"components: {pca.components_}")
    #print(f"n_features: {pca.n_features_}")
    #print(f"n_samples: {pca.n_samples_}")
    #print(f"n_classes: {pca.n_classes_}")
    #print(f"n_outputs: {pca.n_outputs_}")
    #print(f"kernel_params: {pca.kernel_params_}")
    print(f"features_in_: {pca.n_features_in_}")
    print(f"feature_names_in_: {pca.feature_names_in_}")
    print(f"dual_coef_: {pca.dual_coef_}")
    print(f"eigenvalues: {pca.eigenvalues_}")
    print(f"eigenvectors: {pca.eigenvectors_}")
    #print(f"einvalue_ratio: {pca.eigenvalues_ratio_}")


    # plot results
    plt.scatter(result[:, 0], result[:, 1], c=data["y"], cmap="plasma")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()

    _, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

    train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    train_ax.set_ylabel("Feature #1")
    train_ax.set_xlabel("Feature #0")
    train_ax.set_title("Training data")

    test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    test_ax.set_xlabel("Feature #0")
    _ = test_ax.set_title("Testing data")

if __name__ == "__main__":
    main()
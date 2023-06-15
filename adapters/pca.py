from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA
import time

def get_pca(*modes, **data):
    pcas = []
    results = []
    for mode in modes:
        if mode == "PCA":
            pcas.append(PCA(n_components=len(data["X"].columns), whiten=True, svd_solver=data["svd_solver"], random_state=42, ))
        elif mode == "KernelPCA":
            pcas.append(KernelPCA(n_components=len(data["X"].columns), kernel=data["kernel"], degree=3, gamma=0.03, fit_inverse_transform=True, random_state=42))
        elif mode == "SparsePCA":
            pcas.append(SparsePCA(n_components=len(data["X"].columns), alpha=data["sparsity_alpha"], random_state=42))
        else:
            raise ValueError("PCA mode not recognized")
        
    for pca in pcas:
        start = time.time()
        print(f"fitting {pca.__class__.__name__}")
        result = pca.fit(data["X"], data["y"])
        end = time.time()
        print(f"Finished fitting in {end - start :.2f} seconds")
        #data["X"] = pca.transform(data["X"])
        results.append(result)

    return results
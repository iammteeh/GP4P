from application.init_pipeline import init_pipeline, get_numpy_features
from domain.env import USE_DUMMY_DATA, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE
from domain.feature_model.feature_modeling import additive_kernel_permutation

import GPy as gp

def test_additive_kernel_generation():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    # use ndarrays of X and y
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)
    items = [gp.kern.Linear(input_dim=X_train.shape[1]) for item in range(5)]
    basis_kernel = gp.kern.Kern(input_dim=X_train.shape[1], active_dims=None, name='basis_kernel')
    additive_kernels = additive_kernel_permutation(basis_kernel, items, k=3)

if __name__ == "__main__":
    print(test_additive_kernel_generation())
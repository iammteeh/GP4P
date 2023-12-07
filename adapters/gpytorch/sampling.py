from contextlib import ExitStack
import gpytorch.settings as gpts
from gpytorch.variational import VariationalStrategy, NNVariationalStrategy
from botorch.generation import MaxPosteriorSampling, BoltzmannSampling
import torch
from torch.quasirandom import SobolEngine

def draw_random_samples(model, num_samples=1000):
    # draw samples from posterior
    posterior = model.posterior(model.X)
    samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
    samples = {
        "mean": samples[:, :, 0],
        "variance": samples[:, :, 1],
        "lengthscale": samples[:, :, 2],
        "outputscale": samples[:, :, 3],
    }
    return samples

def draw_random_x(X_test, num_samples=100):
    # draw random n x p samples from X_test
    X_test = X_test.repeat(num_samples, 1, 1)
    X_test = X_test + torch.randn_like(X_test)
    return X_test

def generate_test_x_from_tensor(tensor):
    import itertools
    num_features = tensor.shape[1]
    all_combinations = []

    # Generate all possible combinations of features
    for r in range(1, num_features + 1):
        for indices in itertools.combinations(range(num_features), r):
            combination = torch.zeros(num_features)
            combination[list(indices)] = 1
            all_combinations.append(combination)

    return torch.stack(all_combinations)

def get_initial_points(dim, n_pts, seed=None):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts)
    return X_init

def generate_batch(
    model,
    X,
    batch_size,
    n_candidates,
    sampler="cholesky",  # "cholesky", "ciq", "rff"
    use_keops=False,
):
    assert sampler in ("cholesky", "ciq", "rff", "lanczos")

    # Draw samples on a Sobol sequence
    sobol = SobolEngine(X.shape[-1], scramble=True)
    X_cand = sobol.draw(n_candidates).to(X)

    # Thompson sample
    with ExitStack() as es:
        if sampler == "cholesky":
            es.enter_context(gpts.max_cholesky_size(float("inf")))
        elif sampler == "ciq":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(
                gpts.minres_tolerance(2e-3)
            )  # Controls accuracy and runtime
            es.enter_context(gpts.num_contour_quadrature(15))
        elif sampler == "lanczos":
            es.enter_context(
                gpts.fast_computations(
                    covar_root_decomposition=True, log_prob=True, solves=True
                )
            )
            es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(False))
        elif sampler == "rff":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

    with torch.no_grad():
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

from gpytorch.variational import VariationalStrategy, NNVariationalStrategy
import torch

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
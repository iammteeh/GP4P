import gpytorch
import torch
import pyro
from jax import numpy as jnp
from jax import vmap
from adapters.gpytorch.gp_model import GPRegressionModel, SAASGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from application.init_pipeline import init_pipeline, get_numpy_features
from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
from domain.feature_model.feature_modeling import get_feature_model
import numpy as np
import datetime
from time import time
from matplotlib import pyplot as plt
from scipy.stats import norm


from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

def locate_invalid_data(data):
    if isinstance(data, torch.Tensor):
        isnan = torch.isnan(data)
        isinf = torch.isinf(data)
    elif isinstance(data, np.ndarray):
        isnan = np.isnan(data)
        isinf = np.isinf(data)
    
    invalid_data_locs = isnan | isinf
    return invalid_data_locs

def validate_data(*args):
    for arg in args:
        if not torch.isfinite(arg).all():
            print(locate_invalid_data(arg))
            raise ValueError("Data contains NaN or inf values.")
    print(f"data is fine.")

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

def visualize_mixture(model, X_test, y_test):
    """
    
    """

def analyze_model(model, active_dims):
    p, n, m = model.X.shape[1], model.X.shape[0], len(active_dims)
    num_coefficients = p + m * (m - 1) // 2

    probe = jnp.zeros((2 * p + 2 * m * (m - 1)), p)
    vec = jnp.zeros((num_coefficients, 2 * p + 2 * m * (m - 1)))

def get_data():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)
    rank = np.linalg.matrix_rank(X_train)

    # transform test data to tensor
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return (X_train, X_test, y_train, y_test, feature_names)

def choose_model(model="GP", data=None):
    if not data:
        X_train, X_test, y_train, y_test, feature_names = get_data()
    else:
        X_train, X_test, y_train, y_test, feature_names = data
    
    if model == "GP":
        return GPRegressionModel(X_train, y_train, feature_names, likelihood="gaussian", kernel=KERNEL_TYPE, mean_func=MEAN_FUNC, structure=KERNEL_STRUCTURE)
    elif model == "SAASGP":
        return SAASGP(X_train, y_train, feature_names)
    else:
        raise ValueError(f"Model {model} not found.")
    
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

def plot_prior(model, X_test, y_test):
    with torch.no_grad():
        model.eval()
        prior = model(X_test).sample(sample_shape=torch.Size([1]))
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(X_test.T.numpy(), prior.numpy(), lw=0.8)
        #plt.figure(figsize=(4, 3))
        #for i in range(samples.size(0)):
        #    plt.plot(X_test.numpy(), samples[i].numpy(), lw=0.8)

        #plt.xlabel("x")
        #plt.ylabel("y")
        #plt.title("Samples from GP prior")
        #plt.show()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-3, 3)

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
    print(f"{data[:-1]}") # all but last
    print(f"{data[-1:]}") # last
    # for shape (n, d)
    print(f"shape of data: {data.shape[0]} rows, {data.shape[1]} columns")
    # slice of data
    print(f"slice of data: {data[0:5, 0]}") # first 5 rows of first column
    # convert tensor to numpy
    print(f"numpy: {data.numpy()}")

def plot_pairwise_posterior_mean_variances(d1, d2, posterior, X_test, y_test):
    # shape of posterior mean and variance: (d, n, y)
    # shape of posterior mixture mean and variance: (n, y)
    mean_d_1 = posterior.mean[d1, :, :].numpy()
    variance_d_1 = posterior.variance[d1, :, :].numpy()
    mean_d_2 = posterior.mean[d2, :, :].numpy()
    variance_d_2 = posterior.variance[d2, :, :].numpy()
    sample_range = torch.arange(0, posterior.mean.shape[0])
    # plot posterior for dimensions d1 and d2
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_d_1, mean_d_2, alpha=0.5)
    plt.xlabel(f"mean of dimension {d1}")
    plt.ylabel(f"mean of dimension {d2}")
    plt.title(f"Posterior mean of dimensions {d1} and {d2}")
    plt.grid(True)
    # safe figure and show
    #plt.savefig(f"{RESULTS_DIR}/mean_d{d1}_d{d2}_{TIMESTAMP}.png", dpi=300, bbox_inches="tight")
    #plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(variance_d_1, variance_d_2, alpha=0.5)
    plt.xlabel(f"variance of dimension {d1}")
    plt.ylabel(f"variance of dimension {d2}")
    plt.title(f"Posterior variance of dimensions {d1} and {d2}")
    plt.grid(True)
    # safe figure and show
    #plt.savefig(f"{RESULTS_DIR}/variance_d{d1}_d{d2}_{TIMESTAMP}.png", dpi=300, bbox_inches="tight")
    #plt.show()
    mean_d_1 = np.mean(mean_d_1, axis=0)
    variance_d_1 = np.mean(variance_d_1, axis=0)
    mean_d_2 = np.mean(mean_d_2, axis=0)
    variance_d_2 = np.mean(variance_d_2, axis=0)
    print(f"shape of mean_d_1: {mean_d_1.shape}")
    return mean_d_1, variance_d_1, mean_d_2, variance_d_2

def plot_density(mean, variance, d):
    x = np.linspace(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance), 100)
    pdf = norm.pdf(x, mean, np.sqrt(variance))
    plt.plot(x, pdf, linewidth=2, color='k')
    #plt.fill_between(x, pdf, 0, alpha=0.2, color='k-')
    plt.title(f"Normal distribution of dimension {d}")
    plt.xlabel(f"mean: {mean}, variance: {variance}")
    plt.ylabel("density")
    plt.grid(True)
    # safe figure and show
    #plt.savefig(f"{RESULTS_DIR}/density_d{d}_{TIMESTAMP}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_combined_pdf(features):
    x_values = np.linspace(-10, 10, 100)
    for feature, params in features.items():
        #x = np.linspace(means - 3*np.sqrt(variances), means + 3*np.sqrt(variances), 100)
        pdf = norm.pdf(x_values, params["mean"], params["std"])
        #scaled_pdf = pdf / np.sum(pdf)
        scaled_pdf = pdf / np.max(pdf) * params["std"]
        plt.plot(x_values, scaled_pdf, label=f"{feature} (scaled by std)")

    plt.title('Compare feature variances')
    plt.xlabel('x')
    plt.ylabel('density')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combined_pdf_v2(features):
    for feature, param in features.items():
        x = np.linspace(param["mean"] - 3*param["std"], param["mean"] + 3*param["std"], 100)
        pdf = norm.pdf(x, param["mean"], param["std"])
        plt.plot(x, pdf, label=f"{feature} (scaled by std)")

    plt.title('Compare feature variances')
    plt.xlabel('x')
    plt.ylabel('density')
    plt.legend()
    plt.grid(True)
    # safe figure and show
    #plt.savefig(f"{RESULTS_DIR}/density_combined_{features.keys()}_{TIMESTAMP}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_interaction_pdfs(param_features):
    for i, (mean, variance) in enumerate(param_features):
        x = np.linspace(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance), 100)
        pdf = norm.pdf(x, mean, np.sqrt(variance))
        plt.plot(x, pdf, label=f"{i}:{param_features[i]} (scaled by std)")

    plt.title('Compare feature variances')
    plt.xlabel('x')
    plt.ylabel('density')
    plt.legend()
    plt.grid(True)
    # safe figure and show
    plt.savefig(f"{RESULTS_DIR}/kernel_interactions_{param_features}.png", dpi=300, bbox_inches="tight")
    plt.show()

def analyze_posterior(model, X_test, y_test, features):
    words, opposites, literals, interactions = get_feature_model(X_test, features)
    print(f"words: {words}")
    print(f"interactions: {interactions}")

    word_posterior = model.posterior(words)
    mean_words, variance_words = word_posterior.mixture_mean.numpy(), word_posterior.mixture_variance.numpy()
    mean_words = np.mean(mean_words, axis=0)
    variance_words = np.median(variance_words, axis=0)

    opposite_posterior = model.posterior(opposites)
    mean_opposites, variance_opposites = opposite_posterior.mixture_mean.numpy(), opposite_posterior.mixture_variance.numpy()
    mean_opposites = np.mean(mean_opposites, axis=0)
    variance_opposites = np.median(variance_opposites, axis=0)

    literals_posterior = model.posterior(literals)
    mean_literals, variance_literals = literals_posterior.mixture_mean.numpy() , literals_posterior.mixture_variance.numpy()
    mean_literals = np.mean(mean_literals, axis=0)
    variance_literals = np.median(variance_literals, axis=0)

    interactions_posterior = model.posterior(interactions)
    mean_interactions, variance_interactions = interactions_posterior.mixture_mean.numpy(), interactions_posterior.mixture_variance.numpy()
    mean_interactions = np.mean(mean_interactions, axis=0)
    variance_interactions = np.median(variance_interactions, axis=0)

    return (mean_opposites, variance_opposites), (mean_interactions, variance_interactions)

def main():
    GP = "SAASGP"
    # init model
    data = get_data()
    X_train, X_test, y_train, y_test, feature_names = data
    model = choose_model(model=GP, data=data)

    # check for NaN / inf
    validate_data(model.X, X_test, model.y, y_test)

    #plot_prior(model, X_test, y_test)

    # fit
    model.train()
    fit_fully_bayesian_model_nuts(model)

    # Evaluate model
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        observed_pred = model.likelihood(model(X_test))
        mean = observed_pred.mean
        var = observed_pred.variance
        posterior = model.posterior(X_test)
        print(f"shape of posterior mean: {posterior.mean.shape}")  # shape of posterior mean: torch.Size([16, 80, 1]) = (num_samples, num_test_points, num_outputs) = d x n x m
        print(f"shape of posterior variance: {posterior.variance.shape}") # shape of posterior variance: torch.Size([16, 80, 1]) = (num_samples, num_test_points, num_outputs) = d x n x m
        print(f"shape of posterior mixture mean: {posterior.mixture_mean.shape}") 
        print(f"shape of posterior mixture variance: {posterior.mixture_variance.shape}")
        # shape of posterior mean: torch.Size([16, 80, 1]) = (num_samples, num_test_points, num_outputs) = d x n x m
        print(f"kernel lengthscales: {model.covar_module.base_kernel.lengthscale}")
        print(f"kernel outputscale: {model.covar_module.outputscale}")
        print(f"Ground truth:     {y_test.squeeze(-1)}")
        print(f"Mixture mean:     {posterior.mean.squeeze(-1)}")
        print(f"Mixture variance: {posterior.variance.squeeze(-1)}")
        # create feature list
        features = [[(10,1), (15,1)],
                    [(10,1), (15,0)],
                    [(10,0), (15,1)],
                    [(10,0), (15,0)],
                    [(3,1), (4,1)],
                    [(3,1), (4,0)],
                    [(3,0), (4,1)],
                    [(3,0), (4,0)],
                    [(3,1), (4,1), (5,1)],
                    [(3,1), (4,1), (5,0)],
                    [(3,1), (4,0), (5,1)],
                    [(3,1), (4,0), (5,0)],
                    [(3,0), (4,1), (5,1)],
                    [(3,0), (4,1), (5,0)],
                    [(3,0), (4,0), (5,1)],
                    [(3,0), (4,0), (5,0)],
                    [(5,1), (6,1), (7,1)],
                    [(5,1), (6,1), (7,0)],
                    [(5,1), (6,0), (7,1)],
                    [(5,1), (6,0), (7,0)],
                    [(5,0), (6,1), (7,1)],
                    [(5,0), (6,1), (7,0)],
                    [(5,0), (6,0), (7,1)],
                    [(5,0), (6,0), (7,0)],
                    [(5,1), (6,1), (7,1), (8,1)],
                    [(5,1), (6,1), (7,1), (8,0)],
                    [(5,1), (6,1), (7,0), (8,1)],
                    [(5,1), (6,1), (7,0), (8,0)],
                    [(5,1), (6,0), (7,1), (8,1)],
                    [(5,1), (6,0), (7,1), (8,0)],
                    [(5,1), (6,0), (7,0), (8,1)],
                    [(5,1), (6,0), (7,0), (8,0)],
                    [(5,0), (6,1), (7,1), (8,1)],
                    [(5,0), (6,1), (7,1), (8,0)],
                    [(5,0), (6,1), (7,0), (8,1)],
                    [(5,0), (6,1), (7,0), (8,0)],
                    [(5,0), (6,0), (7,1), (8,1)],
                    [(5,0), (6,0), (7,1), (8,0)],
                    [(5,0), (6,0), (7,0), (8,1)],
                    [(5,0), (6,0), (7,0), (8,0)],
                    [(5,1), (6,1), (7,1), (8,1), (9,1)],
                    [(5,1), (6,1), (7,1), (8,1), (9,0)],
        ]
        for feature in features:
            try:
                opposites, interactions = analyze_posterior(model, X_test, y_test, feature)
                plot_interaction_pdfs([opposites, interactions])
            except:
                print("no opposites or interactions found")
        # extend dim_pairs to include all dimensions
        # create dim_pairs for 5 dimensions
        dim_pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), 
            (1, 2), (1, 3), (1, 4), (1, 5), 
            (2, 3), (2, 4), (2, 5), 
            (3, 4), (3, 5), 
            (4, 5),
        ]
        #poly_dim_pairs = [(0,1), (0,2), (1,2)]
        features = {}
        for dim_pair in dim_pairs:
            features = {}
            mean_d_1, variance_d_1, mean_d_2, variance_d_2 = plot_pairwise_posterior_mean_variances(dim_pair[0], dim_pair[1], posterior, X_test, y_test)
            plot_density(mean_d_1, variance_d_1, dim_pair[0])
            plot_density(mean_d_2, variance_d_2, dim_pair[1])
            ## create feature dictionary
            features[feature_names[dim_pair[0]]] = {"mean": mean_d_1, "std": np.sqrt(variance_d_1)} # features["feature_name"] = {"mean": mean, "std": std}
            features[feature_names[dim_pair[1]]] = {"mean": mean_d_2, "std": np.sqrt(variance_d_2)}
            plot_combined_pdf_v2(features)
        #plot_combined_pdf_v2(features)
        # get mixture mean and variance of dimensions d_1, d_2

        #preds = model(X_test)
        #mean = preds.mean
        #variance = preds.variance
        #print(f"preds: {preds.mean}")
        #print(f"preds: {preds.variance}")
        #preds = model.likelihood(model(X_test))

    #print(waic(model, model.likelihood, X_test, y_test))

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f"{MODELDIR}/GPY_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_ARD={ARD}__{timestamp}.pth")

    # Number of lines to plot (n samples)
    n_lines = 5
    offset = 10  # Vertical offset between lines

    vertical_margin = 20
    horizontal_margin = 100
    x_size = 28
    y_size = 25
    linewidth = 4

    fig, ax = plt.subplots(figsize=(x_size,y_size))
    plt.style.use('dark_background')

    for i in range(n_lines):
        # Sample from the posterior
        #sample = mean + variance.sqrt() * torch.randn_like(mean)
        sample = posterior.rsample()
        
        # Create each line plot
        #plt.plot(X_test.T.numpy(), sample.numpy() + i * offset, color='white', lw=0.8)
        #lt.plot(X_test.T.numpy(), sample.numpy() + i * offset, color='white', lw=linewidth)
        #ax.fill_between(x, -5,line,  facecolor='black', zorder=row[0]/n_lines)

    
    #ax.set_yticks([])
    #ax.set_xticks([])
    #ax.set_xlim(min(x)-horizontal_margin, max(x)+horizontal_margin)
    #ax.set_ylim(-vertical_margin, df.shape[0] + vertical_margin)

if __name__ == "__main__":
    main()

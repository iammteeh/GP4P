import gpytorch
import torch
import pyro
from adapters.gpytorch.gp_model import GPRegressionModel, SAASGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from application.init_pipeline import init_pipeline, get_numpy_features
from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from domain.env import USE_DUMMY_DATA, MODELDIR, EXTRAFUNCTIONAL_FEATURES, POLY_DEGREE, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE, ARD, RESULTS_DIR
from adapters.gpytorch.sampling import draw_random_samples, draw_random_x, generate_test_x_from_tensor
from adapters.gpytorch.evaluation import analyze_posterior, explore_data, analyze_model, analyze_mixture
from adapters.gpytorch.plotting import plot_prior, plot_pairwise_posterior_mean_variances, plot_density, plot_combined_pdf_v2, plot_interaction_pdfs
import numpy as np
import datetime
from time import time
from matplotlib import pyplot as plt


from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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


def get_data():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)
    rank = np.linalg.matrix_rank(X_train)

    # transform test data to tensor
    X_test = torch.tensor(X_test).double()
    y_test = torch.tensor(y_test).double()

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

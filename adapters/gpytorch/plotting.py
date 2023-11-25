import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from domain.env import RESULTS_DIR

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
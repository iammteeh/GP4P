import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import jax.random as random
from domain.env import SAVE_FIGURES, RESULTS_DIR
import time

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = beta_values.shape[0]
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()

def generate_heatmap(X_input):
    corrmat = X_input.corr(method='spearman')
    top_corr_features = corrmat.index


    # Visualise a lower-triangle correlation heatmap
    mask_df = np.triu(np.ones(corrmat.shape)).astype(np.bool)
    plt.figure(figsize=(10,8))
    #plot heat map
    g=sns.heatmap(X_input[top_corr_features].corr(), 
                  mask = mask_df, 
                  vmin = -1,
                  vmax = 1,
                  annot=True,
                  cmap="RdBu")
    

def plot_lassoCV(model, regression):
    m_log_alphas = -np.log10(regression.method.alphas_)
    plt.figure(figsize=(10, 8))
    plt.plot(m_log_alphas, regression.method.mse_path_, ':')
    plt.plot(m_log_alphas, regression.method.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(regression.method.alpha_), linestyle='--', color='k',
                label='alpha CV')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold')
    plt.axis('tight')
    plt.show()

def plot_coefs(coefs):
    flattened_coefs = coefs.flatten()

    # create a histogram of the coefficients
    plt.hist(flattened_coefs, bins=50)
    plt.title('Distribution of the coefficients')
    plt.xlabel('Coefficient value')
    plt.ylabel('Frequency')
    plt.show()

    # display a boxplot of the coefficients
    sns.boxplot(flattened_coefs)
    plt.title('Boxplot of the coefficients')
    plt.xlabel('Coefficient value')
    plt.show()

def scatter_plot(y_pred, y_test):
    plt.scatter(y_pred, y_test, alpha=0.2)
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()

def plot_feature_importance(feature_names, coef):
    feature_importance = np.mean(np.abs(coef), axis=0)
    # scale by max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # sort features by importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def plot_dist(dist_object, title, n_samples=1000):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    seed = random.PRNGKey(123456789)
    samples = dist_object.sample(seed, (n_samples,))
    if dist_object == isinstance(dist_object, dist.MultivariateNormal):
        sns.jointplot(samples[:, 0], samples[:, 1], kind="kde")
        plt.title(title)
        plt.savefig(f"{RESULTS_DIR}/{title}_{timestamp}.png") if SAVE_FIGURES else plt.show()
        #sns.distplot(samples[:, 0], bins=30, kde=True)
        #sns.distplot(samples[:, 1], bins=30, kde=True)
        #plt.title(title)
        #plt.show()
        #sns.kdeplot(samples[:, 0], samples[:, 1], shade=True)
        #plt.title(title)
        #plt.show()
    else:
        #sns.distplot(dist_object, hist=False, rug=True)
        plt.hist(samples, bins=50, density=True)
        plt.title(title)
        plt.savefig(f"{RESULTS_DIR}/{title}_{timestamp}.png") if SAVE_FIGURES else plt.show()

def plot_gp_feature(feature_idx, sample_idx, X_train, X_test, mean_pred, std_pred, y_train):
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(X_test[:, feature_idx], mean_pred[:, sample_idx], 'r', lw=2, label="mean prediction")
    plt.fill_between(X_test[:, feature_idx], mean_pred[:, sample_idx] - std_pred[:, sample_idx], mean_pred[:, sample_idx] + std_pred[:, sample_idx], color='r', alpha=0.5, label="1 std deviation")
    plt.plot(X_train, y_train, 'ok', ms=3, alpha=0.5, label="Training points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def ppc(pred_samples, Y, ppc_var='y_obs'):
    plt.figure(figsize=(10,5))

    # PPC samples
    ppc_samples = pred_samples[ppc_var].flatten()

    # Plot the histogram of PPC samples
    sns.histplot(ppc_samples, color='skyblue', kde=True, label='Predicted data');

    # Plot the histogram of observed data
    sns.histplot(Y, color='red', kde=True, label='Observed data');

    plt.legend();
    plt.show();

def plot(title="title"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"{RESULTS_DIR}/{title}_{timestamp}.png") if SAVE_FIGURES else plt.show()
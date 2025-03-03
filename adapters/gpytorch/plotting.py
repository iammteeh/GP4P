import torch
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
import seaborn as sns
import numpy as np
import math
from scipy.stats import norm, mode, multivariate_normal
from streamlit import pyplot
from domain.env import RESULTS_DIR, DATA_SLICE_AMOUNT
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

def plot_2d_mvn(dim1, dim2, mvn, mode="show"):
    import pandas as pd
    mean, cov = [0., 0.], [(1., -0.6), (-0.6, 1.)]
    mean1, mean2 = dim1["mean"].mean().item(), dim2["mean"].mean().item()
    var1, var2 = dim1["variance"].median().item(), dim2["variance"].median().item()
    feature_name1, feature_name2 = dim1["feature_name"], dim2["feature_name"]
    # concatenate the means and covariances with numpy
    mean = np.array([mean1, mean2]).flatten()
    cov = np.array([[var1, 0], [0, var2]])
    print(f"shape of mean: {mean.shape}, shape of cov: {cov.shape}")
    data = np.random.multivariate_normal(mean, cov, 1000)
    df = pd.DataFrame(data, columns=[feature_name1, feature_name2])
    g = sns.jointplot(data=df, x=feature_name1, y=feature_name2, kind="kde", color="m")

    #(sns.jointplot("x1", "x2", data=df).plot_joint(sns.kdeplot))

    g.plot_joint(plt.scatter, c="g", s=30, linewidth=1, marker="+")

    #g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("$x1$", "$x2$");

    #g.ax_joint.legend_.remove()

    # safe figure and show
    if mode == "show":
        plt.show() 
    elif mode == "dashboard":
        pyplot(plt, clear_figure=True)
    else:
        raise NotImplementedError

def plot_3d_mvn(dim1, dim2, mvn):

    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mean1, mean2 = dim1["mean"].mean().item(), dim2["mean"].mean().item()
    var1, var2 = dim1["variance"].median().item(), dim2["variance"].median().item()
    feature_name1, feature_name2 = dim1["feature_name"], dim2["feature_name"]
    # concatenate the means and covariances with numpy
    mu = np.array([mean1, mean2]).flatten()
    Sigma = np.array([[var1, 0], [0, var2]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.2, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.2,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(30, -100)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$P(x_1, x_2)$')

    plt.title('mean, cov = [0., 1.], [(1., 0.8), (0.8, 1.)]')
    #plt.savefig('2d_gaussian3D_0.8.png', dpi=600)
    plt.show()

def plot_cov_insights(dimensional_model, num_dims=[0,1], mode="show"):
    # Assuming dimensional_model is defined with "mean" and "covariance" for each dimension

    # Number of dimensions in your model
    #num_dims = len(dimensional_model)

    # Prepare the figure with a grid of subplots
    fig, axs = plt.subplots(len(num_dims), 3, figsize=(15, 5 * len(num_dims)))

    # Iterate over each dimension pair
    for i in num_dims:
        # Mean and covariance for the current dimension
        mean_i = dimensional_model[i]["mean"]
        cov_i = dimensional_model[i]["covariance"]

        # Contour plot for the mean of two dimensions
        if i < len(dimensional_model) - 1:  # Ensure there is a next dimension to pair with
            mean_j = dimensional_model[i + 1]["mean"].squeeze().numpy()
            
            # Choose appropriate grid limits
            grid_x_min, grid_x_max = mean_i.min() - 3 * np.sqrt(cov_i[i, i]), mean_i.max() + 3 * np.sqrt(cov_i[i, i])
            grid_y_min, grid_y_max = mean_j.min() - 3 * np.sqrt(cov_i[i + 1, i + 1]), mean_j.max() + 3 * np.sqrt(cov_i[i + 1, i + 1])
            
            xi, yi = np.meshgrid(np.linspace(grid_x_min, grid_x_max, 100), np.linspace(grid_y_min, grid_y_max, 100))
            
            # Extract the covariance entries for just these two dimensions
            cov_ij = np.array([
                [cov_i[i, i], cov_i[i, i + 1]],
                [cov_i[i + 1, i], cov_i[i + 1, i + 1]]
            ])
            
            # Create a bivariate normal distribution
            rv_ij = multivariate_normal([mean_i[i], mean_j[i + 1]], cov_ij)
            
            # Compute the PDF over the grid and plot the contour
            zi = rv_ij.pdf(np.dstack((xi, yi)))
            axs[i, 0].contourf(xi, yi, zi, levels=100)
            axs[i, 0].set_title(f'Contour Plot of Mean (Dims {i} & {i+1})')


        # Line plot for the mean and confidence interval of the current dimension
        std_dev_i = np.sqrt(np.diag(cov_i))
        axs[i, 1].plot(mean_i, 'k-', lw=2)
        axs[i, 1].fill_between(range(len(mean_i)), mean_i - 1.96 * std_dev_i, mean_i + 1.96 * std_dev_i, alpha=0.2)
        axs[i, 1].set_title(f'Mean and Confidence (Dim {i})')

        # Heatmap of the covariance matrix for the current dimension
        im = axs[i, 2].imshow(cov_i, cmap='plasma', interpolation='nearest')
        fig.colorbar(im, ax=axs[i, 2])
        axs[i, 2].set_title(f'Heatmap of Covariance Matrix (Dim {i})')

    if mode == "show":
        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()
    elif mode == "dashboard":
        pyplot(plt, clear_figure=True)
    else:
        raise NotImplementedError

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
    variance_d_1 = np.median(variance_d_1, axis=0)
    mean_d_2 = np.mean(mean_d_2, axis=0)
    variance_d_2 = np.median(variance_d_2, axis=0)
    print(f"shape of mean_d_1: {mean_d_1.shape}")
    return mean_d_1, variance_d_1, mean_d_2, variance_d_2

def plot_density(mean, variance, feature_name):
    x = np.linspace(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance), DATA_SLICE_AMOUNT)
    pdf = norm.pdf(x, mean, np.sqrt(variance))
    plt.plot(x, pdf, linewidth=2, color='k')
    #plt.fill_between(x, pdf, 0, alpha=0.2, color='k-')
    plt.title(f"Normal distribution of feature \"{feature_name}\"")
    plt.xlabel(f"mean: {mean}, variance: {variance}")
    plt.ylabel("density")
    plt.grid(True)
    # safe figure and show
    if mode == "show":
        plt.show() 
    elif mode == "dashboard":
        pyplot(plt, clear_figure=True)
    else:
        plt.savefig(f"{RESULTS_DIR}/density_{feature_name}_{TIMESTAMP}.png", dpi=300, bbox_inches="tight")

def plot_combined_pdf(features, mode="show"):
    for feature, param in features.items():
        if type(param["mean"]) is not np.ndarray or type(param["std"]) is not np.ndarray:
            param["mean"] = param["mean"].numpy()
            param["std"] = param["std"].numpy()
        mean = np.mean(param["mean"])
        std = np.median(param["std"])
        x = np.linspace(mean - 3*std, mean + 3*std, DATA_SLICE_AMOUNT)
        pdf = norm.pdf(x, mean, std)
        plt.plot(x, pdf, label=f"{param['feature_name']} (scaled by std)")

    plt.title('Compare feature variances')
    plt.xlabel('µ')
    plt.ylabel('density')
    plt.legend()
    plt.grid(True)
    # safe figure and show
    if mode == "show":
        plt.show() 
    elif mode == "dashboard":
        pyplot(plt, clear_figure=True)
    else:
        plt.savefig(f"{RESULTS_DIR}/density_combined_{features.keys()}_{TIMESTAMP}.png", dpi=300, bbox_inches="tight")

def plot_interaction_pdfs(param_features, selected_features, mode="show"):
    for i, (feature, param) in enumerate(param_features.items()):
        if type(param["mean"]) is not np.ndarray or type(param["std"]) is not np.ndarray:
            param["mean"] = param["mean"].numpy()
            param["std"] = param["std"].numpy()
        
        mean = np.mean(param["mean"])
        std = np.median(param["std"])
        x = np.linspace(mean - 3*std, mean + 3*std, DATA_SLICE_AMOUNT)
        pdf = norm.pdf(x, mean, std)
        plt.plot(x, pdf, label=f"{i}:{param['feature_name']} (scaled by std)")

    plt.title('Compare feature interaction of {}'.format(selected_features))
    plt.xlabel('µ')
    plt.ylabel('density')
    plt.legend()
    plt.grid(True)
    # safe figure and show
    if mode == "show":
        plt.show() 
    elif mode == "dashboard":
        pyplot(plt, clear_figure=True)
    else:
        plt.savefig(f"{RESULTS_DIR}/kernel_interactions_{param_features}.png", dpi=300, bbox_inches="tight")

def mean_and_confidence_region(X_test, X_train, y_train, mean, lower, upper, mode="show"):
    print(f"X_test: {X_test.shape}, X_train: {X_train.shape}, y_train; {y_train.shape}, mean: {mean.shape}, lower: {lower.shape}, upper: {upper.shape}")
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train)
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train)
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper)
    
    with torch.no_grad():
        # Ensure one-dimensional data for plotting
        X_test = X_test.flatten().detach().numpy()
        X_test = np.mean(X_test, axis=0)
        print(f"mean X_test: {X_test}")
        X_test = np.linspace(0, 100, DATA_SLICE_AMOUNT)
        print(f"linspace X_test: {X_test}")
        X_train = X_train.flatten().detach().numpy()
        X_train = np.mean(X_train, axis=0)
        X_train = np.linspace(0, 100, DATA_SLICE_AMOUNT)
        print(f"flattened X_train: {X_train.shape}")
        y_train = y_train.flatten()
        ##y_train = y_train.flatten()
        #mean = mean.flatten(start_dim=0)
        #lower = lower.flatten(start_dim=0)
        #upper = upper.flatten(start_dim=0)
        # print shapes
        print(f"X_test: {X_test.shape}, X_train: {X_train.shape}, mean: {mean.shape}, lower: {lower.shape}, upper: {upper.shape}")
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(X_train, y_train, 'k*')
        ax.plot(X_test, mean, 'b')
        #ax.scatter(X_test.numpy(), mean.numpy(), c='b', s=10)
        ax.fill_between(X_test, lower, upper, alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        if mode == "show":
            plt.show() 
        elif mode == "dashboard":
            pyplot(plt, clear_figure=True)
        else:
            plt.savefig("kde_plots.png")

def grid_plot(X_train, y_train, X_test, mean, confidence_region, mode="show"):
    lower, upper = confidence_region

    num_dimensions = X_train.shape[1] 
    grid_rows = 4
    grid_cols = 4
    # Create a larger figure to give each subplot more room
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(20, 20))  # Adjust the figure size as needed

    # Iterate over each dimension
    for i in range(num_dimensions):
        row = i // grid_cols
        col = i % grid_cols
        ax = axs[row, col]

        # Plotting for the ith dimension
        X_train_feature = X_train[:, i]
        X_test_feature = X_test[:, i]
        mean_feature = mean[i, :]  # Select the ith row for mean
        lower_feature = lower[i, :]  # Select the ith row for lower
        upper_feature = upper[i, :]  # Select the ith row for upper

        ax.plot(X_train_feature, y_train, 'k*')
        ax.plot(X_test_feature, mean_feature, 'b')

        # Improve readability
        ax.set_xlim([min(X_test_feature), max(X_test_feature)])  # Set x-axis limits
        ax.set_ylim([min(lower_feature), max(upper_feature)])    # Set y-axis limits

        # Increase transparency of the confidence intervals
        ax.fill_between(X_test_feature, lower_feature, upper_feature, alpha=0.2)

        # Adjust line thickness
        ax.plot(X_test_feature, mean_feature, 'b', linewidth=2)

        # Adjust subplot spacing
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the spacing between subplots

    # Add a more descriptive title and adjust the top spacing
    fig.suptitle('Gaussian Process Regression for Each Feature', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout

    if mode == "show":
        plt.show() 
    elif mode == "dashboard":
        pyplot(plt, clear_figure=True)
    else:
        plt.savefig("kde_plots.png")

def kde_plots(X_train, y_train, mode="show"):

    # Example synthetic data for demonstration
    # Assume X_train is a torch tensor with shape (200, 16)
    X_train = torch.randn(200, 16)

    # Convert to numpy for compatibility with seaborn and matplotlib
    X_train_np = X_train.numpy()

    # Set the dimensions of the grid
    grid_rows, grid_cols = 4, 4

    # Create a figure with subplots
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(20, 20))

    # Iterate over each subplot and plot the PDF for each feature
    for i, ax in enumerate(axs.flat):
        # Check if we have more subplots than features
        if i < X_train_np.shape[1]:
            sns.histplot(X_train_np[:, i], kde=True, ax=ax)
            ax.set_title(f'PDF of Feature {i + 1}')
        else:
            ax.set_visible(False)  # Hide extra subplots

    # Adjust layout for better spacing
    plt.tight_layout()
    if mode == "show":
        plt.show() 
    elif mode == "dashboard":
        pyplot(plt, clear_figure=True)
    else:
        plt.savefig("kde_plots.png")

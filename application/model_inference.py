from domain.env import MODELDIR, MEAN_FUNC
import numpy as np
import torch
from scipy.stats import norm, multivariate_normal as mvn
from torch.distributions import MultivariateNormal, Normal
from copulae.core.linalg import cov2corr, corr2cov
from copulae.core.misc import rank_data
from adapters.preprocessing import prepare_dataset, preprocessing
from domain.gp_model import SAASGP, MyExactGP, SAASGPJAX
from gpytorch.kernels import AdditiveStructureKernel
from domain.feature_model.feature_modeling import inverse_map
from adapters.gpytorch.util import decompose_matrix, get_alphas, get_beta, get_thetas, LFSR, get_PPAAs, map_inverse_to_sample_feature, get_groups, group_RATE, get_posterior_variations, interaction_distant, measure_subset
from adapters.sklearn.dimension_reduction import kernel_pca
from adapters.gpytorch.plotting import kde_plots, plot_combined_pdf, plot_density, plot_interaction_pdfs, mean_and_confidence_region, plot_2d_mvn, plot_3d_mvn, plot_cov_insights, plot_dimension_insights
from domain.metrics import get_metrics, gaussian_log_likelihood
import random
from time import time
import re
from scipy.stats import pointbiserialr

def get_data(use_synthetic_data=False, inference="exact", sws="LLVM_energy", y_type="performance", training_size=1000):
    ds = prepare_dataset(dummy_data=use_synthetic_data, sws=sws, y_type=y_type)
    feature_names, X_train, X_test, y_train, y_test = preprocessing(ds, training_size=training_size)
    # slice X_test such that it has the same shape as X_train
    # TODO: this shouldn't be necessary
    if len(X_test) > len(X_train):
        X_test = X_test[:len(X_train)]
        y_test = y_test[:len(X_train)]

    # transform test data to tensor
    X_test = torch.tensor(X_test).float() if inference == "exact" else torch.tensor(X_test).double()
    y_test = torch.tensor(y_test).float() if inference == "exact" else torch.tensor(y_test).double()

    return (ds, X_train, X_test, y_train, y_test, feature_names)

def get_model(file_name):
    # split the file name and take the model type as the fifth last element
    model = file_name.split("_")[-5] if not "piecewise_polynomial" in file_name or "spectral_mixture" in file_name else file_name.split("_")[-6]
    kernel_type = file_name.split("_")[-4] if not "piecewise_polynomial" in file_name or "spectral_mixture" in file_name else file_name.split("_")[-5] + "_" + file_name.split("_")[-4]
    kernel_structure = file_name.split("_")[-3]
    training_size = int(file_name.split("_")[-2])
    # for the sws name take all tokes before the model type
    sws = "_".join(file_name.split("_")[:-6])
    y_type = file_name.split("_")[-6]
    use_synthetic_data = False if "synthetic" not in sws else True
    print(f"sws: {sws}")
    model_file = f"{MODELDIR}/{file_name}.pth"

    # get data that produced the model
    #TODO: ensure that the input data is the same as the data in the model
    data = get_data(use_synthetic_data=use_synthetic_data, inference=model, sws=sws, y_type=y_type, training_size=training_size)
    ds, X_train, X_test, y_train, y_test, feature_names = data

    # init model and load state dict
    if model == "exact":
        model = MyExactGP(X_train, y_train, feature_names, likelihood="gaussian", kernel=kernel_type, mean_func=MEAN_FUNC, structure=kernel_structure)
    elif model == "MCMC" and kernel_structure != "additive":
        model = SAASGP(X_train, y_train, feature_names, mean_func="constant", kernel_structure=kernel_structure, kernel_type=kernel_type)
        model.eval()
    elif model == "MCMC" and kernel_structure == "additive":
        model = SAASGPJAX(X_train, y_train, feature_names, mean_func="constant", kernel_structure=kernel_structure, kernel_type=kernel_type)
        model.eval()
    model.load_state_dict(torch.load(model_file, weights_only=True), strict=False)
    model_properties = (sws, y_type, kernel_type, kernel_structure, training_size)
    return model, model_properties, ds, X_train, X_test, y_train, y_test, feature_names

def build_mixture_model(model, meta, mode="inference"):
    X_train, X_test, y_train, y_test, feature_names = meta
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(model.X.float()) if mode == "inference" else model.posterior(X_test)
        posterior_predictive = model.posterior(X_test)
        confidence_region = posterior.mvn.confidence_region()
        # create dimensional model for MCMC mixture model
        if len(model.X.T) <= posterior.base_sample_shape[0]:
            dims = len(model.X.T)
        elif len(model.X.T) > posterior.base_sample_shape[0]:
            dims = posterior.base_sample_shape[0]
        dimensional_model = {}
        for dim in range(dims):
            dimensional_model[dim] = {}
            dimensional_model[dim]["X"] = X_train[:, dim]
            dimensional_model[dim]["feature_name"] = feature_names[dim]
            dimensional_model[dim]["y"] = y_train
            dimensional_model[dim]["X_test"] = X_test[:, dim]
            dimensional_model[dim]["mean"] = posterior.mean[dim]
            dimensional_model[dim]["variance"] = posterior.variance[dim]
            dimensional_model[dim]["std"] = torch.sqrt(posterior.variance[dim])
            dimensional_model[dim]["covariance"] = posterior.covariance_matrix[dim]
            dimensional_model[dim]["correlation"] = cov2corr(np.array(dimensional_model[dim]["covariance"])) if not isinstance(model, MyExactGP) else "fix broadcasting"
            dimensional_model[dim]["marginal"] = Normal(dimensional_model[dim]["mean"], torch.sqrt(posterior.variance[dim])).cdf(dimensional_model[dim]["mean"])
            dimensional_model[dim]["inverse_transform"] = Normal(dimensional_model[dim]["mean"], torch.sqrt(posterior.variance[dim])).icdf(dimensional_model[dim]["marginal"])
            dimensional_model[dim]["lower"] = confidence_region[0][dim]
            dimensional_model[dim]["upper"] = confidence_region[1][dim]
    
    return posterior, dimensional_model

def BAKR(models, data, confidence=None):
    model, posterior, dimensional_model = models
    X_train, X_test, y_train, y_test, feature_names = data

    # posterior predictive checks
    mean_vector = posterior.mvn.mean if isinstance(model, MyExactGP) else dimensional_model[0]["mean"] # example for first dimension
    cov_matrix = posterior.mvn.covariance_matrix if isinstance(model, MyExactGP) else dimensional_model[0]["covariance"] # example for first dimension
    U, lam, V = decompose_matrix(cov_matrix)

    # inference according to BAKR https://github.com/lorinanthony/BAKR/blob/master/Tutorial/BAKR_Tutorial.R
    explained_var = np.cumsum(np.array(lam) / np.sum(np.array(lam)))
    p = len(model.X.T)
    # estimate, how much of the explained variance is explained by p components
    p_explained_var = explained_var[p - 1]
    print(f"{p_explained_var}.2f of the variance is explained by {p} components (the base features)")
    if confidence is None:
        confidence = get_metrics(posterior, y_test, posterior.mixture_mean, type="GP")["explained_variance"] if isinstance(model, SAASGP) or isinstance(model, SAASGPJAX) else get_metrics(posterior, y_test, posterior.mean.squeeze(), type="GP")["explained_variance"]
    q = np.where(explained_var >= confidence)[0][0] + 1 # number of principal components to explain confidential proportion of variance
    #qq = next(x[0] for x in enumerate(explained_var) if x[1] > CONFIDENCE) + 1
    #qqq = next(i + 1 for i, var in enumerate(explained_var) if var >= CONFIDENCE)
    Lambda = np.diag(np.sort(lam)[::-1])[:q] # diagonal matrix with first q eigenvalues 
    full_latent_space = U @ lam @ V.T # full latent space
    U = U[:, :q] # first q columns
    B = inverse_map(model.X.T, U) # inverse map of the latent space with p x q
    Laplace_approximation = B @ B.T
    # project the latent space from a fixed column to all other columns with granularity q
    thetas = get_thetas(cov_matrix, q)
    betas = get_beta(B, thetas)
    lfsr = LFSR(betas)
    print(f"lfsr: {lfsr}")
    
    inference_dict = {}
    inference_dict["mean_vector"] = mean_vector
    inference_dict["cov_matrix"] = cov_matrix
    inference_dict["explained_var"] = explained_var
    inference_dict["p_explained_var"] = p_explained_var
    inference_dict["q"] = q
    inference_dict["full_latent_space"] = full_latent_space
    inference_dict["Lambda"] = Lambda
    inference_dict["U"] = U
    inference_dict["B"] = B
    inference_dict["Laplace_approximation"] = Laplace_approximation
    inference_dict["thetas"] = thetas
    inference_dict["betas"] = betas
    inference_dict["lfsr"] = lfsr

    return inference_dict

def get_influentials(betas, feature_names):
    print(f"look at influencial features on different significance levels")
    # take low rank approximation
    # count from 0.8 to 0.995 in steps of 0.005 to get the influencial features on different significance levels
    for s in range(970, 996, 5):
        feature_idx, PPAAs = get_PPAAs(betas, (feature_names), sigval=s/1000)
        # fully print the PPAAs
        for i in range(1):
            print(f"s={s/1000}: PPAA {i} with {sum(PPAAs[i*random.randint(0, 200)])} influencial features")
            # output the names of the influencial features
            non_influentials = []
            for j in range(len(PPAAs[i])): # 
                if PPAAs[i][j] > 0:
                    print(f"feature {feature_idx[j]}:{feature_names[feature_idx[j]]} on level {j} - influencial? {PPAAs[i][j]} with beta-value {betas[i][j]}") # level j is the significance level
                else:
                    non_influentials.append((feature_names[feature_idx[j]], betas[i][j]))
            print(f"non influencial features: {non_influentials}")

def get_lengthscales(model, kernel_structure, feature_names):
    if model == "MCMC" and kernel_structure == "additive":
        #TODO: this is just a workaround and doesn't look nice
        num_features = len(feature_names)
        num_lengthscales = len(model.median_lengthscale)

        # Sort indices of median_lengthscale
        lengthscale_argsorted = sorted(range(num_lengthscales), key=lambda i: model.median_lengthscale[i])

        # Get the sorted lengthscale values
        lengthscale_sorted = [model.median_lengthscale[i] for i in lengthscale_argsorted]

        # Create pairs of feature names and sorted lengthscales
        sorted_lengthscale = [
            (feature_names[lengthscale_argsorted[i] % num_features], lengthscale_sorted[i]) for i in range(num_lengthscales)
        ]

        for feature, lengthscale in sorted_lengthscale:
            print(f"{feature}: {lengthscale:.4f}")
        
        return sorted_lengthscale
    
    elif model == "MCMC":
        kernel_lengthscales = [(feature_names[i],model.median_lengthscale[i]) for i in range(len(model.median_lengthscale))]
        # make list of tuples with feature name and lengthscale sorted by lengthscale
        lengthscale_sorted = list(torch.sort(model.median_lengthscale)[0])
        lengthscale_argsorted = list(torch.argsort(model.median_lengthscale))
        feature_names = list(feature_names)
        sorted_lengthscale = [(feature_names[lengthscale_argsorted[i]], lengthscale_sorted[i]) for i in range(len(model.median_lengthscale))]
        sorted_features = [feature_names[lengthscale_argsorted[i]] for i in range(len(model.median_lengthscale))]
        print(f"sorted features: {sorted_features} having lengthscales: {lengthscale_sorted}")
        for feature, lengthscale in sorted_lengthscale:
            print(f"lengthscale of {feature}: {lengthscale}")
        return sorted_lengthscale
    else:
        print("Exact GP model doesn't have lengthscale values.")

def main():
    # some example model files that have been checked
    #file_name = "x264_energy_fixed-energy_MCMC_matern52_simple_20_20240528-195232" #works
    #file_name = "Apache_energy_large_performance_exact_matern32_simple_100_20240529-122609" # works
    #file_name = "Apache_energy_large_performance_exact_RFF_additive_100_20240529-122609" # works
    #file_name = "Apache_energy_large_performance_exact_RFF_additive_100_20240528-201735" # works
    #file_name = "synthetic_3_exact_poly2_additive_20_20240529-104346" #works
    #file_name = "synthetic_3_MCMC_matern32_simple_500_20240529-104346" # works
    #file_name = "synthetic_4_MCMC_poly3_simple_100_20240529-104346" # works
    #file_name = "LLVM_energy_performance_MCMC_poly3_simple_20_20240531-090709" # works
    #file_name = "x264_energy_fixed-energy_MCMC_matern52_additive_100_20240528-201735" # works also with interactions
    file_name = "synthetic_2_MCMC_piecewise_polynomial_additive_100_20240529-081912" # works but without interactions
    #file_name = "Apache_energy_large_performance_exact_matern32_additive_20_20240529-122609" # works
    #file_name = "synthetic_2_MCMC_poly2_additive_500_20240531-120101"
    #file_name = "LLVM_energy_performance_MCMC_matern52_additive_20_20240916-170756" # works
    #file_name = "LLVM_energy_fixed-energy_MCMC_matern52_simple_20_20240916-175023"

    # certain models have errors
    #file_name = "VP8_pervolution_energy_bin_performance_MCMC_poly3_additive_20_20240612-183658" # won't work
    #file_name = "Apache_energy_large_performance_MCMC_RFF_simple_100_20240528-201735" # RuntimeError: expected scalar type Float but found Double
    #file_name = "synthetic_3_MCMC_RFF_simple_500_20240531-210016" RuntimeError: expected scalar type Float but found Double

    model, model_properties, ds, X_train, X_test, y_train, y_test, feature_names = get_model(file_name)
    print(f"model: {model}")
    print(f"model properties: {model_properties}")
    sws, y_type, kernel_type, kernel_structure, training_size = model_properties

    posterior, dimensional_model = build_mixture_model(model, (X_train, X_test, y_train, y_test, feature_names), mode="inference")

    # inference according to BAKR
    inference_dict = BAKR((model, posterior, dimensional_model), (X_train, X_test, y_train, y_test, feature_names))

    # get the influencial features
    get_influentials(inference_dict["betas"], feature_names)

    # measure the influence of the features on the target variable
    for i in range(len(feature_names)):
        print(f"feature {feature_names[i]}: {pointbiserialr(X_train[:,i], y_train)}")
    
    # get the lengthscales of the kernel
    get_lengthscales(model, kernel_structure, feature_names)

    ## PLOTTING SECTION FOR MCMC MODELS
    if isinstance(model, SAASGP) or isinstance(model, SAASGPJAX):
        pass
        # plot feature wise
        #grid_plot(X_train, y_train, X_test, posterior.mean, confidence_region)
        kde_plots(X_train, y_train)
        # plot combined
        selected_dimensions = [0,1,2]
        for dim in selected_dimensions:
            mean = np.mean(dimensional_model[dim]["mean"].detach().numpy())
            variance = np.median(dimensional_model[dim]["variance"].detach().numpy())
            feature_name = dimensional_model[dim]["feature_name"]
            plot_density(mean, variance, feature_name)
        dim_pairs = [
            (0,1,3), (0,1,4), (0,1,5),
        ]
        for tuple in dim_pairs:
            dimensional_submodel = {}
            for dim in tuple:
                dimensional_submodel[dim] = dimensional_model[dim]
            plot_combined_pdf(dimensional_submodel)

    plot_combined_pdf(dimensional_model)

    if kernel_structure == "additive" and "synthetic" in sws:
        print(f"Interactions are not supported for additive kernel structures on synthetic data for now. Sorry!")
    elif isinstance(model, SAASGP) or isinstance(model, SAASGPJAX):
        # measure and plot interactions
        SELECTED_FEATURES = [(1,0), (2,1), (3,0), (4,0)]#,(3,0),(5,1),(10,0)] # list of tuples which features are selected (feature_dim, on/off)
        # compute group RATE according to Crawford et al. 2019
        # j is the feature group that occurs in the rows of the data set
        j, groups = get_groups(X_test, SELECTED_FEATURES)
        print(f"groups:\n {groups}")
        group_rate = group_RATE(inference_dict["mean_vector"], inference_dict["U"], j) #TODO: How to visualize the group rate or KLD in general?
        print(f"group rate: {group_rate}")
        opposites, interactions = get_posterior_variations(model, X_train, SELECTED_FEATURES)
        #opposites, interactions = get_posterior_variations(model, X_test, SELECTED_FEATURES)
        measure_subset(model, ds, SELECTED_FEATURES)
        dimensional_submodel = {}
        dimensional_submodel["0"] = {}
        dimensional_submodel["0"]["mean"] = opposites.mixture_mean.detach().numpy()
        dimensional_submodel["0"]["std"] = np.sqrt(opposites.mixture_variance.detach().numpy())
        dimensional_submodel["0"]["feature_name"] = "without subset selection"
        dimensional_submodel["1"] = {}
        dimensional_submodel["1"]["mean"] = interactions.mixture_mean.detach().numpy()
        dimensional_submodel["1"]["std"] = np.sqrt(interactions.mixture_variance.detach().numpy())
        dimensional_submodel["1"]["feature_name"] = "with subset selection"

        # compare the two PDFs
        plot_interaction_pdfs(dimensional_submodel, SELECTED_FEATURES)
    else:
        print("Plotting interactions are only supported for MCMC models now. Sorry!")
    
    
    plot_2d_mvn(dimensional_model[0], dimensional_model[1], posterior.mvn)
    #plot_cov_insights(dimensional_model, num_dims=[0])
    plot_dimension_insights(dimensional_model, dim1=0)
    print(f"done.")

if __name__ == "__main__":
    main()
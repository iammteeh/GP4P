from adapters.model_store import get_store, get_modeldata, get_model_like
import pandas as pd
from domain.env import RESULTS_DIR, SAVE_FIGURES
from matplotlib import pyplot as plt
import seaborn as sns
from adapters.scoring import score_top_models, compare_with, compare_score_with_size, compare_complexity_with_kernels, compare_complexity_by_structure, compare_complexity_by_polykernel, compare_complexity_by_genkernel, compare_with

synthetic_data_with_BIC = ["synthetic_p2_with_BIC", "synthetic_p3_with_BIC", "synthetic_p4_with_BIC"]
synthetic_data = ["synthetic_p2_unscaled", "synthetic_p3_unscaled", "synthetic_p4_unscaled"]
datasets = ["Apache_energy_large_performance", "HSQLDB_energy_fixed-energy", "HSQLDB_pervolution_energy_bin_performance", "LLVM_energy_performance", "PostgreSQL_pervolution_energy_bin_performance", "VP8_pervolution_energy_bin_performance", "x264_energy_fixed-energy"]
dataset_with_BIC = ["Apache_energy_large_performance_with_BIC", "HSQLDB_energy_fixed-energy_with_BIC", "HSQLDB_pervolution_energy_bin_performance_with_BIC", "LLVM_energy_performance_with_BIC", "PostgreSQL_pervolution_energy_bin_performance_with_BIC", "VP8_pervolution_energy_bin_performance_with_BIC", "x264_energy_fixed-energy_with_BIC"]
FIXED_PARAMS = {
    "inference_type": "exact",
    "kernel_type": "RBF",
    "kernel_structure": "simple",
    "training_size": 100,
}

def get_df(data):
    experiment = get_store(f"{RESULTS_DIR}/modelstorage_{data}.json")
    # flatten nested dictionary
    flattened_data = []
    for key, value in experiment.items():
        flat_entry = {'filename': value['filename'], **value['model'], **value['scores']}
        flattened_data.append(flat_entry)
    # convert to DataFrame
    df = pd.DataFrame(flattened_data)
    return df

def compare_internally_by_simple_structure(data, kernels=["poly2", "poly3", "poly4", "piecewise-polynomial", "RBF", "matern32" ,"matern52", "RFF", "spectral_mixture"], structures=["simple"]):
    results = []
    for data in data:
        df = get_df(data)
        for size in [20, 100, 500]:
            for inference in ["exact", "MCMC"]:
                for kernel in kernels:
                    for structure in structures:
                        query_condition = (
                            (df["inference_type"] == inference) &
                            (df["training_size"] == size) &
                            (df["kernel_type"] == kernel) &
                            (df["kernel_structure"] == structure)
                        )
                        filtered_df = df[query_condition]
                        best_model = filtered_df.loc[filtered_df[SCORE].idxmax()]
                        print(f"Best {kernel} and {structure} for training size {size} and inference type {inference}: {best_model[[SCORE]]}")
                        results.append({
                            "dataset": data,
                            "kernel_type": kernel,
                            "kernel_structure": structure,
                            "MAPE": best_model[SCORE],
                            "training_size": size,
                            "inference_type": inference,
                        })
            
    results_df = pd.DataFrame(results)
    # Plotting

    # simple structure
    for structure in structures:
        fig, ax = plt.subplots(figsize=(10, 6))
        for kernel in kernels: 
            subset = results_df[(results_df['kernel_type'] == kernel) & (results_df['kernel_structure'] == structure)]
            sns.lineplot(x='dataset', y='MAPE', data=subset, marker='o', ax=ax, label=f"{kernel} - {structure}")
        
        ax.set_xlabel('Complexity')
        ax.set_ylabel(f"{SCORE}")
        title = f"How complexity affects model performance with fixed {structure} structure."
        ax.set_title(title)
        ax.legend(title='Kernel Type', loc='lower right')
        ax.grid(True)
        kernels = "_".join(kernels)
        plt.savefig(f"{RESULTS_DIR}/6.3/{title}_{SCORE}_with_{kernels}.png") if SAVE_FIGURES else plt.show()

def compare_internally_by_additive_structure(data, kernels=["poly2", "poly3", "poly4", "piecewise-polynomial", "RBF", "matern32" ,"matern52", "RFF", "spectral_mixture"], structures=["additive"]):
    results = []
    for data in data:
        df = get_df(data)
        for size in [20, 100, 500]:
            for inference in ["exact", "MCMC"]:
                for kernel in kernels:
                    for structure in structures:
                        query_condition = (
                            (df["inference_type"] == inference) &
                            (df["training_size"] == size) &
                            (df["kernel_type"] == kernel) &
                            (df["kernel_structure"] == structure)
                        )
                        filtered_df = df[query_condition]
                        best_model = filtered_df.loc[filtered_df[SCORE].idxmax()]
                        print(f"Best {kernel} and {structure} for training size {size} and inference type {inference}: {best_model[[SCORE]]}")
                        results.append({
                            "dataset": data,
                            "kernel_type": kernel,
                            "kernel_structure": structure,
                            "MAPE": best_model[SCORE],
                            "training_size": size,
                            "inference_type": inference,
                        })
            
    results_df = pd.DataFrame(results)
    # Plotting

    # simple structure
    for structure in structures:
        fig, ax = plt.subplots(figsize=(10, 6))
        for kernel in kernels: 
            subset = results_df[(results_df['kernel_type'] == kernel) & (results_df['kernel_structure'] == structure)]
            sns.lineplot(x='dataset', y='MAPE', data=subset, marker='o', ax=ax, label=f"{kernel} - {structure}")
        
    ax.set_xlabel('Complexity')
    ax.set_ylabel(f"{SCORE}")
    title = f"How complexity affects model performance with fixed {structure} structure."
    ax.set_title(title)
    ax.legend(title='Kernel Type', loc='lower right')
    ax.grid(True)
    kernels = "_".join(kernels)
    plt.savefig(f"{RESULTS_DIR}/6.3/{title}_{SCORE}_with_{kernels}.png") if SAVE_FIGURES else plt.show()

def compare_internally_with_kernels(data, kernels=["poly2", "poly3", "poly4", "piecewise-polynomial", "RBF", "matern32" ,"matern52", "RFF", "spectral_mixture"]):
    results = []
    for data in data:
        df = get_df(data)
        for size in [20, 100, 500]:
            for inference in ["exact", "MCMC"]:
                for kernel in kernels:
                    for structure in ["simple", "additive"]:
                        query_condition = (
                            (df["inference_type"] == inference) &
                            (df["training_size"] == size) &
                            (df["kernel_type"] == kernel) &
                            (df["kernel_structure"] == structure)
                        )
                        filtered_df = df[query_condition]
                        best_model = filtered_df.loc[filtered_df[SCORE].idxmax()]
                        print(f"Best {kernel} and {structure} for training size {size} and inference type {inference}: {best_model[[SCORE]]}")
                        results.append({
                            "dataset": data,
                            "kernel_type": kernel,
                            "kernel_structure": structure,
                            "MAPE": best_model[SCORE],
                            "training_size": size,
                            "inference_type": inference,
                        })
            
    results_df = pd.DataFrame(results)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel in kernels:
        for structure in ["simple", "additive"]:
            subset = results_df[(results_df['kernel_type'] == kernel) & (results_df['kernel_structure'] == structure)]
            sns.lineplot(x='dataset', y='MAPE', data=subset, marker='o', ax=ax, label=f"{kernel} - {structure}")

    ax.set_xlabel('Complexity')
    ax.set_ylabel(f"{SCORE}")
    kernels = ", ".join(kernels)
    title = f"How complexity affects model performance with fixed {kernels}."
    ax.set_title(title)
    ax.legend(title='Kernel Type')
    ax.grid(True)
    plt.savefig(f"{RESULTS_DIR}/6.4/{title}_{SCORE}.png") if SAVE_FIGURES else plt.show()

# 1. Interne Validierung

#for data in synthetic_data:
    df = get_df(data)
    # generate table data for 6.2.1
    for score in ["MAPE", "ESS", "RMSE"]:
        #score_top_models(data, df, score=score, k=3)
        # generate Figures 6.1
        compare_score_with_size(df, score="MAPE")
        # generate Figures 6.2
        compare_score_with_size(df, score="ESS")

# generate Figures 6.3
for score in ["MAPE", "ESS", "RMSE"]:
    SCORE = score
    compare_internally_by_simple_structure(data=synthetic_data, kernels=["RBF", "RFF", "matern52"])
    compare_internally_by_additive_structure(data=synthetic_data, kernels=["RBF", "RFF", "matern52"])

    compare_internally_by_simple_structure(data=synthetic_data, kernels=["poly2", "poly3", "poly4"])
    compare_internally_by_additive_structure(data=synthetic_data, kernels=["poly2", "poly3", "poly4"])

    compare_internally_by_simple_structure(data=synthetic_data, kernels=["poly2", "spectral_mixture", "matern52"])
    compare_internally_by_additive_structure(data=synthetic_data, kernels=["poly2", "spectral_mixture", "matern52"])

# generate Figures 6.4
for score in ["MAPE", "ESS", "RMSE", "BIC"]:
    SCORE = score
    for kernel in ["poly2", "RBF", "matern52"]:
        compare_internally_with_kernels(data=synthetic_data_with_BIC, kernels=[kernel])

# 2.Externe Validierung

# generate first run table data 6.1 6.2 6.3
for score in ["ESS", "MAPE", "BIC"]:
    print(f"\\begin{{table}}[H]")
    print(f"\\centering")
    print(f"\\begin{{tabular}}{{l|l|c|c|c|c|c|cc}}")
    if score == "ESS":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & ESS & BIC & MAPE \\\\ \\hline")
    elif score == "MAPE":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & MAPE & BIC & ESS \\\\ \\hline")
    elif score == "BIC":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & BIC & ESS & MAPE \\\\ \\hline")
    elif score == "RMSE":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & RMSE & ESS & MAPE \\\\ \\hline")
    for data in datasets:
        data = data
        df = get_df(data)
        score_top_models(data, df, score=score, k=1)
        #compare_with(data)
        #compare_score_with_size()
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Top model for each dataset with highest {score}.}}")
    print(f"\\label{{tab:top_{score}_models}}")
    print(f"\\end{{table}}")

# generate second run table data 6.4 6.5 6.6
for score in ["ESS", "MAPE", "BIC"]:
    print(f"\\begin{{table}}[H]")
    print(f"\\centering")
    print(f"\\begin{{tabular}}{{l|l|c|c|c|c|c|cc}}")
    if score == "ESS":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & ESS & BIC & MAPE \\\\ \\hline")
    elif score == "MAPE":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & MAPE & BIC & ESS \\\\ \\hline")
    elif score == "BIC":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & BIC & ESS & MAPE \\\\ \\hline")
    elif score == "RMSE":
        print(f"Dataset & Kernel & Structure & Training Size & Inference & RMSE & ESS & MAPE \\\\ \\hline")
    for data in dataset_with_BIC:
        data = data
        df = get_df(data)
        score_top_models(data, df, score=score, k=1)
        #compare_with(data)
        #compare_score_with_size()
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Top model for each dataset with highest {score}.}}")
    print(f"\\label{{tab:top_{score}_models}}")
    print(f"\\end{{table}}")

# generate table data 6.7 6.8 6.9 for average performance
eval_dict = {}
for score in ["ESS", "MAPE", "BIC", "RMSE"]:
    print(f"\\begin{{table}}[H]")
    print(f"\\centering")
    print(f"\\begin{{tabular}}{{l|l|c|cc}}")
    print(f"Dataset & Kernel & Structure & {score} \\\\ \\hline")
    for data in dataset_with_BIC:
        df = get_df(data)
        # get average score for kernel type and structure
        best_models = df.groupby(["kernel_type", "kernel_structure"]).mean(numeric_only=True)
        best_model_so_far = None
        for kernel, structure in best_models.index:
            best_model = best_models.loc[kernel, structure]
            if best_model_so_far is None or best_model[score] < best_model_so_far[score]:
                best_model_so_far = best_model
        data = data.split("_")[0]
        print(f"{data} & {best_model_so_far.name[0]} & {best_model_so_far.name[1]} & {best_model_so_far[score]:.2f} \\\\")
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Best model with best {score} score on average for each dataset.}}")
    print(f"\\label{{tab:best_{score}_models}}")
    print(f"\\end{{table}}")


exit()
    
# 3. Korrelationen
# Korrelation der average kernel scores über alle Datensätze

# Query the DataFrame
query_condition = (
    (df["inference_type"] == FIXED_PARAMS["inference_type"])
)
filtered_df = df[query_condition]
print(filtered_df)
# get model with highest score
best_model = df.loc[df[SCORE].idxmax()]
print(best_model)
# check for correlation between training size and ESS
correlation = df["training_size"].corr(df["ESS"])
print(f"Correlation between training size and ESS: {correlation}")
# check for correlation between RMSE, MAPE
correlation_matrix = df[['RMSE', 'MAPE']].corr()
print(f"Correlation between RMSE and MAPE: {correlation_matrix}")
correlation_matrix = df[['ESS', 'MAPE']].corr()
print(f"Correlation between ESS and MAPE: {correlation_matrix}")
correlation_matrix = df[['ESS', 'BIC']].corr()
print(f"Correlation between ESS and BIC: {correlation_matrix}")
correlation_matrix = df[['BIC', 'RMSE']].corr()
print(f"Correlation between BIC and RMSE: {correlation_matrix}")
filtered_top_10 = filtered_df.nlargest(10, SCORE)
print(filtered_top_10)
# get model with highest score for each dataset
print(df.groupby("kernel_type")) # 
best_models = df.groupby("kernel_type").max()
print(best_models)
# get model with highest score and sort by training size
best_models = best_models.sort_values(by="training_size", ascending=False)
print(best_models)
# get model with highest score and sort by kernel type and kernel structure
best_models = best_models.sort_values(by=["kernel_type", "kernel_structure"])
print(best_models)
# best kernel on average ESS
best_models = df.groupby("kernel_type").mean(numeric_only=True)
print(best_models)
#best_models = df.groupby(["kernel_type", "kernel_structure"]).mean().idxmax() #
score_list = []
for data in synthetic_data:
    experiment = get_store(f"{RESULTS_DIR}/modelstorage_{data}.json")
    df = get_df(data)
    df_dict = {}
    df_dict[data] = df
    df_dict["corr_rmse_mape"] = df[['RMSE', 'MAPE']].corr()
    df_dict["corr_ess_mape"] = df[['ESS', 'MAPE']].corr()
    #df_dict["corr_ess_bic"] = df[['ESS', 'BIC']].corr()
    score_list.append(df_dict)
for scores in score_list:
    print(scores["corr_rmse_mape"])
for scores in score_list:
    print(scores["corr_ess_mape"])
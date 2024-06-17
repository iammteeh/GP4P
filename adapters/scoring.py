from adapters.model_store import get_store, get_modeldata, get_model_like
import pandas as pd
from domain.env import RESULTS_DIR, SAVE_FIGURES
from matplotlib import pyplot as plt
import seaborn as sns

SCORE = "MAPE"
data = None

def score_top_models(data, df, score=["MAPE"], k=10):
    scoresheet = {}
    try:
        top_k = df.nlargest(k, score) if score == "ESS" else df.nsmallest(k, score)
        best_model = df.loc[df[score].idxmax()] if score == "ESS" else df.loc[df[score].idxmin()]
        # sort by score
        if score == "ESS":
            top_10_metrics = top_k[["kernel_type", "kernel_structure", "training_size", "inference_type", "ESS", "RMSE", "MAPE"]]
        elif score == "MAPE":
            top_10_metrics = top_k[["kernel_type", "kernel_structure", "training_size", "inference_type", "MAPE", "RMSE", "ESS"]]
        elif score == "BIC":
            top_10_metrics = top_k[["kernel_type", "kernel_structure", "training_size", "inference_type", "BIC", "ESS", "MAPE"]]
        elif score == "RMSE":
            top_10_metrics = top_k[["kernel_type", "kernel_structure", "training_size", "inference_type", "RMSE", "ESS", "MAPE"]]
        #print(f"Top {k} models with highest {score} for {data}: {top_10_metrics}")
        scoresheet = {
            "dataset": data,
            "kernel_type": best_model["kernel_type"],
            "kernel_structure": best_model["kernel_structure"],
            "training_size": best_model["training_size"],
            "inference_type": best_model["inference_type"],
            f"{score}": best_model[score],  # Convert set to string
        }
        # present scoresheet as latex table
        # display "Apache_energy_large_performance" as "Apache"
        data = data.split("_")[0]
        if score == "ESS":
            print(f"{data} & {best_model['kernel_type']} & {best_model['kernel_structure']} & {best_model['training_size']} & {best_model['inference_type']} & {best_model['ESS']:.2f} & {best_model['BIC']:.2f} & {best_model['MAPE']:.2f} \\\\")
        elif score == "MAPE":
            print(f"{data} & {best_model['kernel_type']} & {best_model['kernel_structure']} & {best_model['training_size']} & {best_model['inference_type']} & {best_model['MAPE']:.2f} & {best_model['BIC']:.2f} & {best_model['ESS']:.2f} \\\\")
        elif score == "BIC":
            print(f"{data} & {best_model['kernel_type']} & {best_model['kernel_structure']} & {best_model['training_size']} & {best_model['inference_type']} & {best_model['BIC']:.2f} & {best_model['ESS']:.2f} & {best_model['MAPE']:.2f} \\\\")
        elif score == "RMSE":
            print(f"{data} & {best_model['kernel_type']} & {best_model['kernel_structure']} & {best_model['training_size']} & {best_model['inference_type']} & {best_model['RMSE']:.2f} & {best_model['ESS']:.2f} & {best_model['MAPE']:.2f} \\\\")
    except KeyError:
        print(f"No {score} found in DataFrame for {data}")

def compare_with(df, query_condition, score="ESS"):
    results = []
    for size in [20, 50, 100, 250, 500]:
        for inference in ["exact", "MCMC"]:
            filtered_df = df[query_condition]
            best_model = filtered_df.loc[filtered_df[score].idxmax()]
            print(f"Best model for training size {size} and inference type {inference}: {best_model[[score]]}")
            results.append({
                "dataset": data,
                "training_size": size,
                "inference_type": inference,
                "MAPE": best_model[score],
            })
    print(results)
    
    results_df = pd.DataFrame(results)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for inference in ["exact", "MCMC"]:
        subset = results_df[results_df['inference_type'] == inference]
        ax.plot(subset['training_size'], subset['MAPE'], marker='o', label=inference)

    ax.set_xlabel('Training Size')
    ax.set_ylabel(f"{SCORE}")
    title = f"Best Model Performance by Training Size and Inference Type for {data}"
    ax.set_title(title)
    ax.legend(title='Inference Type')
    ax.grid(True)
    plt.savefig(f"{RESULTS_DIR}/{title}_{SCORE}.png") if SAVE_FIGURES else plt.show()

def compare_score_with_size(df, score=SCORE):
    results = []
    for size in [20, 50, 100, 250, 500]:
        for inference in ["exact", "MCMC"]:
            query_condition = (
                (df["inference_type"] == inference) &
                (df["training_size"] == size)
            )
            filtered_df = df[query_condition]
            best_model = filtered_df.loc[filtered_df[score].idxmax()]
            print(f"Best model for training size {size} and inference type {inference}: {best_model[[score]]}")
            results.append({
                "dataset": data,
                "training_size": size,
                "inference_type": inference,
                "MAPE": best_model[score],
            })
    print(results)
    
    results_df = pd.DataFrame(results)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for inference in ["exact", "MCMC"]:
        subset = results_df[results_df['inference_type'] == inference]
        ax.plot(subset['training_size'], subset['MAPE'], marker='o', label=inference)

    ax.set_xlabel('Training Size')
    ax.set_ylabel(f"{SCORE}")
    title = f"Best Model Performance by Training Size and Inference Type for {data}"
    ax.set_title(title)
    ax.legend(title='Inference Type')
    ax.grid(True)
    plt.savefig(f"{RESULTS_DIR}/{title}_{SCORE}.png") if SAVE_FIGURES else plt.show()

def compare_complexity_with_kernels(df):
    results = []
    for size in [20, 100, 500]:
        for inference in ["exact", "MCMC"]:
            for kernel in ["poly2", "RBF" ,"matern52"]:
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
                    print(results)
            
    results_df = pd.DataFrame(results)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel in ["poly2", "RBF" ,"matern52"]:
        for structure in ["simple", "additive"]:
            subset = results_df[(results_df['kernel_type'] == kernel) & (results_df['kernel_structure'] == structure)]
            sns.lineplot(x='dataset', y='MAPE', data=subset, marker='o', ax=ax, label=f"{kernel} - {structure}")

    ax.set_xlabel('Complexity')
    ax.set_ylabel(f"{SCORE}")
    title = f"How complexity affects model performance with fixed kernels."
    ax.set_title(title)
    ax.legend(title='Kernel Type')
    ax.grid(True)
    plt.savefig(f"{RESULTS_DIR}/{title}_{SCORE}.png") if SAVE_FIGURES else plt.show()

def compare_complexity_by_structure(df, kernels=["poly2", "poly3", "poly4", "piecewise-polynomial", "RBF", "matern32" ,"matern52", "RFF", "spectral_mixture"], structures=["simple", "additive"]):
    results = []
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
                    print(results)
            
    results_df = pd.DataFrame(results)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

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
        plt.savefig(f"{RESULTS_DIR}/6.3/{title}_{SCORE}.png") if SAVE_FIGURES else plt.show()

def compare_complexity_by_polykernel(df):
    results = []
    for size in [20, 100, 500]:
        for inference in ["exact", "MCMC"]:
            for kernel in ["poly2", "poly3", "poly4", "piecewise-polynomial"]:
                for structure in ["simple", "additive"]:
                    query_condition = (
                        (df["inference_type"] == inference) &
                        (df["training_size"] == size) &
                        (df["kernel_type"] == kernel) &
                        (df["kernel_structure"] == structure)
                    )
                    filtered_df = df[query_condition]
                    try:
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
                        print(results)
                    except ValueError:
                        print(f"No model found for {kernel} and {structure} for training size {size} and inference type {inference}")
            
    results_df = pd.DataFrame(results)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel in ["poly2", "poly3", "poly4", "piecewise_polynomial"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for structure in ["simple", "additive"]:
            subset = results_df[(results_df['kernel_type'] == kernel) & (results_df['kernel_structure'] == structure)]
            sns.lineplot(x='dataset', y='MAPE', data=subset, marker='o', ax=ax, label=f"{kernel} - {structure}")
        
        ax.set_xlabel('Complexity')
        ax.set_ylabel(f"{SCORE}")
        title = f"How complexity affects model performance with fixed {kernel} kernel."
        ax.set_title(title)
        ax.legend(title='Kernel Structure', loc='lower right')
        ax.grid(True)
        plt.savefig(f"{RESULTS_DIR}/{title}_{SCORE}.png") if SAVE_FIGURES else plt.show()

def compare_complexity_by_genkernel(df):
    results = []
    for size in [20, 100, 500]:
        for inference in ["exact", "MCMC"]:
            for kernel in ["RBF", "matern32" ,"matern52", "RFF", "spectral_mixture"]:
                for structure in ["simple", "additive"]:
                    query_condition = (
                        (df["inference_type"] == inference) &
                        (df["training_size"] == size) &
                        (df["kernel_type"] == kernel) &
                        (df["kernel_structure"] == structure)
                    )
                    filtered_df = df[query_condition]
                    try:
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
                        print(results)
                    except ValueError:
                        print(f"No model found for {kernel} and {structure} for training size {size} and inference type {inference}")
            
    results_df = pd.DataFrame(results)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel in ["RBF", "matern52", "RFF", "spectral_mixture"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for structure in ["simple", "additive"]:
            subset = results_df[(results_df['kernel_type'] == kernel) & (results_df['kernel_structure'] == structure)]
            sns.lineplot(x='dataset', y='MAPE', data=subset, marker='o', ax=ax, label=f"{kernel} - {structure}")
        
        ax.set_xlabel('Complexity')
        ax.set_ylabel(f"{SCORE}")
        title = f"How complexity affects model performance with fixed {kernel} kernel."
        ax.set_title(title)
        ax.legend(title='Kernel Structure', loc='lower right')
        ax.grid(True)
        plt.savefig(f"{RESULTS_DIR}/{title}_{SCORE}.png") if SAVE_FIGURES else plt.show()

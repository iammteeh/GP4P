from application.init_pipeline import init_pipeline, get_numpy_features
from domain.env import USE_DUMMY_DATA, EXTRAFUNCTIONAL_FEATURES
from itertools import combinations
import torch
from torch import Tensor
import numpy as np
from adapters.botorch.utility import LinearPredictionUtility
from botorch.test_functions import Branin, Hartmann

def get_X_y():
    ds, feature_names, X_train, X_test, y_train, y_test = init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features="polynomial" if EXTRAFUNCTIONAL_FEATURES else None, scaler="minmax")
    print(f"fit model having {X_train[1].shape[1]} features: {feature_names}")
    X_train, X_test, y_train, y_test = get_numpy_features(X_train, X_test, y_train, y_test)
    return torch.tensor(X_train).double(), torch.tensor(X_test), torch.tensor(y_train).double(), torch.tensor(y_test).double(), feature_names

def init_utility():
    X_train, X_test, y_train, y_test, feature_names = get_X_y()
    return LinearPredictionUtility(X_train, y_train, feature_names)

def utility(X, utility=None):
    #"""Utility function for the Branin test function"""
    #return Branin(negate=True).forward(X)
    if not utility:
        utility = init_utility()
        return utility.forward(X)
    else:
        return utility.forward(X)
    

def outcome_constraint(X):
    """L1 constraint; feasible if less than or equal to zero."""
    return X.sum(dim=-1) - 3


def weighted_obj(X):
    """Feasibility weighted objective; zero if not feasible."""
    neg_hartmann6 = Hartmann(negate=True)
    return neg_hartmann6(X) * (outcome_constraint(X) <= 0).type_as(X)
    
def generate_comparisons(y, n_comp, noise=0.1, replace=False):
    """Create pairwise comparisons with noise"""
    # generate all possible pairs of elements in y
    print(f"generating {n_comp} comparisons from {y.shape[0]} elements")
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    print(f"all_pairs shape: {all_pairs.shape}")
    # randomly select n_comp pairs from all_pairs
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ]
    # add gaussian noise to the latent y values
    c0 = y[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
    c1 = y[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise
    reverse_comp = np.array(c0 < c1)
    comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
    comp_pairs = torch.tensor(comp_pairs).long()

    return comp_pairs

def make_new_data(X, comps, q_comp, next_X=None):
    """Given X and next_X,
    generate q_comp new comparisons between next_X
    and return the concatenated X and comparisons
    """
    # next_X is float by default; cast it to the dtype of X (i.e., double)
    if next_X:
        next_X = next_X.to(X)
    else:
        #next_X = SobolEngine(dimension=X.shape[-1]).draw(1).to(X)
        #next_X = torch.rand(1, X.shape[-1]).to(X)
        # select random X and return it as ndarray
        next_X = X[np.random.choice(range(X.shape[0]), 1, replace=False)]

    next_y = utility(next_X)
    next_comps = generate_comparisons(next_y, n_comp=q_comp)
    comps = torch.cat([comps, next_comps + X.shape[-2]]) # add offset to indices of next_comps 
    X = torch.cat([X, next_X])
    return X, comps

def update_random_observations(best_random, batch_size):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = torch.rand(batch_size, 6)
    next_random_best = weighted_obj(rand_x).max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random
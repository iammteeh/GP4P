from itertools import combinations
from domain.feature_model.boolean_masks import get_word_and_opposite, get_literals_and_interaction
import torch
import numpy as np

def get_feature_model(X, features):
    """
    suited for analyzing boolean feature sets e.g. for distribution analysis
    suited for one-hot encoded numerical features
    """
    words, opposites = get_word_and_opposite(X, features)
    literals, interactions = get_literals_and_interaction(X, features)
    return words, opposites, literals, interactions

def inverse_map(X, U):
    """
    map a latent feature vector back to its original feature space, also called association matrix
    translated from https://github.com/lorinanthony/BAKR/blob/master/Rcpp/BAKRGibbs.cpp
    """
    X = torch.tensor(X)
    U = torch.tensor(U)
    B = torch.pinverse(X.T, rcond=1.490116e-08) @ U
    return B
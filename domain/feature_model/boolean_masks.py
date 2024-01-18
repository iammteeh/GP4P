import numpy as np
import functools, itertools
from pysat.formula import CNF, WCNF
from functools import reduce

def kdnf(a,b):
    a = bool(a)
    b = bool(b)
    kdnf = (a & (not b)) | ((not a) & b) | (not (a & b)) | (a & b)
    return not(a and b) | (a and b) | (a & (not b)) | (b & (not a)) or not kdnf

# X[1][1] = 1
# X[1][3] = 0
#ab1 = X[(X[:,0] == 1) & (X[:,1] == 1)]
#ab1 >>>> array([[1, 1, 1, 0, 1]])
#for d in len(X.T):
#    for i in range(len(X)):
#        for j in range(len(X)):
#            if kdnf(X[i][d],X[j][d]) == 1:
#                print("X[",i,"][",d,"] and X[",j,"][",d,"] are compatible")
#            else:
#                print("X[",i,"][",d,"] and X[",j,"][",d,"] are not compatible")

def get_character(X, pos, opt):
    if len(X.T) < pos < 0:
        raise ValueError("pos must be a valid index of X.")
    if type(opt) != bool and opt not in [0,1]:
        raise ValueError("opt must be a \"boolean\".")
    return X[X[:,pos] == opt]

def get_word(X, pos1, opt1, pos2, opt2):
    if len(X.T) < pos1 < 0 or len(X.T) < pos2 < 0:
        raise ValueError("pos must be a valid index of X.")
    if type(opt1) != bool and opt1 not in [0,1] or type(opt2) != bool and opt2 not in [0,1]:
        raise ValueError("opt must be a \"boolean\".")
    return X[(X[:,pos1] == opt1) & (X[:,pos2] == opt2)] #(A & B)

def get_opposite(X, pos1, opt1, pos2, opt2):
    if len(X.T) < pos1 < 0 or len(X.T) < pos2 < 0:
        raise ValueError("pos must be a valid index of X.")
    if type(opt1) != bool and opt1 not in [0,1] or type(opt2) != bool and opt2 not in [0,1]:
        raise ValueError("opt must be a \"boolean\".")
    return X[(X[:,pos1] == (not opt1)) | (X[:,pos2] == (not opt2))] # (not A | not B)

def build_masks(X, conditions):
    masks = []
    for pos, opt in conditions:
        if len(X.T) < pos < 0:
            raise ValueError("pos must be a valid index of X.")
        if type(opt) != bool and opt not in [0,1]:
            raise ValueError("opt must be a \"boolean\".")
        masks.append(X[:,pos] == opt)
    return masks

def get_n_words(X, masks, clause="or"):
    return X[functools.reduce(lambda a,b: a | b, masks)] if clause == "or" else X[functools.reduce(lambda a,b: a & b, masks)]

def get_words(X, masks, clause="or"):
    if clause == "or":
        word = np.array(reduce(np.logical_or, masks))
    elif clause == "nor":
        words = np.array(np.logical_not(reduce(np.logical_or, masks)))
    elif clause == "and":
        words = np.array(reduce(np.logical_and, masks))
    elif clause == "nand":
        words = np.array(np.logical_not(reduce(np.logical_and, masks)))
    elif clause == "xor":
        # only works for 2 masks
        #return X[np.logical_xor(*masks)]
        # XOR mit reduce
        words = np.array(reduce(np.logical_xor, masks))
    
    return X[words]
    

def get_word_and_opposite(X, features):
    """features is a list of tuples (pos, opt)
    that builds numpy masks and clauses to get 
    feature words and opposites
    """
    masks = build_masks(X, features)
    words = get_words(X, masks, clause="or")
    # invert masks
    inv_features = features.copy()
    for i, (pos, opt) in enumerate(features):
        inv_features[i] = (pos, not opt)
    masks = build_masks(X, inv_features)
    opposites = get_words(X, masks, clause="and")

    return words, opposites

def get_literals_and_interaction(X, features):
    masks = build_masks(X, features)
    literals = get_words(X, masks, clause="xor")
    interactions = get_words(X, masks, clause="and")
    return literals, interactions

def get_opposites_and_interactions(X, features):
    masks = build_masks(X, features)
    opposites = get_words(X, masks, clause="nand")
    interactions = get_words(X, masks, clause="and")
    return opposites, interactions

def build_cnf(X, features):
    cnf = CNF()
    masks = build_masks(X, features)
    for mask in masks:
        clause = [i+1 if mask[i] else -(i+1) for i in range(len(mask))] # (A | B)
        cnf.append(clause)
    return cnf

def build_dnf(X, features):
    dnf = CNF()
    masks = build_masks(X, features)
    for mask in masks:
        clause = [-(i+1) if mask[i] else i+1 for i in range(len(mask))] # (A & B)
        dnf.append(clause)
    return dnf

def get_filtered_union(X, conditions):
    # Create an empty mask with all False values
    mask = np.zeros(len(X), dtype=bool)

    # Apply each condition and update the mask
    for pos, opt in conditions:
        if not (0 <= pos < X.shape[1]) or opt not in [0, 1]:
            raise ValueError("Invalid condition.")
        mask |= (X[:, pos] == opt) # Apply the condition

    # Use the mask to filter rows
    return X[mask]

def get_combined_conditions(X, conditions):
    combined_results = []

    # Generate all possible combinations of conditions
    for r in range(1, len(conditions) + 1):
        for subset in itertools.combinations(conditions, r):
            # Start with a mask that selects all rows
            mask = np.ones(len(X), dtype=bool)
            for pos, opt in subset:
                if not (0 <= pos < X.shape[1]):
                    raise ValueError("Invalid column index.")
                if opt not in [0, 1]:
                    raise ValueError("opt must be 0 or 1.")
                # Update the mask with the current condition
                mask &= (X[:, pos] == opt)
            # Apply the combined mask to filter rows
            combined_results.append(X[mask])

    return combined_results

def get_specific_combinations(X, combinations):
    combined_results = []

    for combination in combinations:
        mask = np.ones(len(X), dtype=bool)
        for pos, opt in combination:
            if not (0 <= pos < X.shape[1]):
                raise ValueError("Invalid column index.")
            if opt not in [0, 1]:
                raise ValueError("opt must be 0 or 1.")
            mask &= (X[:, pos] == opt)
        combined_results.append(X[mask])

    return combined_results


def main():
    # Example usage
    X = np.array([
        [1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0]
    ])

    features = [(0, 1), (1, 1), (2, 1)]  # Example features

    #char = get_character(X, 0, 1)
    words, opposites = get_word_and_opposite(X, features)
    print(f"X: {X}")
    print(f"words: {words}")
    print(f"opposites: {opposites}")

#if __name__ == "__main__":
#    main()
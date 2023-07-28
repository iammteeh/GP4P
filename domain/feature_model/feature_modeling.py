from itertools import combinations

def gen_feature_model(X):
    """
    for each feature in X, generate a new feature that is the product of all other features
    """
    X_constrained = X.copy()
    for a, b in combinations(X,2):
        X_constrained[f'{a} x {b}'] = X[a] * X[b]
    return X_constrained

def gen_feature_graph(X, constrains=None):
    # first draw all features where its value is 1
    for instance in X.iterrows():
        print(instance)
        print(instance[1].index)
        for feature in instance[1].index:
            print(feature)
            if instance[1][feature] == 1:
                # draw feature
                print(instance[1][feature])
        break


    def get_config(self, config_id):
        # something like return self.all_configs[config_id]
        pass


def draw_constrained_feature_set(feature_set, constraints):
    # draw all features where its value is 1
    for feature in feature_set:
        if feature_set[feature] == 1:
            # draw feature
            print(feature_set[feature])
    # draw all constraints
    for constraint in constraints:
        pass

def find_constrained_features(feature_set, constraints):
    # find all features where its value is 1
    for feature in feature_set:
        if feature_set[feature] == 1:
            # draw feature
            print(feature_set[feature])
    # draw all constraints
    for constraint in constraints:
        pass

def find_isolated_features():
    """
    select features that are not connected to any other feature
    """
    pass

def find_independent_features():
    """
    filter out correlated features which also occurs together in the same configuration or generally in the same context
    """
    pass

def additive_kernel_permutation(basis_kernel, items, k=3):
    import itertools
    from GPy.kern import Add, Prod, BasisFuncKernel
    import sys
    sys.setrecursionlimit(10000)
    permutations = [list(p) for p in itertools.combinations(items, r=k)]
    print(f"permutations: {permutations}")
    print(f"len permutations: {len(permutations)}")
    #locals().update({'k{}'.format(k): p for k,p in zip(range(k), range(len(permutations)))})
    additive_kernel = basis_kernel.copy()
    additive_kernel = Add([Prod([combination[0], combination[1]], name=f"Prod_{p}_{c}") 
                      for p, permutation in enumerate(permutations) 
                      for c, combination in enumerate(itertools.combinations(permutation, 2))])
    return additive_kernel
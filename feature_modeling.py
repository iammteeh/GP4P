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


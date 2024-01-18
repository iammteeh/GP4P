from domain.dataset import DataSet
from domain.env import SWS, MODE, DATA_SLICE_MODE, DATA_SLICE_AMOUNT, DATA_SLICE_PROPORTION, X_type, POLY_DEGREE, Y
from adapters.import_data import select_data
from domain.feature_model.boolean_masks import get_word_and_opposite, get_literals_and_interaction, get_opposites_and_interactions

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold, StratifiedKFold, LeaveOneGroupOut

from itertools import combinations

def prepare_dataset(dummy_data=False):
    if dummy_data:
        tips = sns.load_dataset("tips")
        tips = pd.get_dummies(tips)
        y = tips["tip"]
        feature_names = ["total_bill", "sex_Male", "smoker_Yes", "size"]
        X = tips[feature_names]
        return {
            "X": X, 
            "feature_names":feature_names, 
            "y": y
            }

    data = select_data(SWS)
    if MODE != "simple":
        #folder = MODELDIR + data['sws_name']
        return DataSet(folder=data['sws_path'], performance_attribute=Y, value_type=X_type)
    else:
        return pd.read_csv(data['measurements_file_cleared'], sep=';')
    
def poly_feature_names(sklearn_feature_name_output, df):
    """
    This function takes the output from the .get_feature_names() method on the PolynomialFeatures 
    instance and replaces values with df column names to return output such as 'Col_1 x Col_2'

    sklearn_feature_name_output: The list object returned when calling .get_feature_names() on the PolynomialFeatures object
    df: Pandas dataframe with correct column names
    """
    import re
    cols = df.columns.tolist()
    feat_map = {'x'+str(num):cat for num, cat in enumerate(cols)}
    feat_string = ','.join(sklearn_feature_name_output)
    for k,v in feat_map.items():
        feat_string = re.sub(fr"\b{k}\b",v,feat_string)
    return feat_string.replace(" "," x ").split(',') 

# begin preprocessing
def add_polynomial_features(X, degree=POLY_DEGREE):
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    target_feature_names = [' x '.join(['{}'.format(pair[0],pair[1]) for pair in tuple if pair[1] != 0 ]) for tuple in [zip(X.columns,p) for p in poly.powers_]]
    X_poly = pd.DataFrame(X_poly, columns = target_feature_names)
    # remove perfectly colinear features
    for a, b in combinations(X,2):
        # IF (A==B AND #A==#B) pandas style for categorial/binary features
        if (((X[a].all().sum() == len(X[a])) & (X[b].all().sum() == len(X[b]))) | ((X[a].all().sum() == 0) & (X[b].all().sum() == 0) & (len(X[a]) == len(X[b])))):
            X_poly.drop(f'{a} x {b}', axis=1)
    return X_poly

def add_features(X, extra_ft):
    if extra_ft == "polynomial":
        X = add_polynomial_features(X, degree=POLY_DEGREE)
    elif extra_ft == "2_poly":
        X = add_polynomial_features(X, degree=2)
        print(len(X.columns))
    elif extra_ft == "3_poly":
        X = add_polynomial_features(X, degree=3)
    else:
        return X
    print(len(X.columns))
    return X

def scale_features(X, y, scaler):
    if scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "minmax":
        scaler = MinMaxScaler()
    elif scaler == "robust":
        scaler = RobustScaler()
    model = {
        "none": X,
        "standard": scaler.fit_transform(X, y),
        "minmax": scaler.fit_transform(X, y),
        "robust": scaler.fit_transform(X, y)
    }
    return model

def select_features(ds, feature_set, mode="opposites_and_interactions"):
    """
    select certain boolean masked features from the dataset
    """
    # as we require a ndarray here, we need to convert the dataset to ndarray
    if type(ds) is DataSet:
        # as length of feature set stays the same we don't need this for now
        #columns = []
        #for feature in feature_set:
        #    columns.append(ds.get_measurement_df().columns[feature[0]])
        #print(f"Selecting features: {columns}")
        columns = ds.get_measurement_df().columns
        ds = ds.get_measurement_df().to_numpy(copy=False)
    if mode == "literals_and_interactions":
        left, right = get_literals_and_interaction(ds, feature_set)
    elif mode == "opposites_and_interactions":
        left, right = get_opposites_and_interactions(ds, feature_set)
    elif mode == "words_and_opposites":
        left, right = get_word_and_opposite(ds, feature_set)

    print(f"left shape: {left.shape}")
    print(f"right shape: {right.shape}")
    left = pd.DataFrame(left, columns=columns)
    right = pd.DataFrame(right, columns=columns)
    return left, right

def define_subsets(ds, feature_group, mode="literals_and_interactions", to_numpy=True):
    print(f"build subsets (can take a while)...")
    literals, interactions = select_features(ds, feature_group, mode=mode)
    X_literals, y_literals = split_X_y(literals)
    X_interactions, y_interactions = split_X_y(interactions)

    X_literals, y_literals = get_data_slice(X_literals, y_literals)
    X_interactions, y_interactions = get_data_slice(X_interactions, y_interactions)

    if to_numpy:
        return (X_literals.to_numpy(), y_literals.to_numpy()), (X_interactions.to_numpy(), y_interactions.to_numpy())
    else:
        return (X_literals, y_literals), (X_interactions, y_interactions)

def build_train_test(ds):
    X, y = split_X_y(ds)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    
    feature_names = X_train.columns

    return feature_names, X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def store_model():
    pass 

def split_X_y(ds):
    if type(ds) is DataSet:
        df = ds.get_measurement_df()    
        X = df.drop('y', axis=1)
        y = df["y"]
    elif type(ds) is pd.DataFrame:
        X = ds.drop('y', axis=1)
        y = ds["y"]
    else:
        df = ds
        X = df["X"]
        y = df["y"]
    return X, y

def get_data_slice(X, y):
    if DATA_SLICE_MODE == "amount" and DATA_SLICE_AMOUNT < len(X):
        # get minimal data slice of n rows
        n = DATA_SLICE_AMOUNT
        X = X[:n]
        y = y[:n]
    elif DATA_SLICE_MODE == "proportion" and DATA_SLICE_PROPORTION < 1:
        # get proportional data slice of n rows
        n = len(X)-1
        p = DATA_SLICE_PROPORTION
        x = int(n * p)
        X = X[:x]
        y = y[:x]
    else:
        raise NotImplementedError
    return X, y

def preprocessing(ds, extra_ft=None, scaler=None):
    print("Preprocessing...")
    feature_names, X_train, X_test, y_train, y_test = build_train_test(ds)

    if extra_ft:
        # add extrafunctional feature model
        print(f"applying extrafunctional feature model: {extra_ft} (takes a while..)")
        X_train = add_features(X_train, extra_ft)
        X_test = add_features(X_test, extra_ft)
    
    if scaler:
        # scale features
        print(f"apply {scaler} Scaling")
        X_train = scale_features(X_train, y_train, scaler)
        X_test = scale_features(X_test, y_test, scaler)
    
    if len(X_train) > DATA_SLICE_AMOUNT or len(X_test) > DATA_SLICE_AMOUNT:
        print(f"Selected over {DATA_SLICE_AMOUNT} rows. Slicing data...")
        X_train, y_train = get_data_slice(X_train, y_train)
        X_test, y_test = get_data_slice(X_test, y_test)

    return feature_names, X_train, X_test, y_train, y_test
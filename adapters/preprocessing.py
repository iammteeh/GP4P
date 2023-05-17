from domain.dataset import DataSet
from domain.env import SWS, MODE, MODELDIR, X_type, POLY_DEGREE, Y
from adapters.import_data import select_data

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

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
    poly = PolynomialFeatures(degree=degree, interaction_only=True)
    X_poly = poly.fit_transform(X)
    target_feature_names = [' x '.join(['{}'.format(pair[0],pair[1]) for pair in tuple if pair[1] != 0 ]) for tuple in [zip(X.columns,p) for p in poly.powers_]]
    X_poly = pd.DataFrame(X_poly, columns = target_feature_names)
    # remove perfectly colinear features
    for a, b in combinations(X,2):
        # IF (A==B AND #A==#B) pandas style for categorial/binary features
        if (((X[a].all().sum() == len(X[a])) & (X[b].all().sum() == len(X[b]))) | ((X[a].all().sum() == 0) & (X[b].all().sum() == 0) & (len(X[a]) == len(X[b])))):
            X_poly.drop(f'{a} x {b}', axis=1, inplace=True)
    return X_poly

def add_features(X, extra_ft):
    model = {
        "none": X,
        "polynomial": add_polynomial_features(X, degree=POLY_DEGREE),
        "2_poly": add_polynomial_features(X, degree=2),
        "3_poly": add_polynomial_features(X, degree=3)
    }
    return model[extra_ft]

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

def store_model():
    pass 

def preprocessing(ds, extra_ft, scaler):
    print("Preprocessing...")
    if type(ds) is DataSet:
        df = ds.get_measurement_df()    
        X = df.drop('y', axis=1)
        y = df["y"]
    else:
        df = ds
        X = df["X"]
        y = df["y"]
    # use pandas dataframe methods

    # add extrafunctional feature model
    print(f"applying extrafunctional feature model: {extra_ft} (takes a while..)")
    X = add_features(X, extra_ft)
    # scale features
    #print(f"apply {scaler} Scaling")
    #X = scale_features(X, y, scaler)
    # convert to ndarray
    X = np.array(X.iloc[:])
    y = np.array(y.iloc[:])
    print("Preprocessing done!")
    # split data
    return train_test_split(X, y, test_size=0.8)
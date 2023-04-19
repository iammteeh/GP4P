from domain.dataset import DataSet
from env import SWS, MODE, MODELDIR, X_val, POLY_DEGREE, Y
from import_data import select_data

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from itertools import combinations

def prepare_dataset():
    data = select_data(SWS)
    if MODE != "simple":
        #folder = MODELDIR + data['sws_name']
        return DataSet(folder=data['sws_path'], performance_attribute=Y, value_type=X_val)
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
def add_polynomial_features(X):
    poly = PolynomialFeatures(POLY_DEGREE)
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
        "polynomial": add_polynomial_features(X)
    }
    return model[extra_ft]


def preprocessing(ds, extra_ft):
    if type(ds) is DataSet:
        df = ds.get_measurement_df()
    else:
        df = ds
    # use pandas dataframe methods
    X = df.drop('y', axis=1)
    y = df["y"]
    # add extrafunctional feature model
    X = add_features(X, extra_ft)

    return train_test_split(X, y, test_size=0.8)
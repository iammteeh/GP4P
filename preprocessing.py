import pandas as pd
from itertools import combinations
from dataclasses import dataclass
from bayesify.datahandler import ConfigSysProxy
from import_data import select_data
from env import MODE
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
@dataclass
class DataSet(ConfigSysProxy):
    def __init__(self, folder):
        super().__init__(folder)

        self.feature_length = self.get_feature_length()

    def parse_configs_csv(self, file):
        df = pd.read_csv(file, sep=";")
        print(df.head())
        # print(df)
        features = list(self.position_map.keys())
        configs_pd = df[features]
        configs = [tuple(x) for x in configs_pd.values.astype(bool)]
        if not self.attribute:
            nfps = df.drop(features, axis=1)
            col = list(nfps.columns.values)[0]
        else:
            col = self.attribute
        ys_pd = df[col]
        ys = np.array(ys_pd)
        performance_map = {c: y for c, y in zip(configs, ys)}
        return performance_map
    
    def get_feature_length(self):
        return len(self.all_configs.keys()[0])
    
    def generate_config(self, feature_length=1):
        key = tuple(np.random.randbool(feature_length))
        value = np.random.rand() # TODO: replace with real value within confidence interval of y
        return {key: value}

def prepare_dataset():
    data = select_data()
    if MODE != "simple":
        return DataSet(folder=data['sws_path'])
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
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)
    target_feature_names = [' x '.join(['{}'.format(pair[0],pair[1]) for pair in tuple if pair[1] != 0 ]) for tuple in [zip(X.columns,p) for p in poly.powers_]]
    X_poly = pd.DataFrame(X_poly, columns = target_feature_names)
    # remove perfectly colinear features
    for a, b in combinations(X,2):
        # IF (A==B AND #A==#B) pandas style for categorial/binary features
        if (((X[a].all().sum() == len(X[a])) & (X[b].all().sum() == len(X[b]))) | ((X[a].all().sum() == 0) & (X[b].all().sum() == 0) & (len(X[a]) == len(X[b])))):
            X_poly.drop(f'{a} x {b}', axis=1, inplace=True)
    return X_poly


def preprocessing(ds):
    X = ds.drop('y', axis=1)
    y = ds["y"]
    model = {
        "polynomial": add_polynomial_features(X)
    }
    X = model["polynomial"]
    return train_test_split(X, y, test_size=0.8)

# add check if ds is consistent to a panda dataframe
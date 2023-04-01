
from dataclasses import dataclass
from bayesify.datahandler import ConfigSysProxy
import os
import numpy as np
import pandas as pd

@dataclass
class DataSet(ConfigSysProxy):
    def __init__(self, folder=None, performance_attribute=None, value_type=bool):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        if not performance_attribute:
            performance_attribute = input("Enter performance attribute (Y):\n")

        self.value_type = value_type
        
        super().__init__(folder, performance_attribute)

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
    
    def update_prototype(self):
        self.prototype_config = list(self.value_type(0) for i in list(self.position_map.keys()))
    
    def get_feature_length(self):
        return len(list(self.all_configs.keys())[0])
    
    def generate_config(self, feature_length=1):
        key = tuple(np.random.randbool(feature_length))
        value = np.random.rand() # TODO: replace with real value within confidence interval of y
        return {key: value}
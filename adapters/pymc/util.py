import pymc as pm
import pickle
from domain.env import MODELDIR, MEAN_FUNC, KERNEL_TYPE, KERNEL_STRUCTURE
import datetime

def save_model(model, saved_trace, X):
    filename = MODELDIR + f"PMGP_{MEAN_FUNC}_{KERNEL_TYPE}_{KERNEL_STRUCTURE}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(filename, "wb") as buffer:
        pickle.dump({'model': model, 'trace': saved_trace, 'X': X}, buffer)

def load_model(model_fpath):
    return pickle.load(open(model_fpath, "rb"))
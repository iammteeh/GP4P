import os
import json
from domain.env import MODELDIR

STORE_PATH = MODELDIR + 'model_store.json'

def init_store(store_path=STORE_PATH):
    if not os.path.exists(store_path):
        with open(store_path, 'w') as f:
            json.dump({}, f)

def update_store(index, modeldata, store_path=STORE_PATH):
    with open(store_path, 'r') as f:
        store = json.load(f)
    
    store[index] = {
        'filename': modeldata['filename'],
        'model': {
            'dataset': modeldata['model']['dataset'],
            'benchmark': modeldata['model']['benchmark'],
            'kernel_type': modeldata['model']['kernel_type'],
            'kernel_structure': modeldata['model']['kernel_structure'],
            'inference_type': modeldata['model']['inference_type'],
            'training_size': modeldata['model']['training_size'],
            'timestamp': modeldata['model']['timestamp'],
        },
        'scores': {
            'RMSE': modeldata['scores']['RMSE'],
            'BIC': modeldata['scores']['BIC'],
            'MAPE': modeldata['scores']['MAPE'],
            'ESS': modeldata['scores']['ESS'],
            'last_loss': modeldata['scores']['last_loss'],
            'loss_curve': modeldata['scores']['loss_curve'],
            'training_time': modeldata['scores']['training_time'],
        },
    }
    
    with open(store_path, 'w') as f:
        json.dump(store, f, indent=4)

def get_store(store_path=STORE_PATH):
    with open(store_path, 'r') as f:
        store = json.load(f)
    return store

def get_modeldata(index, store_path=STORE_PATH):
    with open(store_path, 'r') as f:
        store = json.load(f)
    return store[index]

def get_model_like(index_regex, store_path=STORE_PATH):
    with open(store_path, 'r') as f:
        store = json.load(f)
    return {index: store[index] for index in store if index_regex in index}
import os
import json
from domain.env import MODELDIR

STORE_PATH = MODELDIR + 'model_store.json'

def init_store(store_path=STORE_PATH):
    if not os.path.exists(store_path):
        with open(store_path, 'w') as f:
            json.dump({}, f)

def update_store(index, filename, last_loss, loss_curve, RMSE, MAPE, ESS, timestamp, training_time, training_size, store_path=STORE_PATH):
    with open(store_path, 'r') as f:
        store = json.load(f)
    
    store[index] = {
        'filename': filename,
        'last_loss': last_loss,
        'loss_curve': loss_curve,
        'RMSE': RMSE,
        'MAPE': MAPE,
        'ESS': ESS,
        'timestamp': timestamp,
        'training_time': training_time,
        'training_size': training_size
    }
    
    with open(store_path, 'w') as f:
        json.dump(store, f, indent=4)

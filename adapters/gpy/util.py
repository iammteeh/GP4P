import GPy
import numpy as np

def save_model(file, model_param_array):
    np.save(file, model_param_array)

def load_model(file, X, y):
    m_load = GPy.models.GPRegression(X, y, initialize=False)
    m_load.update_model(False) # do not call the underlying expensive algebra on load
    m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
    m_load[:] = np.load('model_save.npy') # Load the parameters
    m_load.update_model(True) # Call the algebra only once
    return m_load
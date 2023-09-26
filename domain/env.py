import os 

PWD = os.getcwd()
DATADIR = PWD + '/rawdata/'
MODELDIR = PWD + '/modeldata/'
RESULTS_DIR = PWD + '/evaluation'
SWS = "LLVM_energy"
MODE = "not simple" # to choose which structure is induced with the data
USE_DUMMY_DATA = True
DATA_SLICE_MODE = "proportion"
DATA_SLICE_PROPORTION = 1/7
DATA_SLICE_AMOUNT = 1000
X_type = bool # default = bool for one-hot encoded numerical features, else use float
EXTRAFUNCTIONAL_FEATURES = False
POLY_DEGREE = 2
Y = "energy"
REGRESSION = "LassoCV"
REGRESSION_PENALTY = (-2, 4, 50) # (start, stop, num)
#ALPHAS = [0.1, 0.5, 1, 5, 10, 50, 100]
ALPHAS = (-2, 3, 1000)
COEFFICIENT_THRESHOLD = 3
MEAN_FUNC = "linear_weighted"
KERNEL_TYPE = "matern52"
KERNEL_STRUCTURE = "simple"
JAX = True
ARD = False
SAVE_FIGURES = True

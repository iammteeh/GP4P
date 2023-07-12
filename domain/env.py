import os 

PWD = os.getcwd()
DATADIR = PWD + '/rawdata/'
MODELDIR = PWD + '/modeldata/'
SWS = "LLVM_energy"
MODE = "not simple" # to choose which structure is induced with the data
USE_DUMMY_DATA = True
X_type = bool # default = bool for one-hot encoded numerical features, else use float
EXTRAFUNCTIONAL_FEATURES = False
POLY_DEGREE = 2
Y = "energy"
REGRESSION = "LassoCV"
REGRESSION_PENALTY = (-2, 4, 50) # (start, stop, num)
#ALPHAS = [0.1, 0.5, 1, 5, 10, 50, 100]
ALPHAS = (-2, 3, 1000)
COEFFICIENT_THRESHOLD = 3
SAVE_FIGURES = True
RESULTS_DIR = PWD + '/evaluation/'

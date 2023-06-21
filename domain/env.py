import os 

PWD = os.getcwd()
DATADIR = PWD + '/rawdata/'
MODELDIR = PWD + '/modeldata/'
SWS = "LLVM_energy"
MODE = "not simple" # to choose which structure is induced with the data
USE_DUMMY_DATA = False
X_type = float # for mcmc=float for probit/logistic=bool
EXTRAFUNCTIONAL_FEATURES = True
POLY_DEGREE = 2
Y = "energy"
REGRESSION = "LassoCV"
REGRESSION_PENALTY = (-2, 4, 50) # (start, stop, num)
#ALPHAS = [0.1, 0.5, 1, 5, 10, 50, 100]
ALPHAS = (-2, 3, 1000)
COEFFICIENT_THRESHOLD = 3
SAVE_FIGURES = True
RESULTS_DIR = PWD + '/evaluation/'

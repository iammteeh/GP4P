import os 

PWD = os.getcwd()
DATADIR = PWD + '/rawdata/'
MODELDIR = PWD + '/modeldata/'
SWS = "LLVM_energy"
MODE = "not simple" # to choose which structure is induced with the data
DUMMY_DATA = False
X_type = float # for mcmc=float for probit/logistic=bool
EXTRAFUNCTIONAL_FEATURES = True
POLY_DEGREE = 3
Y = "energy"
REGRESSION = "LassoCV"
REGRESSION_PENALTY = [0.1, 1, 10]
COEFFICIENT_THRESHOLD = 3

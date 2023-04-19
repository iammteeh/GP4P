import os 

PWD = os.getcwd()
DATADIR = PWD + '/rawdata/'
MODELDIR = PWD + '/modeldata/'
SWS = "LLVM_energy"
MODE = "not simple" # to choose which structure is induced with the data
X_val = bool
POLY_DEGREE = 4
Y = "energy"
REGRESSION = "lasso"
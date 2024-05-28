import os 
"""
How to use GP4P:
1. adjust paths as needed(
2. Select desired data set from rawdata directory or use dummy data
(3. try different feature encodings and set the type of X accordingly) UNSUPPORTED FOR NOW
4. for large data sets, use data slicing
5. set the response variable Y; for now only 1D is supported
6. choose mean function and kernel type/structure for GP
"""
PWD = os.getcwd()
DATADIR = PWD + '/rawdata/'
MODELDIR = PWD + '/modeldata/'
RESULTS_DIR = PWD + '/evaluation'
SWS = "LLVM_energy" # or LLVM_energy
MODE = "not simple" # to choose which structure is induced with the data
USE_DUMMY_DATA = False
# for synthetic data only #
EXTRAFUNCTIONAL_FEATURES = False
#
FEATURE_ENCODING = "binary" #TODO: fix one-hot encoding
X_type = bool # default = bool for one-hot or binary encoded numerical features, else use float
DATA_SLICE_MODE = "amount" # "proportion" or "amount"
DATA_SLICE_PROPORTION = 1/1
DATA_SLICE_AMOUNT = 100
Y = "fixed-energy" # {LLVM_energy: "energy", Apache_energy_large: "performance", HSQLDB_energy: "fixed-energy", HSQLDB_pervolution_energy_bin: "performance", PostgreSQL_pervolution_energy_bin: "performance", VP8_pervolution_energy_bin: "performance", x264_energy: "fixed-energy"}
# for GP pipelines #
POLY_DEGREE = 2 # control the degree of the polynomial kernel and synthetic data generation
MEAN_FUNC = "linear_weighted" # "linear_weighted", "constant", "zero"
KERNEL_TYPE = "matern52" # "linear", "RBF", "matern32", "matern52", "spectral_mixture", "RFF"
KERNEL_STRUCTURE = "simple" # "simple" or "additive"
###
SAVE_FIGURES = True
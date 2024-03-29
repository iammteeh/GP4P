import os 

PWD = os.getcwd()
DATADIR = PWD + '/rawdata/'
MODELDIR = PWD + '/modeldata/'
RESULTS_DIR = PWD + '/evaluation'
SWS = "LLVM_energy"
MODE = "not simple" # to choose which structure is induced with the data
USE_DUMMY_DATA = False
FEATURE_ENCODING = "binary" # try "binary", "one-hot" or "ordinal"
SELECTED_FEATURES = [(1,0), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (9,1)] # Example features for interaction scheme
DATA_SLICE_MODE = "amount"
DATA_SLICE_PROPORTION = 1/1
DATA_SLICE_AMOUNT = 333
X_type = bool # default = bool for one-hot or binary encoded numerical features, else use float
Y = "energy"
# for linear regression pipelines #
EXTRAFUNCTIONAL_FEATURES = False
POLY_DEGREE = 2
REGRESSION = "LassoCV"
REGRESSION_PENALTY = (-2, 4, 50) # np.logspace(start, stop, num)
#ALPHAS = [0.1, 0.5, 1, 5, 10, 50, 100]
ALPHAS = (-2, 3, 1000)
COEFFICIENT_THRESHOLD = 3
# for GP pipelines #
MEAN_FUNC = "linear_weighted"
KERNEL_TYPE = "matern52"
KERNEL_STRUCTURE = "simple"
GP_MODE = "latent" # "latent" or "marginal" (PyMC)
ARD = False
CANDIDATE_SAMPLER = "sobol" # one of "sobol", "cholesky", "rff", "lhs", "halton", "hammersley", "grid"
# PyMC sample strategies #
JAX = True
NUTS_SAMPLER = "numpyro"
###
SAVE_FIGURES = True
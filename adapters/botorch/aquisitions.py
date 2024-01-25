from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound, qSimpleRegret
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective
import torch
from botorch.test_functions import Branin, Hartmann
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.pairwise_samplers import PairwiseSobolQMCNormalSampler
from botorch.sampling.deterministic import DeterministicPosterior
from botorch.sampling.stochastic_samplers import StochasticSampler

def get_acq(model, best_f, acq="EI", objective=None):
    # for single ouput GP no need to specify objective
    sampler = SobolQMCNormalSampler(sample_shape=256, seed=torch.randint(1000000, (1,)).item())
    if acq == "EI":
        acq_func = qExpectedImprovement(model=model, best_f=best_f)
    elif acq == "PI":
        acq_func = qProbabilityOfImprovement(model=model, best_f=best_f, sampler=sampler)
    elif acq == "UCB":
        acq_func = qUpperConfidenceBound(model=model, beta=0.1, sampler=sampler)
    else:
        raise NotImplementedError
    return acq_func
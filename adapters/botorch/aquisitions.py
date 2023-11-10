from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound, qSimpleRegret
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective
from botorch.optim import optimize_acqf
import torch
from botorch.test_functions import Branin, Hartmann
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.pairwise_samplers import PairwiseSobolQMCNormalSampler
from botorch.sampling.deterministic import DeterministicPosterior
from botorch.sampling.stochastic_samplers import StochasticSampler

def optimize_acqf_and_get_observation(acq_func, dim, bounds, outcome_constraint, batch_size, num_restarts, raw_samples, noise_se):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    #neg_hartmann = Hartmann(negate=True, dim=dim)
    #exact_obj = neg_hartmann(new_x).unsqueeze(-1)  # add output dimension
    exact_obj = new_x.sum(dim=-1).unsqueeze(-1)
    exact_con = outcome_constraint(new_x).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj + noise_se * torch.randn_like(exact_obj)
    new_con = exact_con + noise_se * torch.randn_like(exact_con)
    return new_x, new_obj, new_con

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
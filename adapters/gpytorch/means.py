import torch
from gpytorch.means import Mean

class LinearMean(Mean):
    def __init__(self, beta, intercept):
        super().__init__()
        self.register_parameter('beta', torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32))) # this is the varying coefficient vector
        self.register_parameter('intercept', torch.nn.Parameter(torch.tensor(intercept, dtype=torch.float32)))
    
    def forward(self, x):
        return (self.beta @ x.t() + self.intercept).squeeze()
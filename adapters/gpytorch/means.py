import torch
from gpytorch.means import Mean

class LinearMean(Mean):
    def __init__(self, beta, intercept):
        super().__init__()
        self.register_parameter('beta', torch.nn.Parameter(torch.tensor(beta))) # this is the varying coefficient vector
        self.register_parameter('intercept', torch.nn.Parameter(torch.tensor(intercept)))
    
    def forward(self, x):
        return (self.beta @ x.t() + self.intercept).squeeze()
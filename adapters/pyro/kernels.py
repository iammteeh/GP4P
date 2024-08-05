import numpy as np
import torch
import math
from itertools import combinations
from botorch.models.fully_bayesian import compute_dists

_sqrt3 = math.sqrt(3)
root_five = math.sqrt(5)
five_thirds = 5.0 / 3.0

def square_scaled_dist(X, Z=None, lengthscale=1.0):
    r"""
    Returns :math:`\|\frac{X-Z}{l}\|^2`.
    """
    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X**2).sum(1, keepdim=True)
    Z2 = (scaled_Z**2).sum(1, keepdim=True)
    XZ = scaled_X.matmul(scaled_Z.t())
    r2 = X2 - 2 * XZ + Z2.t()
    return r2.clamp(min=0)

def scaled_dist(X, Z=None, lengthscale=1.0):
    r"""
    Returns :math:`\|\frac{X-Z}{l}\|`.
    """
    return torch.sqrt(square_scaled_dist(X, Z, lengthscale))

def polynomial_kernel(X, Z, bias=0, degree=2):
    # make sure X and Z are 1D tensors
    dot_product = torch.mm(X, Z.t())
    k = (dot_product + bias) ** degree
    return k  # N_X N_Z

def rbf_kernel(X, Z, lengthscale):
    r2 = square_scaled_dist(X, Z, lengthscale)
    k = torch.exp(-0.5 * r2)
    return k  # N_X N_Z

def matern_kernel(X, Z, inv_length_sq, nu=2.5):
    deltaXsq = torch.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    dsq = torch.sum(deltaXsq, axis=-1)  # N_X N_Z
    if nu == 0.5:
        k = torch.exp(-torch.sqrt(dsq))
    elif nu == 1.5:
        clipped_dsq = torch.Tensor(np.clip(dsq.detach().numpy(), 1.0e-12, None)) # workaround for torch.clip
        exponent = _sqrt3 * torch.sqrt(clipped_dsq)
        poly = 1.0 + exponent
        k = poly * torch.exp(-exponent)
    elif nu == 2.5:
        clipped_dsq = torch.Tensor(np.clip(dsq.detach().numpy(), 1.0e-12, None)) # workaround for torch.clip
        exponent = root_five * torch.sqrt(clipped_dsq)
        poly = 1.0 + exponent + five_thirds * dsq
        k = poly * torch.exp(-exponent)
    return k  # N_X N_Z

def _periodic_kernel(X: torch.Tensor, Z: torch.Tensor, lengthscale: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    r"""
    Periodic kernel:
    
    .. math::
        k(x, z) = \sigma^2 \exp\left(-2 \sin^2\left(\frac{\pi}{p} \|x - z\| \right) \right)
    """
    dist = compute_dists(X=X, lengthscale=lengthscale)
    clipped_dist = torch.Tensor(np.clip(dist.detach().numpy(), 1.0e-6, None)) # workaround for torch.clip
    k = torch.exp(-2 * (torch.sin(math.pi / period * clipped_dist) ** 2))
    return k

def periodic_kernel(X, Z, lengthscale, period=1.0):
    X = X[:, None, :]  # Shape (N_X, 1, P)
    Z = Z[None, :, :]  # Shape (1, N_Z, P)
    
    # Compute the sine squared part
    sin_sq = torch.sin(torch.pi * torch.abs(X - Z) / period) ** 2
    
    # Compute the kernel matrix
    K = torch.exp(-2 * sin_sq / lengthscale**2)
    
    return K  # Shape (N_X, N_Z)
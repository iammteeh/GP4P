import jax.numpy as jnp
from jax import jit, random
from functools import partial
import math
from itertools import combinations

root_five = math.sqrt(5.0)
five_thirds = 5.0 / 3.0

# compute diagonal component of kernel
def kernel_diag(var, noise, jitter=1.0e-6, include_noise=True):
    if include_noise:
        return var + noise + jitter
    else:
        return var + jitter


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def rbf_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    k = var * jnp.exp(-0.5 * jnp.sum(deltaXsq, axis=-1))
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def matern_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    dsq = jnp.sum(deltaXsq, axis=-1)  # N_X N_Z
    exponent = root_five * jnp.sqrt(jnp.clip(dsq, a_min=1.0e-12))
    poly = 1.0 + exponent + five_thirds * dsq
    k = var * poly * jnp.exp(-exponent)
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z

@partial(jit, static_argnums=(5,))
def polynomial_kernel(X, Z, var, bias, degree, include_noise):
    dot_product = jnp.dot(X, Z.T)  # N_X N_Z
    k = var * (dot_product + bias) ** degree
    if include_noise:
        k = k + (bias + 1.0e-6) * jnp.eye(X.shape[0])
    return k  # N_X N_Z

@partial(jit, static_argnums=(5,))
def rff_kernel(X, Z, var, inv_length_sq, num_features, key):
    N_X, P = X.shape
    N_Z = Z.shape[0]
    
    # Generate random weights and biases
    w_key, b_key = random.split(key)
    W = jnp.sqrt(2 * inv_length_sq) * random.normal(w_key, (num_features, P))
    b = 2 * jnp.pi * random.uniform(b_key, (num_features,))
    
    # Compute random features for X and Z
    Z_X = jnp.sqrt(2.0 * var / num_features) * jnp.cos(jnp.dot(X, W.T) + b)
    Z_Z = jnp.sqrt(2.0 * var / num_features) * jnp.cos(jnp.dot(Z, W.T) + b)
    
    # Compute the kernel matrix as the dot product of random features
    K = jnp.dot(Z_X, Z_Z.T)
    
    return K  # N_X N_Z

def periodic_kernel(X, Z, var, length_scale, period, noise, include_noise):
    X = X[:, None, :]  # Shape (N_X, 1, P)
    Z = Z[None, :, :]  # Shape (1, N_Z, P)
    
    # Compute the sine squared part
    sin_sq = jnp.sin(jnp.pi * jnp.abs(X - Z) / period) ** 2
    
    # Compute the kernel matrix
    K = var * jnp.exp(-2 * sin_sq / length_scale**2)
    
    if include_noise:
        K += (noise + 1.0e-6) * jnp.eye(X.shape[0])
    
    return K  # Shape (N_X, N_Z)

class SimpleJAXKernel:
    def __init__(self, var, inv_length_sq, noise, include_noise):
        self.var = var
        self.inv_length_sq = inv_length_sq
        self.noise = noise
        self.include_noise = include_noise

    def __call__(self, X, Z):
        return rbf_kernel(X, Z, self.var, self.inv_length_sq, self.noise, self.include_noise)

# Combine kernels using product
def product_kernel(k1, k2, X, Z, active_dims):
    return k1(X[:, active_dims], Z[:, active_dims]) * k2(X[:, active_dims], Z[:, active_dims])

# Scale kernel
def scale_kernel(k, scale, X, Z, active_dims):
    return scale * k(X[:, active_dims], Z[:, active_dims])

# Additive structure kernel
def additive_structure_kernel(X, Z, base_kernels):
    d_kernels = [
        lambda X, Z, i=i, j=j: scale_kernel(
            lambda X, Z: product_kernel(base_kernels[i], base_kernels[j], X, Z, [i, j]), 
            1.0, X, Z, [i, j]
        )
        for (i, _), (j, _) in combinations(enumerate(base_kernels), 2)
    ]
    
    K = sum(k(X, Z) for k in d_kernels)
    return K
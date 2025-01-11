import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="False"

import jax
import jax.numpy as jnp
from flax import nnx
from jax import random, grad
import optax
import orbax.checkpoint as ocp

print("JAX running on", jax.devices()[0].platform.upper())



class MLP(nnx.Module):
    
    def __init__(self, in_dim: int, out_dim: int, hidden_shape: jnp.ndarray, rngs: nnx.Rngs):
       
        self.rng = rngs
        self.key_seeded = jax.random.key(0)
        
        assert len(hidden_shape) > 0, "hidden_shape must be a list of at least one integer"
        
        self.layers = [nnx.Linear(in_dim, hidden_shape[0], rngs=self.rng)]
        for i in range(len(hidden_shape)-1):
            self.layers.append(nnx.Linear(hidden_shape[i], hidden_shape[i+1], rngs=self.rng))        

        self.optimizer = optax.adam(self, 1e-3)
        
    def __call__(self, x):
        
        for sublayer in self.layers[:-1]:
            x = nnx.leaky_relu(sublayer(x))
        
        return self.layers[-1](x)




class Emulator(nnx.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_shape: jnp.ndarray, rngs: nnx.Rngs):
        
        self.mlp = MLP(in_dim, out_dim, hidden_shape, rngs)
    
    
    
    
    
    
    
    
    
    # Define the loss function (mean squared error).
    @nnx.jit
    def mse_loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 
        return jnp.mean((self.mlp.__call__(x) - y) ** 2)

    # Define training step function
    @nnx.jit
    def train_step(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 

        grad_fn = nnx.value_and_grad(self.mse_loss)
        loss, grads = grad_fn(self, x, y)
        #self.optimizer.update(grads)  # In place updates.

        return loss

    # Define evaluation function
    @nnx.jit
    def eval_step(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 
        return self.mse_loss(model, x, y)



# Instantiate the model.
model = MLP(rngs=rng)
#graphdef, params = nnx.split(model, nnx.Param)
# Visualize it.






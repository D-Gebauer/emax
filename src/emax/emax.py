import time
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="False"

import jax
import jax.numpy as jnp
from flax import nnx
from jax import random, grad
import optax
import orbax.checkpoint as ocp

print("JAX running on", jax.devices()[0].platform.upper())



class Emulator(nnx.Module):
    
    def __init__(self, in_dim: int, out_dim: int, hidden_shape: jnp.ndarray, rngs: nnx.Rngs):
       
        self.rng = nnx.Rngs(0)
        self.key_seeded = jax.random.key(0)
        
        assert len(hidden_shape) > 0, "hidden_shape must be a list of at least one integer"
        
        self.layers = [nnx.Linear(in_dim, hidden_shape[0], rngs=self.rng)]
        
        
        for i in range(len(hidden_shape)):
            self.layers.append(nnx.Linear(hidden_shape[i], rngs=self.rng))        

        
    def __call__(self, x):
        
        for sublayer in self.layers[:-1]:
            x = self.leaky_relu(sublayer(x))
        
        return self.layers[-1](x)

    #l1 regularization
    @nnx.jit
    def l1_loss(x, alpha):
        return alpha * (jnp.abs(x)).sum()

    #l2 regularization
    @nnx.jit
    def l2_loss(x, alpha):
        return alpha * (x ** 2).sum()

    # Define the loss function (mean squared error).
    @nnx.jit
    def mse_loss(model: MLP, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 
        
        #y_pred = model(x)
        #loss = jnp.mean((y_pred - y) ** 2)
        #loss += 0.2 * jnp.mean(jnp.abs(y_pred - y)/(jnp.abs(y)))
        
        #loss += sum(l1_loss(w, alpha=0.001) for w in jax.tree.leaves(nnx.state(model, nnx.Param)))
        #loss += sum(l2_loss(w, alpha=0.001) for w in jax.tree.leaves(nnx.state(model, nnx.Param)))
        
        return jnp.mean((model(x) - y) ** 2)





# Instantiate the model.
model = MLP(rngs=rng)
#graphdef, params = nnx.split(model, nnx.Param)
# Visualize it.



# Define training step function
@nnx.jit
def train_step(model : MLP, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 

  grad_fn = nnx.value_and_grad(mse_loss)
  loss, grads = grad_fn(model, x, y)
  optimizer.update(grads)  # In place updates.

  return loss


# Define evaluation function
@nnx.jit
def eval_step(model : MLP, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 
  return mse_loss(model, x, y)

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from functools import partial

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
       
        assert len(hidden_shape) > 0, "hidden_shape must be a list of at least one integer"
        
        self.layers = [nnx.Linear(in_dim, hidden_shape[0], rngs=rngs)]
        for i in range(len(hidden_shape)-1):
            self.layers.append(nnx.Linear(hidden_shape[i], hidden_shape[i+1], rngs=rngs))        
        self.layers.append(nnx.Linear(hidden_shape[-1], out_dim, rngs=rngs))
        
    def __call__(self, x):
        
        for sublayer in self.layers[:-1]:
            x = nnx.leaky_relu(sublayer(x))
        
        return self.layers[-1](x)



@nnx.jit
def mse_loss(model: MLP, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 
    return jnp.mean((model(x) - y) ** 2)

@nnx.jit
def __train_step__(model: MLP, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray: 

    grad_fn = nnx.value_and_grad(mse_loss)
    loss, grads = grad_fn(model, x, y)
    optimizer.update(grads) # update weights in place

    return loss

@nnx.jit
def __eval_step__(model: MLP, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:  
    return mse_loss(model, x, y)


class Emulator(nnx.Module):

    def __init__(self, hidden_shape: jnp.ndarray, rngs: nnx.Rngs):
        
        self.data_loaded = False
        self.trained = False
        self.rng = rngs
        self.hidden_shape = hidden_shape
    
    def data_stream(self, x: jnp.ndarray, y: jnp.ndarray):
        assert x.shape[0] == y.shape[0]
        n = x.shape[0]
        
        indices = jax.random.permutation(self.rng(), n)
        for i in range(0, n, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield x[batch_indices], y[batch_indices]
    
    def load_data(self, x: jnp.ndarray, y: jnp.ndarray, batch_size : int=512, val_split : float=0.1, standardize: bool=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        
        self.standardize = standardize
        
        if standardize:
            self.x_mean = jnp.mean(x, axis=0)
            self.x_std = jnp.std(x, axis=0)
            self.y_mean = jnp.mean(y, axis=0)
            self.y_std = jnp.std(y, axis=0)
            
            self.x = (x - self.x_mean) / self.x_std
            self.y = (y - self.y_mean) / self.y_std
        
        
        val_inds = np.random.choice(x.shape[0], int(x.shape[0]*val_split), replace=False)
        self.x_val = x[val_inds]
        self.y_val = y[val_inds]
        self.x_train = np.delete(x, val_inds, axis=0)
        self.y_train = np.delete(y, val_inds, axis=0)
        
        self.n_batches_train = self.x_train.shape[0] // batch_size
        self.n_batches_val = self.x_val.shape[0] // batch_size
        if self.x_train.shape[0] % batch_size != 0:
            self.n_batches_train += 1
        if self.x_val.shape[0] % batch_size != 0:
            self.n_batches_val += 1
        
        self.data_loaded = True
    
        self.in_dim = x.shape[1]
        self.out_dim = y.shape[1]
    
        self.mlp = MLP(self.in_dim, self.out_dim, self.hidden_shape, self.rng)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        assert self.data_loaded, "No data loaded. Use load_data() first"
        
        if not self.standardize:
            return self.mlp.__call__(x)
        
        
        x = (x - self.x_mean) / self.x_std
        return self.mlp.__call__(x) * self.y_std + self.y_mean
    

    def train_epoch(self, optimizer: nnx.Optimizer, epoch: int) -> (float, float):
        train_loss = []
        for step, batch in tqdm(enumerate(self.data_stream(self.x_train, self.y_train)), total=self.n_batches_train, desc=f"Training epoch {epoch+1:03d}", leave=True, ):
            train_loss.append(__train_step__(self.mlp, optimizer, *batch))

        # Compute the metrics on the test set after each training epoch.
        val_loss = __eval_step__(self.mlp, self.x_val, self.y_val)
        
        return jnp.mean(jnp.array(train_loss)), val_loss

    def train(self, lr: float, max_epochs: int):
        
        momentum = 0.9

        optimizer = nnx.Optimizer(self.mlp, optax.nadam(lr, momentum, nesterov=True))

        cpath = os.getcwd()
        self.ckpt_dir = os.path.join(cpath, '.ckpt/')
        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.ckpt_dir = ocp.test_utils.erase_and_create_empty(self.ckpt_dir)

        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
        }

        checkpointer = ocp.StandardCheckpointer()

        epoch_lr_decreased = 0
        best_val_loss = np.inf
        best_val_loss_epoch = 0
        lr_decrease_step = 0

        for epoch in range(max_epochs):
            
            train_loss, val_loss = self.train_epoch(optimizer, epoch)
            
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1:03d}/{max_epochs} -- train loss: {self.metrics_history['train_loss'][-1]:.2e} -- val loss: {self.metrics_history['val_loss'][-1]:.2e}\n")
            
            if jnp.isnan(jnp.array([val_loss, train_loss])).any():
                print("\n NaN detected. Exiting.")
                break
            
            if val_loss < best_val_loss:
                
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                
                _, state = nnx.split(self.mlp)
                checkpointer.save(self.ckpt_dir / f'state_{epoch}', state)
            
            
            if epoch - best_val_loss_epoch > 5 and epoch - epoch_lr_decreased > 5:
                
                lr_decrease_step += 1
                epoch_lr_decreased = epoch
                optimizer = nnx.Optimizer(self.mlp, optax.adamw(lr*jnp.sqrt(0.1)**lr_decrease_step, momentum))
                
                graphdef, old_state = nnx.split(self.mlp)
                state_restored = checkpointer.restore(self.ckpt_dir / f'state_{best_val_loss_epoch}', old_state)
                self.mlp = nnx.merge(graphdef, state_restored)
                
                print(f"\n Decreasing learning rate to {(lr*jnp.sqrt(0.1)**lr_decrease_step):.3e}\n")
            
            
            if epoch - best_val_loss_epoch > 20:
                
                graphdef, old_state = nnx.split(self.mlp)
                state_restored = checkpointer.restore(self.ckpt_dir / f'state_{best_val_loss_epoch}', old_state)
                self.mlp = nnx.merge(graphdef, state_restored)
                
                print("\n Early stopping.")
                break
            
            if epoch == max_epochs-1:
                
                graphdef, old_state = nnx.split(self.mlp)
                state_restored = checkpointer.restore(self.ckpt_dir / f'state_{best_val_loss_epoch}', old_state)
                self.mlp = nnx.merge(graphdef, state_restored)
                
                print("\n Maximum number of epochs reached.")
                break
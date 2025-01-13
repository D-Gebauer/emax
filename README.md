# emax
## Introduction
Emax is a jax based implementation of a simple emulator architecture.

## Installation
To install emax and all necessary dependencies, run:
```
pip install git+https://github.com/D-Gebauer/emax.git
```

## Usage

First import the emulator class from emax and create an instance:

    from emax import Emulator

    model = Emulator(hidden_shape, rng)

where ```hidden_shape``` is a list with the number of neurons per layer and ```rng``` is an instance of ```flax.nnx.Rngs```.

Training data (with shape ```(n_samples, n_params)``` and ```(n_samples, n_features)``` for x and y, respectively) is then loaded with:

    model.load_data(x, y, batch_size=512, val_split=0.1, normalize=False)

At this step a neural network with input and output dimension matching the training data is automatically created. The argument ```normalize``` automatically rescales ```(x,y) -> ((x-x.mean(axis=0))/x.std(axis=0), (y-y.mean(axis=0))/y.std(axis=0))``` for training.

The model is then trained with:

    model.train(learning_rate, max_epochs)

Learning rate will automatically be reduced on plateau and training will be stopped if loss stops decreasing on the validation set.

Predictions are obtained with:

    model(x_test)

In case ```normalize=True``` ```x_test``` will automatically be rescaled, and the predictions will be transformed back to the original range. 

## Contact
For any questions or feedback, please contact me at git[at]gebauer.ai.
from src.activation import relu 
import numpy as np

def forward_pass_one_input(X, W, B):
    """
    Performs a forward pass through the recurrent neural network for one single input.
    
    :param X: vector of input features (e.g. words encoded into numbers)
    :param W: matrix of weights for the different neurons (shape: [layer_count, neurons_per_layer, coefficients_per_neuron])
    :param B: matrix of biases for the different neurons (shape: [layer_count, neurons_per_layer])
    """
    
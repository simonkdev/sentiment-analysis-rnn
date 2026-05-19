import pandas as pd
import numpy as np

def initialize_weights(neurons_hidden, output_neurons):
    """
    Initializes the weights for the recurrent neural network. 
    The weights are initialized randomly from a normal distribution with mean 0 and std 1. 

    :param neurons_layer_n: number of neurons in layer n
    :param output_neurons: number of output neurons (number of classes)
    :return: weight matrices for each layer
    """
    print("[ INIT ] Initializing weights...")
    W_1 = np.random.normal(0, 1, (neurons_hidden, 6)) # 6 = 1 (input) + 5 (hidden state)
    # W_2 = np.random.normal(0, 1, (neurons_hidden, neurons_hidden))
    # W_3 = np.random.normal(0, 1, (neurons_hidden,  neurons_hidden))
    # W_4 = np.random.normal(0, 1, (neurons_hidden, neurons_hidden))
    W_out = np.random.normal(0, 1, (output_neurons, neurons_hidden))
    print("[ INIT ] Weights initialized.")
    return W_1, W_out

def initialize_biases(neurons_hidden, output_neurons):
    """
    Initializes the biases for the recurrent neural network. 
    The biases are initialized randomly from a normal distribution with mean 0 and std 1. 

    :param neurons_layer_n: number of neurons in layer n
    :param output_neurons: number of output neurons (number of classes)
    :return: bias vectors for each layer
    """
    print("[ INIT ] Initializing biases...")
    B_1 = np.random.normal(0, 1, (neurons_hidden, 1))
    # B_2 = np.random.normal(0, 1, (neurons_hidden, 1))
    # B_3 = np.random.normal(0, 1, (neurons_hidden, 1))
    # B_4 = np.random.normal(0, 1, (neurons_hidden, 1))
    B_out = np.random.normal(0, 1, (output_neurons, 1))
    print("[ INIT ] Biases initialized.")
    return B_1, B_out
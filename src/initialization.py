import pandas as pd
import numpy as np

def initialize_weights(neurons_layer_1, neurons_layer_2, neurons_layer_3, neurons_layer_4, output_neurons):
    """
    Initializes the weights for the recurrent neural network. 
    The weights are initialized randomly from a normal distribution with mean 0 and std 1. 

    :param neurons_layer_n: number of neurons in layer n
    :param output_neurons: number of output neurons (number of classes)
    :return: weight matrices for each layer
    """
    W_1 = np.random.normal(0, 1, (neurons_layer_1, 2))
    W_2 = np.random.normal(0, 1, (neurons_layer_2, neurons_layer_1))
    W_3 = np.random.normal(0, 1, (neurons_layer_3, neurons_layer_2))
    W_4 = np.random.normal(0, 1, (neurons_layer_4, neurons_layer_3))
    W_out = np.random.normal(0, 1, (output_neurons, neurons_layer_4))
    return W_1, W_2, W_3, W_4, W_out

def initialize_biases(neurons_layer_1, neurons_layer_2, neurons_layer_3, neurons_layer_4, output_neurons):
    """
    Initializes the biases for the recurrent neural network. 
    The biases are initialized randomly from a normal distribution with mean 0 and std 1. 

    :param neurons_layer_n: number of neurons in layer n
    :param output_neurons: number of output neurons (number of classes)
    :return: bias vectors for each layer
    """
    B_1 = np.random.normal(0, 1, (neurons_layer_1, 1))
    B_2 = np.random.normal(0, 1, (neurons_layer_2, 1))
    B_3 = np.random.normal(0, 1, (neurons_layer_3, 1))
    B_4 = np.random.normal(0, 1, (neurons_layer_4, 1))
    B_out = np.random.normal(0, 1, (output_neurons, 1))
    return B_1, B_2, B_3, B_4, B_out
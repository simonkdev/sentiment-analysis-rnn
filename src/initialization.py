import numpy as np

def initialize_weights(neurons_hidden, output_neurons, input_features=1):
    """
    Initializes the weights for the recurrent neural network. 
    The weights are initialized randomly from a normal distribution with mean 0 and std 1. 

    :param neurons_layer_n: number of neurons in layer n
    :param output_neurons: number of output neurons (number of classes)
    :return: weight matrices for each layer
    """
    print("[ INIT ] Initializing weights...")
    # Xavier/Glorot initialization for tanh
    input_dim = input_features + neurons_hidden
    scale_w1 = np.sqrt(6.0 / (input_dim + neurons_hidden))
    scale_wout = np.sqrt(6.0 / (neurons_hidden + output_neurons))
    W_1 = np.random.normal(0, scale_w1, (neurons_hidden, input_dim))
    W_out = np.random.normal(0, scale_wout, (output_neurons, neurons_hidden))
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
    B_1 = np.random.normal(0, 1, (neurons_hidden, 1 ))
    # B_2 = np.random.normal(0, 1, (neurons_hidden, 1))
    # B_3 = np.random.normal(0, 1, (neurons_hidden, 1))
    # B_4 = np.random.normal(0, 1, (neurons_hidden, 1))
    B_out = np.random.normal(0, 1, (output_neurons, 1))
    print("[ INIT ] Biases initialized.")
    return B_1, B_out

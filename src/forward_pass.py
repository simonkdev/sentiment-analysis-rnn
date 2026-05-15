from src.activation import relu, softmax
import numpy as np

def forward_pass_one_input(X, W_1, W_2, W_3, W_4, W_out, B_1, B_2, B_3, B_4, B_out):
    """
    Performs a forward pass through the recurrent neural network for one single input.
    
    :param X: vector (list) of initial input features (shape: [1, d])
    :param W_t: weight matrix for layer t, shape: [neurons_in_previous_layer, 2]    
    :param B_t: bias matrix for layer t, shape: [neuron_layer_t, 1]
    """
    W_hh = 0
    for layer_index in range(len(X)):
        W_t = np.array(X[layer_index], W_hh)
        A_1 = forward_pass_one_layer_hidden(W_t, W_1, B_1)
        Z_1 = relu(A_1) # shape: [1, neurons_first_layer]
        A_2 = forward_pass_one_layer_hidden(Z_1, W_2, B_2)
        Z_2 = relu(A_2) # shape: [1, neurons_second_layer]
        A_3 = forward_pass_one_layer_hidden(Z_2, W_3, B_3)
        Z_3 = relu(A_3) # shape: [1, neurons_third_layer]
        A_4 = forward_pass_one_layer_hidden(Z_3, W_4, B_4)
        Z_4 = relu(A_4) # shape: [1, neurons_fourth_layer]
        if layer_index == len(X) - 1:
                A_out = forward_pass_one_input(Z_4, W_out, B_out)
                output = softmax(A_out)
                return output
        W_hh = Z_4
            
def forward_pass_one_layer_hidden(X_n, W_n, B):
    """
    Performs fwd pass for one hidden layer. 

    :param X_n: vector of input features for the n-th layer (shape: [n, d])
    :param W_n: matrix of weights for the n-th layer (1 row per neuron: (shape: [neurons_per_layer, 2]))
    :param B: vector of biases for the n-th layer (shape: [neurons_per_layer, 1])
    """
    A_n = np.dot(W_n, X_n.T) + B
    Z_n = relu(A_n)
    return Z_n.T # shape: 1, neurons_layer_n
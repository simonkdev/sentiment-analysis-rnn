from src.activation import relu, softmax
import numpy as np

def forward_pass_one_input(X, W_1, W_2, W_3, W_4, W_out, B_1, B_2, B_3, B_4, B_out):
    """
    Performs a forward pass through the recurrent neural network for one single input.
    
    :param X: vector (list) of initial input features (shape: [1, d])
    :param W_t: weight matrix for layer t, shape: [neurons_in_previous_layer, 2]    
    :param B_t: bias matrix for layer t, shape: [neuron_layer_t, 1]
    """
    index = 0
    W_hh = np.zeros((1, 5)) # initialize W_hh as an empty arrays
    for x_t in X:
        x_t = np.array([[x_t]])
        print(f"x_t: {x_t.shape}"
              f"\nW_hh: {W_hh.shape}")
        x_h = np.hstack((x_t, W_hh)) # shape: [1, 2]
        print(f"x_h: {x_h.shape}")
        Z_1 = forward_pass_one_layer_hidden(x_h, W_1, B_1)
        Z_2 = forward_pass_one_layer_hidden(Z_1, W_2, B_2)
        Z_3 = forward_pass_one_layer_hidden(Z_2, W_3, B_3)
        Z_4 = forward_pass_one_layer_hidden(Z_3, W_4, B_4)
        if index == len(X) - 1:
                output = forward_pass_one_layer_hidden(Z_4, W_out, B_out, output_layer=True)
                return output
        W_hh = Z_4
        index += 1
            
def forward_pass_one_layer_hidden(X_n, W_n, B, output_layer=False):
    """
    Performs fwd pass for one hidden layer. 

    :param X_n: vector of input features for the n-th layer (shape: [n, d])
    :param W_n: matrix of weights for the n-th layer (1 row per neuron: (shape: [neurons_per_layer, 2]))
    :param B: vector of biases for the n-th layer (shape: [neurons_per_layer, 1])
    """
    A_n = np.dot(X_n, W_n.T) + B.T
    if not output_layer: 
        Z_n = relu(A_n) 
    else: 
        Z_n = softmax(A_n)
    return Z_n # shape: 1, neurons_layer_n
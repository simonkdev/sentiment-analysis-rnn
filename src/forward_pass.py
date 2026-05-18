from src.activation import relu, softmax
import numpy as np

def forward_pass_one_input(X, W_1, W_2, W_3, W_4, W_out, B_1, B_2, B_3, B_4, B_out, passActivations=False):
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
        x_h = np.hstack((x_t, W_hh)) # shape: [1, 2]
        Z_1, A_1 = forward_pass_one_layer_hidden(x_h, W_1, B_1, passActivations=True)
        Z_2, A_2 = forward_pass_one_layer_hidden(Z_1, W_2, B_2, passActivations=True)
        Z_3, A_3 = forward_pass_one_layer_hidden(Z_2, W_3, B_3, passActivations=True)
        Z_4, A_4 = forward_pass_one_layer_hidden(Z_3, W_4, B_4, passActivations=True)
        if index == len(X) - 1:
                output, A_5 = forward_pass_one_layer_hidden(Z_4, W_out, B_out, output_layer=True, passActivations=True)
                if passActivations:
                    return output, A_1, A_2, A_3, A_4, A_5
                return output
        W_hh = Z_4
        index += 1
            
def forward_pass_one_layer_hidden(X_n, W_n, B, output_layer=False, passActivations=False):
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
    if passActivations:
        return Z_n, A_n
    return Z_n # shape: 1, neurons_layer_n

def forward_pass_input_vector(X, W_1, W_2, W_3, W_4, W_out, B_1, B_2, B_3, B_4, B_out, passActivations=False):
    X_list = X #.tolist()
    out_list = []
    a1_list = []
    a2_list = []
    a3_list = []
    a4_list = []
    a5_list = []
    for x in X_list:
        out, a1, a2, a3, a4, a5 = forward_pass_one_input(x, W_1, W_2, W_3, W_4, W_out, B_1, B_2, B_3, B_4, B_out, passActivations=True)
        out_list.append(out)
        a1_list.append(a1)
        a2_list.append(a2)
        a3_list.append(a3)
        a4_list.append(a4)
        a5_list.append(a5)
    if passActivations:
        return np.array(out_list).reshape(len(X), -1), np.array(a1_list), np.array(a2_list), np.array(a3_list), np.array(a4_list), np.array(a5_list)
    return np.array(out_list).reshape(X.shape[0], -1)
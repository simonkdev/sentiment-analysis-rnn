from src.activation import tanh, softmax
import numpy as np
import tqdm as tqdm

def forward_pass_one_input(X, W_1, W_out, B_1, B_out, passActivations=False):
    """
    Performs a forward pass through the recurrent neural network for one single input.
    
    :param X: vector (list) of initial input features (shape: [1, d])
    :param W_t: weight matrix for layer t, shape: [neurons_in_previous_layer, 2]    
    :param B_t: bias matrix for layer t, shape: [neuron_layer_t, 1]
    """
    neurons_hidden = W_1.shape[0]
    index = 0
    W_hh = np.zeros((1, neurons_hidden)) # initialize W_hh as an empty arrays
    all_Z_1 = []
    for x_t in X:
        x_t = np.array([[x_t]])
        x_h = np.hstack((x_t, W_hh)) # shape: [1, 2]
        Z_1, A_1 = forward_pass_one_layer_hidden(x_h, W_1, B_1, passActivations=True)
        all_Z_1.append(Z_1)
        if index == len(X) - 1:
                avg_Z_1 = np.mean(np.array(all_Z_1), axis=0)
                output, A_5 = forward_pass_one_layer_hidden(Z_1, W_out, B_out, output_layer=True, passActivations=True)
                if passActivations:
                    #print(f"x_h shape: {x_h.shape}")
                    return output, Z_1, A_1, A_5, x_h
                return output
        W_hh = Z_1
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
        Z_n = tanh(A_n) 
    else: 
        Z_n = softmax(A_n)
    if passActivations:
        return Z_n, A_n
    return Z_n # shape: 1, neurons_layer_n

def forward_pass_input_vector(X, W_1, W_out, B_1, B_out, passActivations=False):
    X_list = X #.tolist()
    out_list = []
    z1_list = []
    a1_list = []
    a5_list = []
    x_h_list = []
    for x in tqdm.tqdm(X_list):
        out, z1, a1, a5, x_h = forward_pass_one_input(x, W_1, W_out, B_1, B_out, passActivations=True)
        out_list.append(out)
        a1_list.append(a1.flatten())
        z1_list.append(z1.flatten())
        a5_list.append(a5.flatten())
        x_h_list.append(x_h.flatten())
    if passActivations:
        return np.array(out_list).reshape(len(X), -1), np.array(a1_list), np.array(z1_list), np.array(a5_list), np.array(x_h_list)
    return np.array(out_list).reshape(len(X), -1)
from src.activation import tanh, softmax
import numpy as np
import tqdm as tqdm

NEURONS_HIDDEN = 1

# def forward_pass_one_input(X, W_1, W_out, B_1, B_out, passActivations=False):
#     neurons_hidden = W_1.shape[0]
#     index = 0
#     W_hh = np.zeros((1, neurons_hidden))  # Initial hidden state
#     all_Z_1 = []  # Store ALL hidden states (not just the average)
#     all_A_1 = []
#     all_x_h = []  # Store ALL x_h (input + hidden) for backprop
#     for x_t in X:
#         x_t = np.array([[x_t]])
#         x_h = np.hstack((x_t, W_hh))  # [x_t, h_{t-1}]
#         all_x_h.append(x_h)  # Save for backprop
#         Z_1, A_1 = forward_pass_one_layer_hidden(x_h, W_1, B_1, passActivations=True)
#         all_Z_1.append(Z_1)
#         all_A_1.append(A_1)
#         if index == len(X) - 1:
#             # Use LAST hidden state (not average) for output
#             output, A_5 = forward_pass_one_layer_hidden(Z_1, W_out, B_out, output_layer=True, passActivations=True)
#             if passActivations:
#                 return output, np.array(all_Z_1), np.array(all_A_1), A_5, np.array(all_x_h)
#             return output
#         W_hh = Z_1  # Update hidden state for next step
#         index += 1
            
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

# def forward_pass_input_vector(X, W_1, W_out, B_1, B_out, passActivations=False):
#     X_list = X
#     out_list = []
#     all_Z_1_list = []  # List of all hidden states for EACH sequence
#     all_A_1_list = []
#     a5_list = []
#     all_x_h_list = []  # List of all x_h for EACH sequence
#     for x in tqdm.tqdm(X_list):
#         out, all_Z_1, all_A_1, a5, all_x_h = forward_pass_one_input(x, W_1, W_out, B_1, B_out, passActivations=True)
#         out_list.append(out)
#         all_Z_1_list.append(np.array(all_Z_1).reshape(-1, NEURONS_HIDDEN))
#         all_A_1_list.append(np.array(all_A_1).reshape(-1, NEURONS_HIDDEN))
#         all_x_h_list.append(np.array(all_x_h).reshape(-1, 1 + NEURONS_HIDDEN))
#         a5_list.append(a5)
        
#     if passActivations:
#         return (
#             np.array(out_list).reshape(len(X), -1),  # Predictions (batch, output_dim)
#             np.array(all_Z_1_list),                  # All hidden states (batch, seq_len, hidden_dim)
#             np.array(all_A_1_list),                  # All pre-activations (batch, seq_len, hidden_dim)
#             np.array(a5_list),                      # Output pre-activations (batch, output_dim)
#             np.array(all_x_h_list)                   # All x_h (batch, seq_len, 2)
#         )
#     return np.array(out_list).reshape(len(X), -1)

def forward_pass_input_vector(X, W_1, W_out, B_1, B_out, passActivations=False):
    """
    Vectorized forward pass for a batch of sequences.
    X: [batch_size, seq_len] array of token indices (padded)
    W_1: [neurons_hidden, 2] weight matrix for hidden layer
    W_out: [output_neurons, neurons_hidden] weight matrix for output layer
    B_1: [neurons_hidden, 1] bias for hidden layer
    B_out: [output_neurons, 1] bias for output layer
    passActivations: whether to return intermediate activations
    """
    batch_size, seq_len = X.shape
    neurons_hidden = W_1.shape[0]
    output_neurons = W_out.shape[0]

    # Initialize hidden state for all sequences: [batch_size, neurons_hidden]
    h_prev = np.zeros((batch_size, neurons_hidden))

    # Pre-allocate arrays for intermediate values
    all_Z_1 = np.zeros((batch_size, seq_len, neurons_hidden))
    all_A_1 = np.zeros((batch_size, seq_len, neurons_hidden))
    all_x_h = np.zeros((batch_size, seq_len, 1 + neurons_hidden))

    # Process each time step
    for t in range(seq_len):
        # Current token for all sequences: [batch_size, 1]
        x_t = X[:, t].reshape(-1, 1)

        # Concatenate with previous hidden state: [batch_size, 1 + neurons_hidden]
        x_h = np.concatenate([x_t, h_prev], axis=1)
        all_x_h[:, t, :] = x_h

        # Hidden layer forward pass
        A_1 = np.dot(x_h, W_1.T) + B_1.T  # [batch_size, neurons_hidden]
        Z_1 = tanh(A_1)

        all_A_1[:, t, :] = A_1
        all_Z_1[:, t, :] = Z_1

        # Update hidden state for next time step
        h_prev = Z_1

    # Output layer using last hidden state
    A_out = np.dot(h_prev, W_out.T) + B_out.T  # [batch_size, output_neurons]
    predictions = softmax(A_out)
    
    if passActivations:
        return predictions, all_Z_1, all_A_1, A_out, all_x_h
    return predictions

def forward_pass_one_input(X, W_1, W_out, B_1, B_out, passActivations=False):
    """
    Vectorized forward pass for a single sequence.
    X: [seq_len] array of token indices
    """
    # Reshape to batch of size 1 and use the batch version
    X_batch = X.reshape(1, -1)
    result = forward_pass_input_vector(X_batch, W_1, W_out, B_1, B_out, passActivations)

    if passActivations:
        predictions, all_Z_1, all_A_1, A_out, all_x_h = result
        return predictions[0], all_Z_1[0], all_A_1[0], A_out[0], all_x_h[0]
    return result[0]
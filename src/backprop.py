import numpy as np 

try:
    import tqdm
except ImportError:
    tqdm = None

from src.activation import tanh_derivative
from src.forward_pass import forward_pass_input_vector

class Backprop:
    def __init__(self, W_1, W_out, B_1, B_out, learning_rate=0.0009):
        self.W_1 = W_1
        self.W_out = W_out
        self.B_1 = B_1
        self.B_out = B_out
        self.learning_rate = learning_rate

    def calculate_gradients(self, X, Y_true, passPredictions=False):
        """
        X: [batch_size, seq_len] array of token indices (padded)
        Y_true: [batch_size, output_neurons] one-hot encoded labels
        """
        predictions, all_Z_1, all_A_1, _, all_x_h = forward_pass_input_vector(
            X, self.W_1, self.W_out, self.B_1, self.B_out, passActivations=True
        )

        batch_size, seq_len, neurons_hidden = all_Z_1.shape
        output_neurons = self.W_out.shape[0]

        # Output layer gradients
        delta = predictions - Y_true  # [batch_size, output_neurons]
        dW_out = delta.T @ all_Z_1[:, -1, :] / batch_size  # [output_neurons, neurons_hidden]
        dB_out = delta.sum(axis=0, keepdims=True).T / batch_size  # [output_neurons, 1]

        # Hidden layer gradients
        dW_1 = np.zeros_like(self.W_1)  # [neurons_hidden, 2]
        dB_1 = np.zeros_like(self.B_1)  # [neurons_hidden, 1]

        # Initial hidden delta for all sequences
        delta_hidden = np.einsum('bi,ij->bj', delta, self.W_out) * tanh_derivative(all_A_1[:, -1, :])  # [batch_size, neurons_hidden]

        tanh_derivs = tanh_derivative(all_A_1)  # [batch_size, seq_len, neurons_hidden]
        recurrent_weights = self.W_1[:, 1:]  # [neurons_hidden, neurons_hidden]

        # Process time steps in reverse
        for t in reversed(range(seq_len)):
            x_h_t = all_x_h[:, t, :]  # [batch_size, 1 + neurons_hidden]

            # Accumulate gradients (vectorized across batch)
            dW_1 += np.einsum('bi,bj->ij', delta_hidden, x_h_t)  # [neurons_hidden, 1 + neurons_hidden]
            dB_1 += delta_hidden.sum(axis=0, keepdims=True).T  # [neurons_hidden, 1]

            if t > 0:
                delta_hidden = (delta_hidden @ recurrent_weights) * tanh_derivs[:, t - 1, :]

        # Average gradients over batch
        dW_1 /= batch_size
        dB_1 /= batch_size

        if passPredictions:
            return dW_1, dB_1, dW_out, dB_out, predictions
        return dW_1, dB_1, dW_out, dB_out
    
    # def calculate_gradients(self, X, Y_true, passPredictions=False):
    #     predictions, all_Z_1, all_A_1, _, all_x_h = forward_pass_input_vector(X, self.W_1, self.W_out, self.B_1, self.B_out, passActivations=True)
        
    #     batch_size = len(X)
    #     seq_len = len(all_Z_1[0])

    #     delta = predictions - Y_true  # shape: (batch_size, output_neurons)
    #     dW_out = np.dot(delta.T, all_Z_1[:, -1]) / batch_size
    #     dB_out = np.sum(delta, axis=0, keepdims=True).T / batch_size

    #     dW_1 = np.zeros_like(self.W_1)
    #     dB_1 = np.zeros_like(self.B_1)
        
    #     for seq_idx in range(batch_size):
    #         seq_Z = all_Z_1[seq_idx]  # shape: (seq_len, neurons_hidden)
    #         seq_x_h = all_x_h[seq_idx]  # shape: (seq_len, 1 + neurons_hidden)
    #         seq_A = all_A_1[seq_idx]  # shape: (seq_len, neurons_hidden)

    #         seq_delta = delta[seq_idx]
    #         seq_delta_hidden = np.dot(seq_delta.T, self.W_out) * tanh_derivative(seq_A[-1])

    #         for t in reversed(range(seq_len)):
    #             dW_1 += np.dot(seq_delta_hidden.reshape(-1,1), seq_x_h[t].reshape(1, -1))
    #             dB_1 += seq_delta_hidden.reshape(-1, 1)

    #             if t > 0:
    #                 # Jacobian: ∂h_t / ∂h_{t-1} = diag(1 - h_t^2) * W_h
    #                 # W_h is the second column of W_1 (weights for h_{t-1})
    #                 jacobian = np.diag(1 - seq_Z[t].flatten()**2) @ self.W_1[:, 1:].T
    #                 seq_delta_hidden = seq_delta_hidden @ jacobian

    #     dW_1 /= batch_size
    #     dB_1 /= batch_size

    #     if passPredictions:
    #         return dW_1, dB_1, dW_out, dB_out, predictions
    #     return dW_1, dB_1, dW_out, dB_out

    def update_parameters(self, dW_1, dB_1, dW_out, dB_out, learning_rate):
        # Clip gradients to avoid exploding values
        dW_1 = np.clip(dW_1, -1.0, 1.0)
        dB_1 = np.clip(dB_1, -1.0, 1.0)
        dW_out = np.clip(dW_out, -1.0, 1.0)
        dB_out = np.clip(dB_out, -1.0, 1.0)


        self.W_1 -= learning_rate * dW_1
        self.B_1 -= learning_rate * dB_1
        self.W_out -= learning_rate * dW_out
        self.B_out -= learning_rate * dB_out

    def training_step(self, X, Y_true, passPredictions=False):
        dW_1, dB_1, dW_out, dB_out, predictions = self.calculate_gradients(X, Y_true, passPredictions=True)
        self.update_parameters(dW_1, dB_1, dW_out, dB_out, self.learning_rate)
        if passPredictions: return predictions


    def train(self, X_train, Y_train, epochs, batch_size=256, shuffle=True):
        loss_history = []
        progress = tqdm.tqdm(range(epochs), desc="Training") if tqdm else range(epochs)
        for epoch in progress:
            if shuffle:
                indices = np.random.permutation(len(X_train))
                X_epoch = X_train[indices]
                Y_epoch = Y_train[indices]
            else:
                X_epoch = X_train
                Y_epoch = Y_train

            epoch_losses = []
            for start in range(0, len(X_epoch), batch_size):
                end = start + batch_size
                X_batch = X_epoch[start:end]
                Y_batch = Y_epoch[start:end]
                predictions = self.training_step(X_batch, Y_batch, passPredictions=True)
                epoch_losses.append(-np.mean(Y_batch * np.log(predictions + 1e-18)))

            loss = np.mean(epoch_losses)
            loss_history.append(loss)
            if tqdm:
                progress.set_postfix(loss=f"{loss:.4f}")
        return loss_history


    
    def save_state(self):
        np.save("params/W_1", self.W_1)
        np.save("params/W_out", self.W_out)
        np.save("params/B_1", self.B_1)
        np.save("params/B_out", self.B_out)
    
    def load_state(self):
        self.W_1 = np.load("params/W_1.npy")
        self.W_out = np.load("params/W_out.npy")
        self.B_1 = np.load("params/B_1.npy")
        self.B_out = np.load("params/B_out.npy")
    
    def get_parameters(self):
        return self.W_1, self.W_out, self.B_1, self.B_out
    
    def jacobian_hidden_to_hidden(Z_t, W_1):
        """
        Computes the Jacobian matrix for the transition from hidden state Z_{t-1} to Z_t.

        :param Z_t: Post-activation hidden state at time step t (shape: [1, neurons_hidden])
        :param W_1: Weight matrix for the hidden layer (shape: [neurons_hidden, 2])
        :return: Jacobian matrix (shape: [neurons_hidden, neurons_hidden])
        """
        diag_matrix = np.diag(1 - Z_t.flatten()**2)
        jacobian = np.dot(diag_matrix, W_1.T)
        return jacobian

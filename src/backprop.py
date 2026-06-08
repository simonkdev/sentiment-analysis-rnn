import numpy as np 
import pandas as pd
import tqdm as tqdm

from src.activation import tanh_derivative
from src.forward_pass import forward_pass_input_vector

class Backprop:
    def __init__(self, W_1, W_out, B_1, B_out, learning_rate=0.01):
        self.W_1 = W_1
        self.W_out = W_out
        self.B_1 = B_1
        self.B_out = B_out
        self.learning_rate = learning_rate
    
    def calculate_gradients(self, X, Y_true, passPredictions=False):
        predictions, hidden_outputs, hidden_pre_activation, output_pre_activation, x_h_last = forward_pass_input_vector(X, self.W_1, self.W_out, self.B_1, self.B_out, passActivations=True)
        
        delta = predictions - Y_true
        batch_size = len(X)

        dW_out = np.dot(delta.T, hidden_outputs) / batch_size
        dB_out = np.sum(delta, axis=0, keepdims=True).T / batch_size

        dW_1 = np.zeros_like(self.W_1)
        dB_1 = np.zeros_like(self.B_1)
        
        delta_hidden = np.dot(delta, self.W_out) * tanh_derivative(hidden_pre_activation)

        for t in reversed(range(len(X))):
            dW_1 += np.dot(delta_hidden.T, x_h_last[t] / batch_size)
            dB_1 += np.sum(delta_hidden, axis=0, keepdims=True).T / batch_size

            if t > 0:
                jacobian = np.diag(1 - hidden_outputs[t]**2) @ self.W_1[:, 1:].T
                delta_hidden = delta_hidden @ jacobian

        if passPredictions:
            return dW_1, dB_1, dW_out, dB_out, predictions
        return dW_1, dB_1, dW_out, dB_out

    def update_parameters(self, dW_1, dB_1, dW_out, dB_out, learning_rate):
        self.W_1 -= learning_rate * dW_1
        self.B_1 -= learning_rate * dB_1
        self.W_out -= learning_rate * dW_out
        self.B_out -= learning_rate * dB_out

    def training_step(self, X, Y_true, passPredictions=False):
        dW_1, dB_1, dW_out, dB_out, predictions = self.calculate_gradients(X, Y_true, passPredictions=True)
        self.update_parameters(dW_1, dB_1, dW_out, dB_out, self.learning_rate)
        if passPredictions: return predictions

    def train(self, X_train, Y_train, epochs):
        loss_history = []
        for epoch in tqdm.tqdm(range(epochs)):
            predictions = self.training_step(X_train, Y_train, passPredictions=True)
            loss = -np.mean(Y_train * np.log(predictions + 1e-18))
            loss_history.append(loss)
            print(f"Epoch {epoch}, Loss: {loss}")
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
        diag_matrix = np.diag(1 - Z_t**2)
        jacobian = np.dot(diag_matrix, W_1.T)
        return jacobian
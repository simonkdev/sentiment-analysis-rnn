import numpy as np 
import pandas as pd
import tqdm as tqdm

from src.activation import tanh_derivative
from src.forward_pass import forward_pass_input_vector

class Backprop:
    def __init__(self, W_1, W_out, B_1, B_out, learning_rate=0.02):
        self.W_1 = W_1
        self.W_out = W_out
        self.B_1 = B_1
        self.B_out = B_out
        self.learning_rate = learning_rate
    
    def calculate_gradients(self, X, Y_true, passPredictions=False):
        predictions, hidden_outputs, hidden_pre_activation, output_pre_activation, x_h_last = forward_pass_input_vector(X, self.W_1, self.W_out, self.B_1, self.B_out, passActivations=True)
        
        delta = predictions - Y_true
        
        dW_out = np.dot(delta.T, hidden_outputs)
        dB_out = np.sum(delta, axis=0, keepdims=True).T

        delta_hidden = np.dot(delta, self.W_out) * tanh_derivative(hidden_pre_activation.reshape(-1, 5))

        dW_1 = delta_hidden.T.dot(x_h_last)
        dB_1 = np.sum(delta_hidden, axis=0, keepdims=True).T

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
            if epoch % 10 == 0:
                loss = np.mean((predictions - Y_train) ** 2)
                loss_history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss}")
        return loss_history
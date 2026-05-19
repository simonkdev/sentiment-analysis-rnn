import numpy as np 
import pandas as pd
from src.activation import relu, softmax, relu_derivative, softmax_derivative
from src.forward_pass import forward_pass_input_vector

class Backprop:
    def __init__(self, W_1, W_out, B_1, B_out):
        self.W_1 = W_1
        # self.W_2 = W_2
        # self.W_3 = W_3
        # self.W_4 = W_4
        self.W_out = W_out
        self.B_1 = B_1
        # self.B_2 = B_2
        # self.B_3 = B_3
        # self.B_4 = B_4
        self.B_out = B_out
    
    def calculate_gradients(self, X, Y_true):
        predictions, activation1, activation5, x_h_last = forward_pass_input_vector(X, self.W_1, self.W_out, self.B_1, self.B_out, passActivations=True)
        
        delta = predictions - Y_true
        
        dW_out = np.dot(activation5.T, delta)
        dB_out = np.sum(delta, axis=0, keepdims=True)

        delta_hidden = np.dot(delta, self.W_out) * relu_derivative(activation1.T)
        # dW_4 = dW_out.dot(self.W_out)
        # dB_4 = dW_4 * relu_derivative(activation4)

        # dW_3 = dW_4.dot(self.W_4)
        # dB_3 = dW_3 * relu_derivative(activation3)

        # dW_2 = dW_3.dot(self.W_3)
        # dB_2 = dW_2 * relu_derivative(activation2)

        dW_1 = delta_hidden.T.dot(x_h_last)
        dB_1 = np.sum(delta_hidden, axis=0, keepdims=True).T

        return dW_1, dB_1, dW_out, dB_out


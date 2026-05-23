from src.forward_pass import forward_pass_one_input
from src.initialization import initialize_weights, initialize_biases
from src.data_prep import load_data, process_new_data
from src.backprop import Backprop

import numpy as np

NEURONS_LAYER_HIDDEN = 5
OUTPUT_NEURONS = 2

class RNN:
    def __init__(self, dataPath = "data/IMDB Dataset.csv"):
        self.W_1, self.W_out = initialize_weights(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)
        self.B_1, self.B_out = initialize_biases(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)
        self.backprop = Backprop(self.W_1, self.W_out, self.B_1, self.B_out)
        self.sequences, self.labels, self.max_token, self.tokenizer = load_data(dataPath)

    def classify_sentiment(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        sequence = [t / self.max_token for t in sequence]  # Normalize
        sequence = np.array(sequence).flatten()  # Ensure 1D
        label = forward_pass_one_input(sequence, self.W_1, self.W_out, self.B_1, self.B_out)
        if np.argmax(label) == 1:
            return "negative"
        return "positive"
        
    def train(self, iterations):
        self.backprop.train(self.sequences, self.labels, iterations)
        self.W_1, self.W_out, self.B_1, self.B_out = self.backprop.get_parameters()
    
    def load_trained_state(self):
        self.backprop.load_state()
        self.W_1, self.W_out, self.B_1, self.B_out = self.backprop.get_parameters()




# TODO: Restructure this file into the RNN class and provide clean access for the api file. 
# Should also be able to load parameters from a file and assign them to its own.


# OTHER TODOS: 
# 1. Implement state saving (saving weights to .npy files after training)   done
# 2. Implement state loading from path  done 
# 3. Finish frontend
# 4. Documentation
# 5. clean up code, add docstrings where missing, etc.


# W_1, W_out = initialize_weights(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)
# B_1, B_out = initialize_biases(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)

# print("[ INIT ] Loading data...")

# print(sequences[0])




# print(forward_pass_one_input(sequences[0], W_1, W_out, B_1, B_out))
# print(classify_sentiment("Ich bin ein Berliner", max_token, tokenizer))

# print("[ INIT ] Initializing backpropagation...")
# backprop = Backprop(W_1, W_out, B_1, B_out)
# print("[ INIT ] Calculating gradients for the first training example...")
# backprop.save_state()
# print(backprop.train(sequences, labels, 100))
# backprop.save_state()

# def forward_pass(X):
#     if X == "Hello There":
#         return "General Kenobi!"
#     return False;
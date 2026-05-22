from src.forward_pass import forward_pass_one_input
from src.initialization import initialize_weights, initialize_biases
from src.data_prep import load_data, process_new_data
from src.backprop import Backprop

import numpy as np

NEURONS_LAYER_HIDDEN = 5
OUTPUT_NEURONS = 2

W_1, W_out = initialize_weights(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)
B_1, B_out = initialize_biases(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)

print("[ INIT ] Loading data...")
sequences, labels, max_token, tokenizer = load_data("data/IMDB Dataset.csv")

print(sequences[0])


def classify_sentiment(text, max_token, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = [t / max_token for t in sequence]  # Normalize
    sequence = np.array(sequence).flatten()  # Ensure 1D
    label = forward_pass_one_input(sequence, W_1, W_out, B_1, B_out)
    return np.argmax(label)


print(forward_pass_one_input(sequences[0], W_1, W_out, B_1, B_out))
print(classify_sentiment("Ich bin ein Berliner", max_token, tokenizer))

print("[ INIT ] Initializing backpropagation...")
backprop = Backprop(W_1, W_out, B_1, B_out)
print("[ INIT ] Calculating gradients for the first training example...")
print(backprop.train(sequences, labels, 100))


def forward_pass(X):
    if X == "Hello There":
        return "General Kenobi!"
    return False;
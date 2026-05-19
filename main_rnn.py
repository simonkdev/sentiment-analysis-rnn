from src.forward_pass import forward_pass_one_input
from src.initialization import initialize_weights, initialize_biases
from src.data_prep import load_data
from src.backprop import Backprop

NEURONS_LAYER_HIDDEN = 5
OUTPUT_NEURONS = 1

W_1, W_out = initialize_weights(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)
B_1, B_out = initialize_biases(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)

print("[ INIT ] Loading data...")
sequences, labels = load_data("data/IMDB Dataset.csv")

print(sequences[0])

print(forward_pass_one_input(sequences[0], W_1, W_out, B_1, B_out))

print("[ INIT ] Initializing backpropagation...")
backprop = Backprop(W_1, W_out, B_1, B_out)
print("[ INIT ] Calculating gradients for the first training example...")
print(backprop.calculate_gradients(sequences, labels))

def forward_pass(X):
    if X == "Hello There":
        return "General Kenobi!"
    return False;
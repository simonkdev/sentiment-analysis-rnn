from src.forward_pass import forward_pass_one_input
from src.initialization import initialize_weights, initialize_biases
from src.data_prep import load_data

NEURONS_LAYER_HIDDEN = 5
OUTPUT_NEURONS = 2

W_1, W_2, W_3, W_4, W_out = initialize_weights(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)
B_1, B_2, B_3, B_4, B_out = initialize_biases(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)

sequences, labels = load_data("data/IMDB Dataset.csv")

print(sequences[0])

print(forward_pass_one_input(sequences[0], W_1, W_2, W_3, W_4, W_out, B_1, B_2, B_3, B_4, B_out))

def forward_pass(X):
    if X == "Hello There":
        return "General Kenobi!"
    return False;
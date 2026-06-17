from src.forward_pass import forward_pass_one_input, forward_pass_input_vector
from src.initialization import initialize_weights, initialize_biases
from src.data_prep import load_data, load_embedding_layer, load_tokenizer, process_new_data
from src.backprop import Backprop

import numpy as np

NEURONS_LAYER_HIDDEN = 8
OUTPUT_NEURONS = 2

class RNN:
    def __init__(self, dataPath="data/IMDB Dataset.csv", load_dataset=False):
        self.sequences = None
        self.labels = None
        self.test_seq = None
        self.test_lab = None

        if load_dataset:
            self.sequences, self.labels, self.max_token, self.tokenizer, self.test_seq, self.test_lab, self.embedding_layer = load_data(dataPath)
        else:
            self.tokenizer = load_tokenizer()
            self.max_token = max(self.tokenizer.word_index.values(), default=1)
            self.embedding_layer = load_embedding_layer()

        self.W_1, self.W_out = initialize_weights(
            NEURONS_LAYER_HIDDEN,
            OUTPUT_NEURONS,
            input_features=self.embedding_layer.embedding_dim,
        )
        self.B_1, self.B_out = initialize_biases(NEURONS_LAYER_HIDDEN, OUTPUT_NEURONS)
        self.backprop = Backprop(self.W_1, self.W_out, self.B_1, self.B_out)

    def classify_sentiment(self, text):
        label = self.predict_sentiment_scores(text)
        if np.argmax(label) == 1:
            return "positive"
        return "negative"

    def predict_sentiment_scores(self, text):
        sequence = process_new_data(text, self.max_token, self.tokenizer, self.embedding_layer)
        return forward_pass_one_input(sequence, self.W_1, self.W_out, self.B_1, self.B_out)

    def train(self, iterations, batch_size=256):
        if self.sequences is None or self.labels is None:
            raise RuntimeError("Training requires RNN(load_dataset=True).")
        self.backprop.train(
            self.sequences,
            self.labels,
            iterations,
            batch_size=batch_size,
            embedding_layer=self.embedding_layer,
        )
        self.W_1, self.W_out, self.B_1, self.B_out = self.backprop.get_parameters()

    def load_trained_state(self):
        self.backprop.load_state()
        self.W_1, self.W_out, self.B_1, self.B_out = self.backprop.get_parameters()
        expected_input_width = self.embedding_layer.embedding_dim + NEURONS_LAYER_HIDDEN
        if self.W_1.shape != (NEURONS_LAYER_HIDDEN, expected_input_width):
            raise ValueError(
                "Saved RNN weights do not match the current embedding shape. "
                "Retrain with `python train.py` so params/*.npy and params/embedding_matrix.npy stay in sync."
            )

    def save_trained_state(self):
        self.tokenizer.save_state()
        self.embedding_layer.save_state()
        self.backprop.save_state()

    def calculate_accuracy(self):
        if self.test_seq is None or self.test_lab is None:
            raise RuntimeError("Accuracy calculation requires RNN(load_dataset=True).")
        predictions = forward_pass_input_vector(
            self.embedding_layer(self.test_seq),
            self.W_1,
            self.W_out,
            self.B_1,
            self.B_out,
        )
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.test_lab, axis=1)
        correct_count = np.sum(pred_classes == true_classes)
        accuracy = correct_count / len(self.test_seq) * 100
        print(f"ACCURACY IS APPROXIMATELY {accuracy}%")
        return accuracy



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
#     return False

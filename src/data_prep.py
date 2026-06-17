import numpy as np
import csv
import json
from pathlib import Path

max_len = 150
DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "params" / "tokenizer.json"
DEFAULT_EMBEDDING_PATH = Path(__file__).resolve().parent.parent / "params" / "embedding_matrix.npy"

class SimpleTokenizer:
    def __init__(self, word_index=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "):
        self.word_index = word_index or {}
        self.filters = filters
        self.lower = lower
        self.split = split

    @property
    def _translation(self):
        return str.maketrans({char: " " for char in self.filters})

    def fit_on_texts(self, texts):
        counts = {}
        first_seen = {}
        position = 0

        for text in texts:
            for token in self._tokenize(text):
                counts[token] = counts.get(token, 0) + 1
                if token not in first_seen:
                    first_seen[token] = position
                    position += 1

        ordered_tokens = sorted(counts, key=lambda token: (-counts[token], first_seen[token]))
        self.word_index = {token: index for index, token in enumerate(ordered_tokens, start=1)}

    def texts_to_sequences(self, texts):
        return [
            [self.word_index[token] for token in self._tokenize(text) if token in self.word_index]
            for text in texts
        ]

    def _tokenize(self, text):
        if self.lower:
            text = text.lower()
        return [token for token in text.translate(self._translation).split(self.split) if token]

    def save_state(self, path=DEFAULT_TOKENIZER_PATH):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "filters": self.filters,
                    "lower": self.lower,
                    "split": self.split,
                    "word_index": self.word_index,
                },
                file,
                separators=(",", ":"),
            )

class EmbeddingLayer:
    def __init__(self, vocab_size=None, embedding_dim=50, embedding_matrix=None):
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
            self.vocab_size = embedding_matrix.shape[0] - 1
            self.embedding_dim = embedding_matrix.shape[1]
            return

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(vocab_size + 1, embedding_dim) * 0.01

    def __call__(self, sequences):
        embedded_sequences = np.zeros((len(sequences), max_len, self.embedding_dim))
        for i, seq in enumerate(sequences):
            for j, token in enumerate(seq):
                if token != 0 and token < len(self.embedding_matrix):
                    embedded_sequences[i, j] = self.embedding_matrix[token]
        return embedded_sequences

    def update_parameters(self, token_sequences, embedding_gradients, learning_rate):
        token_gradients = {}
        for sequence, sequence_gradients in zip(token_sequences, embedding_gradients):
            for token, gradient in zip(sequence, sequence_gradients):
                if token == 0 or token >= len(self.embedding_matrix):
                    continue
                if token not in token_gradients:
                    token_gradients[token] = np.zeros(self.embedding_dim)
                token_gradients[token] += gradient

        for token, gradient in token_gradients.items():
            self.embedding_matrix[token] -= learning_rate * np.clip(gradient, -1.0, 1.0)

    def save_state(self, path=DEFAULT_EMBEDDING_PATH):
        np.save(path, self.embedding_matrix)

def load_tokenizer(path=DEFAULT_TOKENIZER_PATH):
    with open(path, encoding="utf-8") as file:
        tokenizer_data = json.load(file)

    return SimpleTokenizer(
        word_index=tokenizer_data["word_index"],
        filters=tokenizer_data.get("filters", SimpleTokenizer().filters),
        lower=tokenizer_data.get("lower", True),
        split=tokenizer_data.get("split", " "),
    )

def load_embedding_layer(path=DEFAULT_EMBEDDING_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding matrix not found at {path}. Run `python train.py` to create params/embedding_matrix.npy."
        )
    embedding_matrix = np.load(path)
    return EmbeddingLayer(embedding_matrix=embedding_matrix)

def pad_sequences(sequences, maxlen, padding="pre", truncating="pre"):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=int)

    for index, sequence in enumerate(sequences):
        truncated = np.asarray(sequence[:maxlen] if truncating == "post" else sequence[-maxlen:], dtype=int)
        if padding == "post":
            padded_sequences[index, : len(truncated)] = truncated
        else:
            padded_sequences[index, -len(truncated) :] = truncated

    return padded_sequences

def data_tokenization(dataframe):
    print("[ INIT ] Tokenizing text data...")
    tokenizer = SimpleTokenizer()
    print("[ INIT ] Fitting tokenizer on text data...")
    tokenizer.fit_on_texts([row["text"] for row in dataframe])
    print("[ INIT ] Text data tokenized.")
    sequences = tokenizer.texts_to_sequences([row["text"] for row in dataframe])

    max_token = max(max(seq) for seq in sequences if seq) if sequences else 1
    vocab_size = max_token
    embedding_layer = EmbeddingLayer(vocab_size)

    sequences = pad_sequences(sequences, maxlen=max_len)

    return sequences, max_token, tokenizer, embedding_layer

def load_data(file_path):
    print(f"[ INIT ] Loading data from {file_path}...")
    with open(file_path, newline="", encoding="utf-8") as file:
        dataframe = list(csv.DictReader(file))

    sequences, max_token, tokenizer, embedding_layer = data_tokenization(dataframe)
    label_index = {"negative": 0, "positive": 1}
    labels = np.array(
        [[1 if index == label_index[row["label"]] else 0 for index in range(len(label_index))] for row in dataframe]
    )
    print(f"[ INIT ] Data loaded from {file_path}.")
    return sequences[5000:], labels[5000:], max_token, tokenizer, sequences[:5000], labels[:5000:], embedding_layer

def process_new_data(string, max_token, tokenizer, embedding_layer):
    sequence = tokenizer.texts_to_sequences([string])[0]
    sequence = pad_sequences([sequence], maxlen=max_len)[0]
    embedded = embedding_layer([sequence])
    return embedded[0]

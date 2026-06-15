import numpy as np
import csv
import json
from pathlib import Path

max_len = 150
DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "params" / "tokenizer.json"


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


def load_tokenizer(path=DEFAULT_TOKENIZER_PATH):
    with open(path, encoding="utf-8") as file:
        tokenizer_data = json.load(file)

    return SimpleTokenizer(
        word_index=tokenizer_data["word_index"],
        filters=tokenizer_data.get("filters", SimpleTokenizer().filters),
        lower=tokenizer_data.get("lower", True),
        split=tokenizer_data.get("split", " "),
    )


def pad_sequences(sequences, maxlen, padding="pre", truncating="pre"):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=float)

    for index, sequence in enumerate(sequences):
        truncated = np.asarray(sequence[:maxlen] if truncating == "post" else sequence[-maxlen:], dtype=float)
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
    sequences = [[token / max_token for token in sequence] for sequence in sequences]

    sequences = pad_sequences(sequences, maxlen=max_len)
    
    return sequences, max_token, tokenizer

def load_data(file_path):
    print(f"[ INIT ] Loading data from {file_path}...")
    with open(file_path, newline="", encoding="utf-8") as file:
        dataframe = list(csv.DictReader(file))

    sequences, max_token, tokenizer = data_tokenization(dataframe)
    label_index = {"negative": 0, "positive": 1}
    labels = np.array(
        [[1 if index == label_index[row["label"]] else 0 for index in range(len(label_index))] for row in dataframe]
    )
    print(f"[ INIT ] Data loaded from {file_path}.")
    return sequences[5000:], labels[5000:], max_token, tokenizer, sequences[:5000], labels[:5000:]

def process_new_data(string, max_token, tokenizer):
    sequence = tokenizer.texts_to_sequences([string])[0]
    sequence = [token / max_token for token in sequence]
    return pad_sequences([sequence], maxlen=max_len)[0]

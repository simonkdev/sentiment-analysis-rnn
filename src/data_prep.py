import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def data_tokenization(dataframe):
    print("[ INIT ] Tokenizing text data...")
    tokenizer = Tokenizer()
    print("[ INIT ] Fitting tokenizer on text data...")
    tokenizer.fit_on_texts(dataframe['text'])
    print("[ INIT ] Text data tokenized.")
    sequences = tokenizer.texts_to_sequences(dataframe['text'])

    max_token = max(max(seq) for seq in sequences) if sequences else 1
    sequences = [[t / max_token for t in seq] for seq in sequences]
    
    return sequences, max_token, tokenizer

def load_data(file_path):
    print(f"[ INIT ] Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    sequences, max_token, tokenizer = data_tokenization(df)
    labels = pd.get_dummies(df['label'], dtype=int).values
    print(f"[ INIT ] Data loaded from {file_path}.")
    print(labels)
    return sequences[5000:], labels[5000:], max_token, tokenizer, sequences[:5000], labels[:5000:]

def process_new_data(string, max_token, tokenizer):
    sequence = np.array(tokenizer.texts_to_sequences([string])[0])
    sequence = [t / max_token for t in sequence]
    return np.array(sequence).flatten()


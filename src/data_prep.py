# The goal is to load the csv containing one column of text and one label. 
# The label column should be the basis to a n * 1 matrix of either 0 or 1, being the number of training examples.

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
    return sequences

def load_data(file_path):
    print(f"[ INIT ] Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    sequences = data_tokenization(df)
    labels = pd.get_dummies(df['label']).values
    print(f"[ INIT ] Data loaded from {file_path}.")
    return sequences, labels


# The goal is to load the csv containing one column of text and one label. 
# The label column should be the basis to a n * 1 matrix of either 0 or 1, being the number of training examples.

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

def data_tokenization(dataframe):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataframe['text'])
    sequences = tokenizer.texts_to_sequences(dataframe['text'])
    return sequences

def load_data(file_path):
    df = pd.read_csv(file_path)
    sequences = data_tokenization(df)
    labels = pd.get_dummies(df['label']).values
    return sequences, labels


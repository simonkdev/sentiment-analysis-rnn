# The goal is to load the csv containing one column of text and one label. 
# The label column should be the basis to a n * 1 matrix of either 0 or 1, being the number of training examples.

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def data_tokenization(dataframe):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataframe['text'])
    sequences = tokenizer.texts_to_sequences(dataframe['text'])
    return sequences


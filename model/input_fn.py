"""Create the input data pipeline using `tf.data`"""

import numpy as np

from typing import List

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def encode_sentences(sentences: List[str], tokenizer: Tokenizer, data_params: dict) -> np.ndarray:
    encoded = tokenizer.texts_to_sequences(sentences)
    max_len = data_params['max_sentence_length']
    return pad_sequences(encoded, maxlen=max_len, padding='post')


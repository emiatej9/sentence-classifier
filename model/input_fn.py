"""Create the input data pipeline using `tf.data`"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def input_fn(sentences, data_params):
    num_words = data_params['vocab_size']
    tokenizer = Tokenizer(num_words=num_words, split=' ', oov_token='<unk>')
    tokenizer.fit_on_texts(sentences)
    encoded = tokenizer.texts_to_sequences(sentences)

    max_len = data_params['max_sentence_length']
    return pad_sequences(encoded, maxlen=max_len, padding='post')


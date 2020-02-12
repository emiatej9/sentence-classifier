"""Train the model"""

import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from model.input_fn import encode_sentences
from model.model_fn import model_fn


if __name__ == '__main__':
    dataset_fields = ['sentences', 'labels']
    dataset_splits = ['train', 'dev', 'test']

    dataset = dict()
    for split in dataset_splits[:3]:
        dataset[split] = dict()
        for field in dataset_fields:
            with open(f'./data/nsmc/{split}/{field}.txt') as f:
                lines = f.readlines()
                dataset[split][field] = tuple([line.strip() for line in lines])

    with open('./params/dataset_params.json') as f:
        data_params = json.load(f)

    with open('./params/model_params.json') as f:
        model_params = json.load(f)

    with open('./params/training_params.json') as f:
        training_params = json.load(f)

    model = model_fn(data_params, model_params)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    num_words = data_params['vocab_size']
    tokenizer = Tokenizer(num_words=num_words, split=' ', oov_token='<unk>')
    tokenizer.fit_on_texts(dataset['train']['sentences'])

    training_sentences = encode_sentences(dataset['train']['sentences'], tokenizer, data_params)
    training_labels = np.asarray([int(label) for label in dataset['train']['labels']])
    dev_sentences = encode_sentences(dataset['dev']['sentences'], tokenizer, data_params)
    dev_labels = np.asarray([int(label) for label in dataset['dev']['labels']])

    print(training_sentences.shape, training_sentences[1])
    print(training_labels.shape, training_labels[1])
    print(type(dev_sentences), dev_sentences.dtype)

    batch_size = training_params['batch_size']
    epochs = training_params['epochs']

    model.fit(training_sentences[:100], training_labels[:100],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(dev_sentences[:30], dev_labels[:30]))



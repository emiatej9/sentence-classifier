"""Train the model"""

import json
import numpy as np
import tensorflow as tf

from model.input_fn import input_fn
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

    training_sentences = input_fn(dataset['train']['sentences'], data_params)
    training_labels = np.asarray([int(label) for label in dataset['train']['labels']])
    dev_sentences = input_fn(dataset['dev']['sentences'], data_params)
    dev_labels = np.asarray([int(label) for label in dataset['dev']['labels']])

    print(training_sentences.shape, training_sentences[1])
    print(training_labels.shape, training_labels[1])
    print(type(dev_sentences), dev_sentences.dtype)

    batch_size = training_params['batch_size']
    epochs = training_params['epochs']

    model.fit(training_sentences, training_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(dev_sentences, dev_labels))



"""Train the model"""

import json
import numpy as np

from ray.tune.integration.keras import TuneReporterCallback

from model.input_fn import input_fn
from model.model_fn import model_fn


def trainable(config, reporter):
    import tensorflow as tf

    model = model_fn(data_params, model_params)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_sentences = input_fn(dataset['train']['sentences'], data_params)
    training_labels = np.asarray([int(label) for label in dataset['train']['labels']])
    dev_sentences = input_fn(dataset['dev']['sentences'], data_params)
    dev_labels = np.asarray([int(label) for label in dataset['dev']['labels']])

    batch_size = training_params['batch_size']
    epochs = training_params['epochs']

    for i in range(3):
        history = model.fit(training_sentences[:64], training_labels[:64],
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(dev_sentences[:32], dev_labels[:32]))

        tune.track.init()
        tune.track.log(mean_accuracy=history.history['accuracy'])


if __name__ == '__main__':
    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler

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

    ray.init()

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        max_t=400,
        grace_period=20)

    tune.run(
        trainable,
        name="exp",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 5,
        },
        num_samples=10,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },
        config={
            "threads": 2,
            "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1)),
            "momentum": tune.sample_from(
                lambda spec: np.random.uniform(0.1, 0.9)),
            "hidden": tune.sample_from(
                lambda spec: np.random.randint(32, 512)),
        })


    '''
    model.fit(training_sentences, training_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(dev_sentences, dev_labels))
    '''

    analysis = tune.run(
        trainable, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})



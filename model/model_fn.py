"""Define the model."""

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers, Model


def model_fn(data_params, model_params):
    input_length = data_params['max_sentence_length']
    input_tensor = keras.Input(shape=(input_length,), dtype='int32')

    # embedding
    embedding_dim = model_params['embedding_dim']
    feature_size = data_params['vocab_size']
    tensor_from_embedding = Embedding(input_dim=feature_size,
                                      output_dim=embedding_dim,
                                      input_length=input_length)(input_tensor)

    # reshape
    tensor_from_reshape = Reshape((input_length, embedding_dim, 1))(tensor_from_embedding)

    # convolution & max-pooling
    num_filters = model_params['num_filters']
    kernel_regularizer = regularizers.l2(model_params['regularization_factor'])
    base_kernel_size = model_params['base_kernel_size']

    tensor_from_conv1 = Conv2D(num_filters,
                               kernel_size=(base_kernel_size, embedding_dim),
                               activation='relu',
                               kernel_regularizer=kernel_regularizer)(tensor_from_reshape)

    tensor_from_conv2 = Conv2D(num_filters,
                               kernel_size=(base_kernel_size+1, embedding_dim),
                               activation='relu',
                               kernel_regularizer=kernel_regularizer)(tensor_from_reshape)

    tensor_from_conv3 = Conv2D(num_filters,
                               kernel_size=(base_kernel_size+2, embedding_dim),
                               activation='relu',
                               kernel_regularizer=kernel_regularizer)(tensor_from_reshape)

    base_pool_size = input_length - base_kernel_size + 1
    tensor_from_max_pooling1 = MaxPool2D(pool_size=(base_pool_size, 1),
                                         strides=(1, 1),
                                         padding='valid')(tensor_from_conv1)

    tensor_from_max_pooling2 = MaxPool2D(pool_size=(base_pool_size-1, 1),
                                         strides=(1, 1),
                                         padding='valid')(tensor_from_conv2)

    tensor_from_max_pooling3 = MaxPool2D(pool_size=(base_pool_size-2, 1),
                                         strides=(1, 1),
                                         padding='valid')(tensor_from_conv3)

    # remains
    tensor_from_concat = Concatenate(axis=1)([tensor_from_max_pooling1,
                                              tensor_from_max_pooling2,
                                              tensor_from_max_pooling3])

    tensor_from_flatten = Flatten()(tensor_from_concat)
    tensor_from_dropout = Dropout(0.5)(tensor_from_flatten)

    # output
    output_tensor = Dense(units=1, activation='sigmoid')(tensor_from_dropout)

    return Model(inputs=input_tensor, outputs=output_tensor)

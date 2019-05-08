#!/usr/bin/env python

from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
import tensorflow


CLASS_MAPPING = {
    'normal': 0,
    'dos': 1,
    'probe': 2,
    'r2l': 3,
    'u2r': 4
}

CLASS_MAPPING_REVERSE = {
    0: 'normal',
    1: 'dos',
    2: 'probe',
    3: 'r2l',
    4: 'u2r'
}

def parse_attack_types(filename):
    """
    Generate a mapping that looks like:

    {
        'teardrop.': 'dos',
        'smurf.': 'dos',
        ...
    }

    This will be used to further encode the training data because we want to reduce
    the labels to 0-4.
    """
    attack_map = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        attack, category = line.split()
        attack_map[attack + '.'] = category
    return attack_map


def encode_data(train_data, cols, attack_map=None, encodings=None):
    """
    Encode any strings in the training data so that they are integers.
    Also return the map of `encodings` and `decodings`.

    Plan of use:

    1. Pass the training matrix in here without providing 'encodings'.
       The caller should save the `encodings` and `decodings`.
    2. When you need to encode your test data, make sure to pass in
       the `encodings` generated from step 1 so that we encode the
       test data the same way.
    """
    if not encodings:
        encodings = {}
        decodings = {}
        for col in cols:
            unique_values = train_data[col].unique()
            encoding = {}
            decoding = {}  # Used for lookup later if we need it
            if col != 41:
                for i, attack in enumerate(unique_values):
                    encoding[attack] = i
                    decoding[i] = attack
                # Encode strings like ('tcp', 'udp', 'icmp') into (0, 1, 2)
                train_data[col] = train_data[col].map(encoding)
                encodings[col] = encoding
                decodings[col] = decoding
            else:
                # This is the label. We want to reduce our classes to be 0-4
                for attack in unique_values:
                    encoding[attack] = CLASS_MAPPING[attack_map.get(attack, 'normal')]
                # Encode strings like ('tcp', 'udp', 'icmp') into (0, 1, 2)
                train_data[col] = train_data[col].map(encoding)
                # The new encodings for the labels basically become something like:
                # {'normal': 0, 'dos': 1, 'u2r': 3, 'r2l': 3, 'probe': 4}
                encodings[col] = CLASS_MAPPING
                decodings[col] = CLASS_MAPPING_REVERSE
    else:
        decodings = None
        for col in cols:
            train_data[col] = train_data[col].map(encodings[col])
    return encodings, decodings


def parse_data(filename):
    return pd.read_csv(filename, header=None)


def neural_networks_train(train_X, train_y):
    # Create model
    model = Sequential()
    # Get number of columns in training data
    n_cols = train_X.shape[1]
    # Add model layers
    num_classes = len(CLASS_MAPPING)
    model.add(Dense(10, activation='softmax', input_shape=(n_cols,)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model using mse as a measure of model performance
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)
    # Train model
    train_y_binary = to_categorical(train_y, num_classes)
    model.fit(train_X, train_y_binary, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    return model


if __name__ == '__main__':
    print('Running project')
    attack_map = parse_attack_types('./dataset/attack_types.txt')
    print('Attack mapping:')
    print(attack_map)
    data = parse_data('./dataset/kddcup.data')
    encodings, decodings = encode_data(data, cols=(1, 2, 3, 41), attack_map=attack_map)
    x = data.drop(columns=[41])
    y = data[41]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    print('Encoded data:')
    print(x_train[:2])
    print('Encodings:')
    print(encodings)
    print('Decodings:')
    print(decodings)
    model = neural_networks_train(x_train, y_train)
    model.save('keras.model')
    predictions = model.predict_classes(x_test)
    print(predictions)
    for i in range(5):
        if i in predictions:
            print(f"{i} is in the prediction")
    #y_test_binary = to_categorical(y_test, 5)
    #model.evaluate(x_test, y_test_binary)

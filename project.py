#!/usr/bin/env python

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import scipy
import tensorflow


np.random.seed(255)

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


def neural_networks_train(train_data):
    train_X = train_data.drop(columns=[41])
    train_y = train_data[[41]]
    print("Neural networks train_X: ")
    print(train_X)
    print("Neural networks train_y: ")
    print(train_y)
    # Create model
    model = Sequential()
    # Get number of columns in training data
    n_cols = train_X.shape[1]
    # Add model layers
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    # Compile model using mse as a measure of model performance
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)
    # Train model
    model.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    return model


if __name__ == '__main__':
    """
    Whoever is working on their local branch should probably comment out other people's
    training since you don't want to run them. I will also comment my own when I check
    in my stuff.

    For faster development I usually keep running under the same Docker container. I do
    this by modifying the Dockerfile so that the ENTRYPOINT looks like:

    ENTRYPOINT ["sleep", "infinity"]

    This basically keeps the container up and running. You can go inside the container
    as if you were SSHing into a remote host and run the application manually. You can
    do this by running:

    docker ps   # Note down the CONTAINER ID
    docker exec -it <CONTAINER_ID> bash

    This will bring to a bash prompt inside the container. The container doesn't have
    many packages. I use vim for editing the source file so consider running:

    apt-get update
    apt-get install vim

    Now you can go to /opt/project and:

    1. Modify the source code
    2. Run it all inside the container
    """
    print('Running project')
    attack_map = parse_attack_types('./dataset/attack_types.txt')
    print('Attack mapping:')
    print(attack_map)
    train_data = parse_data('./dataset/kddcup.data_10_percent')
    print('Raw data:')
    print(train_data[:2])
    encodings, decodings = encode_data(train_data, cols=(1, 2, 3, 41), attack_map=attack_map)
    print('Encoded data:')
    print(train_data[:2])
    print('Encodings:')
    print(encodings)
    print('Decodings:')
    print(decodings)
    print("Looking for NaN in training data")
    print(train_data.isnull().values.any())
    model = neural_networks_train(train_data)
    model.save('keras.model')
    test_data = parse_data('./dataset/kddcup.testdata.unlabeled_10_percent')
    encode_data(test_data, cols=(1, 2, 3), attack_map=attack_map)
    print("Looking for NaN in testing data")
    print(test_data.isnull().values.any())
    results = model.fit(test_data)

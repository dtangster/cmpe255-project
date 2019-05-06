#!/usr/bin/env python

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
import numpy
import pandas as pd
import scipy
import tensorflow


# Not sure if we need this yet.
CATEGORY_ENCODING = {
    'normal': 0,
    'dos': 1,
    'probe': 2,
    'r2l': 3,
    'u2r': 4,
}


def parse_attack_types(filename):
    """
    Generate a mapping that looks like:

    {
        'teardrop': {
            'encoding': 0,
            'category': 'dos'
        },
        'smurf': {
            'encoding': 1,
            'category': 'dos'
        },
        ...
    }

    The 'encoding' becomes important in some learning algorithms. We have to encode text
    into numbers so some algorithms can process them.
    """
    attack_map = {}
    attack_encoding = {}
    count = 0
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        attack, category = line.split()
        if attack not in attack_map:
            attack_map[attack] = {
                'encoding': count,
                'category': category
            }
            count += 1
    return attack_map


def parse_data(filename):
    return pd.read_csv(filename, header=None)


def neural_networks_train(train_data):
    train_X = train_data.drop(columns=[41])
    train_y = train_data[[41]]

    t = Tokenizer()
    t.fit_on_texts(train_X)
    train_X = t.texts_to_matrix(train_X, mode='count')

    t2 = Tokenizer()
    t2.fit_on_texts(train_y)
    train_y = t2.texts_to_matrix(train_y, mode='count')

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


if __name__ == '__main__':
    print('Running project')
    attack_map = parse_attack_types('./dataset/attack_types.txt')
    print('Attack mapping:')
    print(attack_map)
    train_data = parse_data('./dataset/kddcup.data_10_percent')
    print(train_data[:2])
    #neural_networks_train(train_data)

#!/usr/bin/env python

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
import scipy
import tensorflow


def parse_attack_types(filename):
    attack_map = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        attack, category = line.split()
        if category in attack_map:
            attack_map[category].add(attack)
        else:
            attack_map[category] = {attack}
    return attack_map


def parse_data(filename):
    train_data = pd.read_csv(filename, header=None)
    train_X = train_data.drop(columns=[41])
    train_y = train_data[[41]]
    return train_data, train_X, train_y


def neural_networks_train(train_X, train_y):
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
    print(attack_map)
    # train_data = full training data
    # train_X    = full training data without labels
    # train_y    = labels only
    train_data, train_X, train_y = parse_data('./dataset/kddcup.data_10_percent')
    print(train_data[:2])
    #neural_networks_train(train_X, train_y)

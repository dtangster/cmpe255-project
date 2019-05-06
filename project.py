#!/usr/bin/env python

import keras
import numpy
import pandas
import scipy
import tensorflow


def parse_attack_types(filename):
    attack_map = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        attack, attack_type = line.split()
        if attack_type in attack_map:
            attack_map[attack_type].add(attack)
        else:
            attack_map[attack_type] = {attack}
    return attack_map


def parse_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = list(map(lambda x: x.strip().split(','), lines))
    return lines


if __name__ == '__main__':
    print('Running project')
    attack_map = parse_attack_types('./dataset/attack_types.txt')
    print(attack_map)
    lines = parse_data('./dataset/kddcup.data_10_percent')
    print(lines[:2])

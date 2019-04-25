#!/usr/bin/env python


def parse_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = list(map(lambda x: x.strip().split(','), lines))
    return lines


if __name__ == '__main__':
    print('Running project')
    lines = parse_data('/opt/project/dataset/kddcup.data_10_percent')
    print(lines[:2])

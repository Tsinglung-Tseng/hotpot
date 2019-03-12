import itertools


def to_iter(dataset):
    for i in itertools.count(0):
        yield dataset[i]



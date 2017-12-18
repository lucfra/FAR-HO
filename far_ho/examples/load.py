"""
Just MNIST.  Loaders for other datasets can be found at https://github.com/lucfra/ExperimentManager
"""

from far_ho.examples.datasets import Datasets, Dataset
from far_ho.examples.utils import redivide_data
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


def mnist(folder=None, one_hot=True, partitions=None, shuffle=False):
    """
    Loads (download if necessary) Mnist dataset, and optionally splits it to form different training, validation
    and test sets (use partitions parameters for that)

    :param folder:
    :param one_hot:
    :param partitions:
    :param shuffle:
    :return:
    """
    datasets = read_data_sets(folder, one_hot=one_hot)
    train = Dataset(datasets.train.images, datasets.train.labels, name='MNIST')
    validation = Dataset(datasets.validation.images, datasets.validation.labels, name='MNIST')
    test = Dataset(datasets.test.images, datasets.test.labels, name='MNIST')
    res = [train, validation, test]
    if partitions:
        res = redivide_data(res, partition_proportions=partitions, shuffle=shuffle)
        res += [None] * (3 - len(res))
    return Datasets.from_list(res)

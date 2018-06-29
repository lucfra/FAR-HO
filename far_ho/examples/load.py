"""
Just MNIST.  Loaders for other datasets can be found at https://github.com/lucfra/ExperimentManager
"""
from __future__ import absolute_import, print_function, division

from far_ho.examples.datasets import Datasets, Dataset
from far_ho.examples.utils import redivide_data, experiment_manager_not_available, datapackage_not_available
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

import os, sys

try:
    import experiment_manager as em

    try:
        import datapackage
    except ImportError:
        datapackage = datapackage_not_available()

except ImportError as e:
    em = experiment_manager_not_available('NOT ALL DATASETS AVAILABLE')




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


def meta_omniglot(folder=None, std_num_classes=None, std_num_examples=None,
                  one_hot_enc=True, rand=0, n_splits=None):
    if em is None:
        return experiment_manager_not_available('meta_omniglot NOT AVAILABLE!')

    if folder is None:
        folder = os.path.join(os.getcwd(), 'DATA')
        if not os.path.exists(folder):
            os.mkdir(folder)
    try:
        return em.load.meta_omniglot(folder, std_num_classes=std_num_classes, std_num_examples=std_num_examples,
                                     one_hot_enc=one_hot_enc, _rand=rand, n_splits=n_splits)
    except FileNotFoundError:
        print('DOWNLOADING DATA')

        package = Package('https://datahub.io/lucfra/omniglot_resized/datapackage.json')

        # print list of all resources:
        print(package.resource_names)

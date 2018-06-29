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


def mnist(data_root_folder=None, one_hot=True, partitions=(0.8, .1,), shuffle=False):
    """
    Loads (download if necessary) Mnist dataset, and optionally splits it to form different training, validation
    and test sets (use partitions parameters for that)
    """
    data_folder_name = 'mnist'

    if data_root_folder is None:
        data_root_folder = os.path.join(os.getcwd(), 'DATA')
        if not os.path.exists(data_root_folder):
            os.mkdir(data_root_folder)
    data_folder = os.path.join(data_root_folder, data_folder_name)

    datasets = read_data_sets(data_folder, one_hot=one_hot)
    train = Dataset(datasets.train.images, datasets.train.labels, name='MNIST')
    validation = Dataset(datasets.validation.images, datasets.validation.labels, name='MNIST')
    test = Dataset(datasets.test.images, datasets.test.labels, name='MNIST')
    res = [train, validation, test]
    if partitions:
        res = redivide_data(res, partition_proportions=partitions, shuffle=shuffle)
    return Datasets.from_list(res)


def meta_omniglot(data_root_folder=None, std_num_classes=None, std_num_examples=None,
                  one_hot_enc=True, rand=0, n_splits=None):
    """
    Loads, and downloads if necessary, Omniglot meta-dataset
    """
    data_folder_name = 'omniglot_resized'

    if em is None:
        return experiment_manager_not_available('meta_omniglot NOT AVAILABLE!')

    if data_root_folder is None:
        data_root_folder = os.path.join(os.getcwd(), 'DATA')
        if not os.path.exists(data_root_folder):
            os.mkdir(data_root_folder)
    data_folder = os.path.join(data_root_folder, data_folder_name)

    if os.path.exists(data_folder):
        print('DATA FOLDER IS:', data_folder)
        print('LOADING META-DATASET')
        return em.load.meta_omniglot(data_folder, std_num_classes=std_num_classes, std_num_examples=std_num_examples,
                                     one_hot_enc=one_hot_enc, _rand=rand, n_splits=n_splits)
    else:
        print('DOWNLOADING DATA')

        package = datapackage.Package('https://datahub.io/lucfra/omniglot_resized/datapackage.json')

        with open('tmp_omniglot_resized.zip', 'wb') as f:
            f.write(package.get_resource('omniglot_resized').raw_read())

        import zipfile
        zip_ref = zipfile.ZipFile('tmp_omniglot_resized.zip', 'r')
        print('EXTRACTING DATA')
        zip_ref.extractall(data_root_folder)
        zip_ref.close()

        os.remove('tmp_omniglot_resized.zip')

        print('DONE')

        # os.tmpfile()
        return meta_omniglot(data_root_folder, std_num_classes, std_num_examples,
                             one_hot_enc, rand, n_splits)


if __name__ == '__main__':
    print(meta_omniglot())
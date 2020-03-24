from __future__ import absolute_import, print_function, division


from functools import reduce
import numpy as np

from far_ho.examples.datasets import Dataset
from far_ho.utils import merge_dicts

import sys


def test():
    print(1)


def get_data(d_set):
    if hasattr(d_set, 'images'):
        data = d_set.images
    elif hasattr(d_set, 'data'):
        data = d_set.data
    else:
        raise ValueError("something wrong with the dataset %s" % d_set)
    return data


def get_targets(d_set):
    if hasattr(d_set, 'labels'):
        return d_set.labels
    elif hasattr(d_set, 'target'):
        return d_set.target
    else:
        raise ValueError("something wrong with the dataset %s" % d_set)


def redivide_data(datasets, partition_proportions=None, shuffle=False, seed=None):
    """
    Function that redivides datasets. Can be use also to shuffle or filter or map examples.

    :param datasets: original datasets, instances of class Dataset (works with get_data and get_targets for
                        compatibility with mnist datasets
    :param partition_proportions: (optional, default None)  list of fractions that can either sum up to 1 or less
                                    then one, in which case one additional partition is created with
                                    proportion 1 - sum(partition proportions).
                                    If None it will retain the same proportion of samples found in datasets
    :param shuffle: (optional, default False) if True shuffles the examples
    :return: a list of datasets of length equal to the (possibly augmented) partition_proportion
    """
    rnd = np.random.RandomState(seed)
    all_data = np.vstack([get_data(d) for d in datasets])
    all_labels = np.vstack([get_targets(d) for d in datasets])

    all_infos = np.concatenate([d.sample_info for d in datasets])

    N = all_data.shape[0]

    if partition_proportions:  # argument check
        partition_proportions = list([partition_proportions] if isinstance(partition_proportions, float)
                                     else partition_proportions)
        if isinstance(partition_proportions[0], float):
            sum_proportions = sum(partition_proportions)
            assert sum_proportions <= 1, "partition proportions must sum up to at most one: %d" % sum_proportions
            if sum_proportions < 1.: partition_proportions += [1. - sum_proportions]
        if isinstance(partition_proportions[0], int):
            if sum(partition_proportions) < N:
                partition_proportions += [N - sum(partition_proportions)]
    else:
        partition_proportions = [1. * get_data(d).shape[0] / N for d in datasets]

    if shuffle:
        permutation = np.arange(all_data.shape[0])
        rnd.shuffle(permutation)

        all_data = all_data[permutation]
        all_labels = np.array(all_labels[permutation])
        all_infos = np.array(all_infos[permutation])

    assert N == all_labels.shape[0]

    calculated_partitions = reduce(
        lambda v1, v2: v1 + [v1[-1] + v2],
        [int(prp if isinstance(prp, int) else N * prp) for prp in partition_proportions],
        [0]
    )
    calculated_partitions[-1] = N

    print('datasets.redivide_data:, computed partitions numbers -',
          calculated_partitions, 'len all', N, end=' ')

    new_general_info_dict = merge_dicts(*[d.info for d in datasets])

    new_datasets = [
        Dataset(data=all_data[d1:d2], target=all_labels[d1:d2], sample_info=all_infos[d1:d2],
                info=new_general_info_dict)
        for d1, d2 in zip(calculated_partitions, calculated_partitions[1:])
        ]

    print('DONE')
    return new_datasets


def experiment_manager_not_available(message=None):
    if message: print(message, file=sys.stderr)
    print('PLEASE INSTALL experiment_manager package', file=sys.stderr)
    print('get it at https://github.com/lucfra/ExperimentManager', file=sys.stderr)
    return None


def datapackage_not_available(message=None):
    if message: print(message, file=sys.stderr)
    print('PLEASE INSTALL datapackage (see https://datahub.io)', file=sys.stderr)
    print("execute 'pip install datapackage'", file=sys.stderr)
    return None

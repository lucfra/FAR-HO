from __future__ import absolute_import, print_function, division

import sys
import numpy as np
from far_ho import utils


def maybe_cast_to_scalar(what):
    return what[0] if len(what) == 1 else what


class Datasets:
    """
    Simple object for standard datasets. Has the field `train` `validation` and `test` and supports iterations and
    indexing
    """

    def __init__(self, train=None, validation=None, test=None):
        self.train = train
        self.validation = validation
        self.test = test
        self._lst = [train, validation, test]

    def setting(self):
        return {k: v.setting() if hasattr(v, 'setting') else None for k, v in vars(self).items()}

    def __getitem__(self, item):
        return self._lst[item]

    def __len__(self):
        return len([_ for _ in self._lst if _ is not None])

    @property
    def name(self):
        return self.train.name   # could be that different datasets have different names....

    @staticmethod
    def from_list(list_of_datasets):
        """
        Generates a `Datasets` object from a list.

        :param list_of_datasets: list containing from one to three dataset
        :return:
        """
        train, valid, test = None, None, None
        train = list_of_datasets[0]
        if len(list_of_datasets) > 3:
            print('There are more then 3 Datasets here...')
            return list_of_datasets
        if len(list_of_datasets) > 1:
            test = list_of_datasets[-1]
            if len(list_of_datasets) == 3:
                valid = list_of_datasets[1]
        return Datasets(train, valid, test)

NAMED_SUPPLIER = {}


class Dataset:
    """
    Class for managing a single dataset, includes data and target fields and has some utility functions.
     It allows also to convert the dataset into tensors and to store additional information both on a
     per-example basis and general infos.
    """

    def __init__(self, data, target, sample_info=None, info=None, name=None):
        """

        :param data: Numpy array containing data
        :param target: Numpy array containing targets
        :param sample_info: either an array of dicts or a single dict, in which case it is cast to array of
                                  dicts.
        :param info: (optional) dictionary with further info about the dataset
        """
        self._tensor_mode = False
        # self._name = name

        self._data = data
        self._target = target
        if self._data is not None:  # in meta-dataset data and target can be unspecified
            if sample_info is None:
                sample_info = {}
            self.sample_info = np.array([sample_info] * self.num_examples) \
                if isinstance(sample_info, dict) else sample_info

            assert self.num_examples == len(self.sample_info), str(self.num_examples) + ' ' + str(len(self.sample_info))
            assert self.num_examples == self._shape(self._target)[0]

        self.info = info or {}
        self.info.setdefault('_name', name)

    @property
    def name(self):
        return self.info['_name']

    def _shape(self, what):
        return what.get_shape().as_list() if self._tensor_mode else what.shape

    def setting(self):
        """
        for save setting purposes, does not save the actual data

        :return:
        """
        return {
            'num_examples': self.num_examples,
            'dim_data': self.dim_data,
            'dim_target': self.dim_target,
            'info': self.info
        }

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    @property
    def num_examples(self):
        """

        :return: Number of examples in this dataset
        """
        return self._shape(self.data)[0]

    @property
    def dim_data(self):
        """

        :return: The data dimensionality as an integer, if input are vectors, or a tuple in the general case
        """
        return maybe_cast_to_scalar(self._shape(self.data)[1:])

    @property
    def dim_target(self):
        """

        :return: The target dimensionality as an integer, if targets are vectors, or a tuple in the general case
        """
        shape = self._shape(self.target)
        return 1 if len(shape) == 1 else maybe_cast_to_scalar(shape[1:])

    def create_supplier(self, x, y, batch_size=None, other_feeds=None, name=None):
        """
        Return a standard feed dictionary for this dataset.

        :param name: if not None, register this supplier in dict NAMED_SUPPLIERS (this can be useful for instance
                        when recording with rf.Saver)
        :param x: placeholder for data
        :param y: placeholder for target
        :param batch_size: A size for the mini-batches. If None builds a supplier
                            for the entire dataset
        :param other_feeds: optional other feeds (dictionary or None)
        :return: a callable.
        """
        if batch_size:
            _supplier = SamplingWithoutReplacement(
                self, batch_size).create_supplier(x, y, other_feeds)
        else:
            def _supplier(step=0):
                return utils.merge_dicts({x: self.data, y: self.target},
                                         utils.maybe_call(other_feeds, step))

            if name:
                NAMED_SUPPLIER[name] = _supplier

        return _supplier


class SamplingWithoutReplacement:
    def __init__(self, dataset, batch_size, epochs=None):
        """
        Class for stochastic sampling of data points. It is most useful for feeding examples for the the
        training ops of `ReverseHG` or `ForwardHG`. Most notably, if the number of epochs is specified,
        the class takes track of the examples per mini-batches which is important for the backward pass
        of `ReverseHG` method.

        :param dataset: instance of `Dataset` class
        :param batch_size:
        :param epochs: number of epochs (can be None, in which case examples are
                        fed continuously)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.T = int(np.ceil(dataset.num_examples / batch_size))
        if self.epochs: self.T *= self.epochs

        self.training_schedule = None
        self.iter_per_epoch = int(dataset.num_examples / batch_size)

    def setting(self):
        excluded = ['training_schedule', 'datasets']
        dictionary = {k: v for k, v in vars(self).items() if k not in excluded}
        if hasattr(self.dataset, 'setting'):
            dictionary['dataset'] = self.dataset.setting()
        return dictionary

    def generate_visiting_scheme(self):
        """
        Generates and stores example visiting scheme, as a numpy array of integers.

        :return: self
        """

        def all_indices_shuffled():
            _res = list(range(self.dataset.num_examples))
            np.random.shuffle(_res)
            return _res

        # noinspection PyUnusedLocal
        _tmp_ts = np.concatenate([all_indices_shuffled()
                                  for _ in range(self.epochs or 1)])
        self.training_schedule = _tmp_ts if self.training_schedule is None else \
            np.concatenate([self.training_schedule, _tmp_ts])  # do not discard previous schedule,
        # this should allow backward passes of arbitrary length

        return self

    def create_supplier(self, x, y, other_feeds=None, name=None):
        return self.create_feed_dict_supplier(x, y, *other_feeds, name=name)

    def create_feed_dict_supplier(self, x, y, other_feeds=None, name=None):
        """

        :param name: optional name for this supplier
        :param x: placeholder for independent variable
        :param y: placeholder for dependent variable
        :param other_feeds: dictionary of other feeds (e.g. dropout factor, ...) to add to the input output
                            feed_dict
        :return: a function that generates a feed_dict with the right signature for Reverse and Forward HyperGradient
                    classes
        """

        def _training_supplier(step=0):

            if step >= self.T:
                if step % self.T == 0:
                    if self.epochs:
                        print('WARNING: End of the training scheme reached.'
                              'Generating another scheme.', file=sys.stderr)
                    self.generate_visiting_scheme()
                step %= self.T

            if self.training_schedule is None:
                self.generate_visiting_scheme()

            # noinspection PyTypeChecker
            nb = self.training_schedule[step * self.batch_size: min(
                (step + 1) * self.batch_size, len(self.training_schedule))]

            bx = self.dataset.data[nb, :]
            by = self.dataset.target[nb, :]

            # if lambda_feeds:  # this was previous implementation... dunno for what it was used for
            #     lambda_processed_feeds = {k: v(nb) for k, v in lambda_feeds.items()}  previous implementation...
            #  looks like lambda was
            # else:
            #     lambda_processed_feeds = {}
            return utils.merge_dicts({x: bx, y: by}, utils.maybe_call(other_feeds, step))

        if name:
            NAMED_SUPPLIER[name] = _training_supplier

        return _training_supplier

import tensorflow as tf
from experiment_manager.utils import *


# noinspection PyClassHasNoInit
class GraphKeys(tf.GraphKeys):
    """
    adds HYPERPARAMETERS key
    """

    HYPERPARAMETERS = 'hyperparameters'
    LAGRANGIAN_MULTIPLIERS = 'lagrangian_multipliers'
    HYPERGRADIENTS = 'hypergradients'


def hyperparameters(scope=None):
    return tf.get_collection(GraphKeys.HYPERPARAMETERS, scope=scope)


def lagrangian_multipliers(scope=None):
    return tf.get_collection(GraphKeys.LAGRANGIAN_MULTIPLIERS, scope=scope)


def hypergradients(scope=None):
    return tf.get_collection(GraphKeys.HYPERGRADIENTS, scope=scope)


def vectorize_all(var_list, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, 'Vectorization', var_list):
        return tf.concat([tf.reshape(_w, [-1]) for _w in var_list], 0)


def maybe_call(obj, *args, **kwargs):
    if callable(obj): return obj(*args, **kwargs)
    else: return obj


def dot(a, b, name=None):
    with tf.name_scope(name, 'Dot', [a, b]):
        return tf.reduce_sum(a*b, name=name)
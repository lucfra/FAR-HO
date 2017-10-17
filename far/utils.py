import tensorflow as tf


# noinspection PyClassHasNoInit
class GraphKeys(tf.GraphKeys):
    """
    adds HYPERPARAMETERS key
    """

    HYPERPARAMETERS = 'hyperparameters'


def vectorize_all(var_list, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, 'Vectorization', var_list):
        return tf.concat([tf.reshape(_w, [-1]) for _w in var_list], 0)


def dot(a, b, name=None):
    with tf.name_scope(name, 'Dot', [a, b]):
        return tf.reduce_sum(a*b, name=name)
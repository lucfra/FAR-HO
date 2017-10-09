import tensorflow as tf


class GraphKeys(tf.GraphKeys):
    """
    adds HYPERPARAMETERS key
    """

    HYPERPARAMETERS = 'hyperparameters'


def get_hyperparameter(name, initializer=None, shape=None, dtype=None):
    return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                           collections=[GraphKeys.HYPERPARAMETERS])

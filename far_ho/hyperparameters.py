import tensorflow as tf
from far_ho.utils import GraphKeys


def get_hyperparameter(name, initializer=None, shape=None, dtype=None):
    return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                           collections=[GraphKeys.HYPERPARAMETERS])

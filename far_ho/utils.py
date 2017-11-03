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
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


def dot(a, b, name=None):
    with tf.name_scope(name, 'Dot', [a, b]):
        return tf.reduce_sum(a*b, name=name)


def check():
    print(3)


def maybe_eval(a, ss):
    if hasattr(a, 'eval') or hasattr(a, 'run'):
        return ss.run(a)
    return a


def remove_from_collection(key, *lst):
    """
    Remove tensors in lst from collection given by key
    :param key:
    :param lst:
    :return: None
    """
    # noinspection PyProtectedMember
    [tf.get_default_graph()._collections[key].remove(e) for e in lst]


def _maybe_add(a, b):
    return a if b is None else a + b


def val_or_zero(a, b):
    return a if a is not None else tf.zeros_like(b)


def cross_entropy_loss(labels, logits, linear_input=True, eps=1.e-5, name='cross_entropy_loss'):
    """
    Clipped standard-version cross entropy loss. Implemented because  the standard function
    tf.nn.softmax_cross_entropy_with_logits has wrong (?) Hessian.
    Clipped because it easily brings to nan otherwise, especially when calculating the Hessian.

    Maybe the code could be optimized since ln(softmax(z_j)) = z_j - prod z_i . Should benchmark it.

    :param labels:
    :param logits: softmax or linear output of the model
    :param linear_input: True (default) if y is linear in which case tf.nn.softmax will be applied to y
    :param eps: (optional, default 1.e-5) clipping value for log.
    :param name: (optional, default cross_entropy_loss) name scope for the defined operations.
    :return: tensor for the cross_entropy_loss (WITHOUT MEAN ON THE EXAMPLES)
    """
    with tf.name_scope(name):
        softmax_out = tf.nn.softmax(logits) if linear_input else logits
        return -tf.reduce_sum(
            labels * tf.log(tf.clip_by_value(softmax_out, eps, 1. - eps)), reduction_indices=[1]
        )


def binary_cross_entropy(labels, logits, linear_input=True, eps=1.e-5, name='binary_cross_entropy_loss'):
    """
    Same as cross_entropy_loss for the binary classification problem. the model should have a one dimensional output,
    the targets should be given in form of a matrix of dimensions batch_size x 1 with values in [0,1].

    :param labels:
    :param logits: sigmoid or linear output of the model
    :param linear_input: (default: True) is y is linear in which case tf.nn.sigmoid will be applied to y
    :param eps: (optional, default 1.e-5) clipping value for log.
    :param name: (optional, default binary_cross_entropy_loss) name scope for the defined operations.
    :return: tensor for the cross_entropy_loss (WITHOUT MEAN ON THE EXAMPLES)
    """
    with tf.name_scope(name):
        sigmoid_out = tf.nn.sigmoid(logits)[:, 0] if linear_input else logits
        # tgs = targets if len(targets.)
        return - (labels * tf.log(tf.clip_by_value(sigmoid_out, eps, 1. - eps)) +
                  (1. - labels) * tf.log(tf.clip_by_value(1. - sigmoid_out, eps, 1. - eps)))
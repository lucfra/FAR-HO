from __future__ import absolute_import, print_function, division

import numpy as np

import tensorflow as tf
import sys
# noinspection PyUnresolvedReferences
try:
    from experiment_manager.utils import *
except ImportError:
    # print('package experiment_manager not found')
    pass


def flatten_list(lst):
    from itertools import chain
    return list(chain(*lst))


def merge_dicts(*dicts):
    from functools import reduce
    return reduce(lambda a, nd: merge_two_dicts(a, nd), dicts, {})


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.
    """
    return obj if isinstance(obj, list) else [obj]


# noinspection PyClassHasNoInit
class GraphKeys(tf.GraphKeys):
    """
    adds some hyperparameters and hypergradients computation related keys
    """

    HYPERPARAMETERS = 'hyperparameters'
    LAGRANGIAN_MULTIPLIERS = 'lagrangian_multipliers'
    HYPERGRADIENTS = 'hypergradients'
    ZS = 'zs'


def hyperparameters(scope=None):
    """
    List of variables in the collection HYPERPARAMETERS.

    Hyperparameters constructed with `get_hyperparameter` are in this collection by default.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.HYPERPARAMETERS, scope=scope)


def lagrangian_multipliers(scope=None):
    """
    List of variables in the collection LAGRANGIAN_MULTIPLIERS.

    These variables are created by `far.ReverseHG`.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.LAGRANGIAN_MULTIPLIERS, scope=scope)


def hypergradients(scope=None):
    """
    List of tensors and/or variables in the collection HYPERGRADIENTS.

    These variables are created by `far.HyperGradient`.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.HYPERGRADIENTS, scope=scope)


def vectorize_all(var_list, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, 'Vectorization', var_list) as scope:
        return tf.concat([tf.reshape(_w, [-1]) for _w in var_list], 0, name=scope)


def reduce_all_sums(lst1, lst2, name=None):
    with tf.name_scope(name, 'Vectorization', lst1 + lst2) as scope:
        return tf.add_n([tf.reduce_sum(v1*v2) for v1, v2 in zip(lst1, lst2)], name=scope)


def maybe_call(obj, *args, **kwargs):
    """
    Calls obj with args and kwargs and return its result if obj is callable, otherwise returns obj.
    """
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


def dot(a, b, name=None):
    """
    Dot product between vectors `a` and `b` with optional name.
    If a and b are not vectors, formally this computes <vec(a), vec(b)>.
    """
    # assert a.shape.ndims == 1, '{} must be a vector'.format(a)
    # assert b.shape.ndims == 1, '{} must be a vector'.format(b)
    with tf.name_scope(name, 'Dot', [a, b]):
        return tf.reduce_sum(a*b)


def _check():
    print(5)


def maybe_eval(a, ss=None):
    """
    Run or eval `a` and returns the result if possible.

    :param a: object, or `tf.Variable` or `tf.Tensor`
    :param ss: `tf.Session` or get default session (if any)
    :return: If a is not a tensorflow evaluable returns it, or returns the
                resulting call
    """
    if ss is None: ss = tf.get_default_session()
    if hasattr(a, 'eval') or hasattr(a, 'run'):
        return ss.run(a)
    return a


def remove_from_collection(key, *lst):
    """
    Remove tensors in lst from collection given by key
    """
    try:
        # noinspection PyProtectedMember
        [tf.get_default_graph()._collections[key].remove(e) for e in lst]
    except ValueError:
        print('WARNING: Collection -> {} <- does not contain some tensor in {}'.format(key, lst),
              file=sys.stderr)


def maybe_add(a, b):
    """
    return a if b is None else a + b
    """
    return a if b is None else a + b


def val_or_zero(a, b):
    """
    return a if a is not None else tf.zeros_like(b)
    """
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


# TODO put some other useful errors (with mean and so on) and scores, and add them to correct collections...
def maybe_track_tensor(iter_op, tensor):
    """
    :return: a list of ops to run and a boolean that is true if tensor was actually a tensor to be tracked
    """
    to_be_run = [iter_op]
    track_tensor = isinstance(tensor, tf.Tensor)
    if track_tensor:  # in most cases this check should be fine
        with tf.control_dependencies(iter_op):  # be sure that tensor is computed AFTER the (optimization) iteration
            to_be_run.append(tf.identity(tensor))
    return to_be_run, track_tensor


def isinteger(num):
    return isinstance(num, (int, np.int_, np.int8, np.int16, np.int32, np.int64))

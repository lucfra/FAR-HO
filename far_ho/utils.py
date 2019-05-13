from __future__ import absolute_import, print_function, division

import sys

import tensorflow as tf

# noinspection PyUnresolvedReferences
try:
    from experiment_manager.utils import *
except ImportError as err:
    print('Warning: Experiment manager not loaded (https://github.com/lucfra/ExperimentManager)')
    print(err)


def as_tuple_or_list(obj):
    """
    Make sure that `obj` is a tuple or a list and eventually converts it into a list with a single element

    :param obj:
    :return: A `tuple` or a `list`
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def flatten_list(lst):
    from itertools import chain
    return list(chain(*lst))


def merge_dicts(*dicts):
    """
    Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
    """
    from functools import reduce
    # if len(dicts) == 1: return dicts[0]
    return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


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


def solve_int_or_generator(int_or_generator):
    return range(int_or_generator) if isinteger(int_or_generator) else int_or_generator


def remove_from_collection(key, *lst):
    """
    Remove tensors in lst from collection given by key
    """
    try:
        # noinspection PyProtectedMember
        [tf.get_default_graph()._collections[key].remove(_e) for _e in lst]
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


def isinteger(num):
    return isinstance(num, (int, np.int_, np.int8, np.int16, np.int32, np.int64))

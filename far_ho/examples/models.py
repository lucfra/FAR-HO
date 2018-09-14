"""
This is a very small module that contains an abstract implementation of a parametric function and
functions for initializing very classic feed forward neural networks.
`fixed_init_ffnn` is a function for controlling once for all the initialization of the weights of the nn, without
the need of re-initializing tensorflow session!
"""

import tensorflow as tf
from sys import stderr

try:
    from experiment_manager import maybe_get
except ModuleNotFoundError or ImportError as e:
    maybe_get = None
    print('models.py; WARNING: some functions may not work', file=stderr)
    print(e, file=stderr)

from tensorflow.python.client.session import register_session_run_conversion_functions


class ParametricFunction:
    def __init__(self, x, params, y, rule, **kwargs):
        self.x = x
        self.y = y
        self.params = params
        self.rule = rule
        self._kwargs = kwargs

    @property
    def var_list(self):
        return self.params

    @property
    def out(self):
        return self.y

    def with_params(self, new_params):
        return self.rule(self.x, new_params, **self._kwargs)

    def for_input(self, new_x):
        return self.rule(new_x, self.params, **self._kwargs)

    # noinspection PyProtectedMember
    def __add__(self, other):
        """
        Note: assumes that the inputs are the same
        """
        assert isinstance(other, ParametricFunction)
        return ParametricFunction(self.x, [self.params, other.params], self.y + other.y,
                                  lambda _inp, _prm, **kwa: kwa['add_arg1'].rule(
                                      _inp, _prm[0], **kwa['add_arg1']._kwargs) + kwa['add_arg2'].rule(
                                      _inp, _prm[1], **kwa['add_arg2']._kwargs),
                                  add_arg1=self, add_arg2=other)

    # noinspection PyProtectedMember
    def __matmul__(self, other):
        """
        Implements mathematical composition: self \circ other
        """
        assert isinstance(other, ParametricFunction)
        return ParametricFunction(other.x, [self.params, other.params], self.for_input(other.y),
                                  lambda _inp, _prm, **kwa:
                                  kwa['comp_arg1'].rule(
                                      _inp, _prm[0], **kwa['comp_arg1']._kwargs).__matmul__(
                                      kwa['comp_arg2'].rule(_inp, _prm[1], **kwa['comp_arg2']._kwargs)),
                                  comp_arg1=self, comp_arg2=other)


tf.register_tensor_conversion_function(ParametricFunction,
                                       lambda value, dtype=None, name=None, as_ref=False:
                                       tf.convert_to_tensor(value.y, dtype, name))

register_session_run_conversion_functions(ParametricFunction,
                                          lambda pf: ([pf.y], lambda val: val[0]))


def _process_initializer(initializers, j, default):
    if callable(initializers):
        return initializers
    elif initializers is not None:
        return maybe_get(initializers, j)
    else:
        return default


def _pass_shape(shape, initializer, j):
    init = maybe_get(initializer, j)
    return None if (hasattr(init, 'shape') or isinstance(init, list)) else shape


# noinspection PyUnusedLocal
def id_pf(x, weights=None):
    """
    Identity as ParametricFunction
    """
    return ParametricFunction(x, [], x, id_pf)


def lin_func(x, weights=None, dim_out=None, activation=None, initializers=None,
             name='lin_model', variable_getter=tf.get_variable):
    assert dim_out or weights
    with tf.variable_scope(name):
        if weights is None:
            weights = [variable_getter('w', initializer=_process_initializer(initializers, 0, tf.zeros_initializer),
                                       shape=_pass_shape((x.shape[1], dim_out), initializers, 0)),
                       variable_getter('b', initializer=_process_initializer(initializers, 1, tf.zeros_initializer),
                                       shape=_pass_shape((dim_out,), initializers, 1))]
        out = tf.matmul(x, weights[0]) + weights[1]
        if activation:
            out = activation(out)
        return ParametricFunction(x, weights, out, lin_func, activation=activation)


def ffnn(x, weights=None, dims=None, activation=tf.nn.relu, name='ffnn', initiazlizers=None,
         variable_getter=tf.get_variable, verbose=False):
    """
    Constructor for a feed-forward neural net as Parametric function
    """
    assert dims or weights

    with tf.variable_scope(name):
        params = weights if weights is not None else []
        out = x
        n_layers = len(dims) if dims else len(weights) // 2 + 1

        if verbose: print('begin of ', name, '-' * 5)
        for i in range(n_layers - 1):
            if weights is None:
                with tf.variable_scope('layer_{}'.format(i + 1)):
                    params += [variable_getter('w',
                                               shape=_pass_shape((dims[i], dims[i + 1]), initiazlizers, 2 * i),
                                               dtype=tf.float32,
                                               initializer=_process_initializer(initiazlizers, 2 * i, None)),
                               variable_getter('b', shape=_pass_shape((dims[i + 1],), initiazlizers, 2 * i),
                                               initializer=_process_initializer(initiazlizers, 2 * i + 1,
                                                                                tf.zeros_initializer))]
            out = tf.matmul(out, params[2 * i]) + params[2 * i + 1]
            if i < n_layers - 2: out = activation(out)
            if verbose: print(out)
        if verbose: print('end of ', name, '-' * 5)
        return ParametricFunction(x, params, out, ffnn, activation=activation)


def fixed_init_ffnn(x, weights=None, dims=None, activation=tf.nn.relu, name='ffnn', initializers=None,
                    variable_getter=tf.get_variable):
    from time import time
    import numpy as np
    _temp = ffnn(x, weights, dims, activation, 'tempRandomNet' + str(int(time())) + str(np.random.randint(0, 10000)),
                 initializers)
    new_session = False
    session = tf.get_default_session()
    if session is None:
        session = tf.InteractiveSession()
        new_session = True
    tf.variables_initializer(_temp.var_list).run()
    net = ffnn(x, weights, dims, activation, name, session.run(_temp.var_list), variable_getter=variable_getter)
    if new_session: session.close()
    return net

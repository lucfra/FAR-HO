from __future__ import print_function, absolute_import, division

# import numpy as np

import tensorflow as tf
from collections import OrderedDict

from far_ho import utils

GRADIENT_NONE_MESSAGE = 'WARNING: the gradient w.r.t.the tf.Variable\n {}\n is None;\n ' \
                        'Check the computational graph of the inner objective, and be sure you\n' \
                        'are not considering including variables that should not be there among the\n' \
                        'inner variables.'


class OptimizerDict(object):
    def __init__(self, ts, dynamics, objective):
        self._ts = ts
        self._dynamics = dynamics
        self._iteration = None
        self._initialization = None
        self._init_dyn = None  # for phi_0 (will be a dictionary (state-variable, phi_0 op)
        self.objective = objective

    @property
    def ts(self):
        """
        Descent step, as returned by `tf.train.Optimizer.apply_gradients`.
        :return:
        """
        return self._ts

    @property
    def iteration(self):
        """
        Performs a descent step (as return by `tf.train.Optimizer.apply_gradients`) and computes the values of
        the variables after it.

        :return: A list of operation that, after performing one iteration, return the value of the state variables
                    being optimized (possibly including auxiliary variables)
        """
        if self._iteration is None:
            with tf.control_dependencies([self._ts]):
                self._iteration = self._state_read()  # performs an iteration and returns the
                # value of all variables in the state (ordered according to dyn)

        return self._iteration

    @property
    def initialization(self):
        """
        :return: a list of operations that return the values of the state variables for this
                    learning dynamics after the execution of the initialization operation. If
                    an initial dynamics is set, then it also executed.
        """
        if self._initialization is None:
            with tf.control_dependencies([tf.variables_initializer(self.state)]):
                if self._init_dyn is not None:  # create assign operation for initialization
                    self._initialization = [k.assign(v) for k, v in self._init_dyn.items()]
                    # return these new initialized values (and ignore variable initializers)
                else:
                    self._initialization = self._state_read()  # initialize state variables and
                    # return the initialized value

        return self._initialization

    @property
    def dynamics(self):
        """
        :return: A generator for the dynamics (state_variable_{k+1})
        """
        return self._dynamics.values()

    @property
    def dynamics_dict(self):
        return self._dynamics

    @property
    def state(self):
        """
        :return: A generator for all the state variables (optimized variables and possibly auxiliary variables)
        being optimized
        """
        return self._dynamics.keys()  # overridden in Adam

    def _state_read(self):
        """
        :return: generator of read value op for the state variables
        """
        return [v.read_value() for v in self.state]  # not sure about read_value vs value

    def state_feed_dict(self, his):
        """
        Builds a feed dictionary of (past) states
        """
        return {v: his[k] for k, v in enumerate(self.state)}

    def set_init_dynamics(self, init_dictionary):
        """
        With this function is possible to set an initializer for the dynamics. Multiple calls of this method on the
        same variable will override the dynamics.

        :param init_dictionary: a dictionary of (state_variable: tensor or variable, that represents the initial
                                dynamics Phi_0.
        """
        if self._init_dyn is None:
            self._init_dyn = OrderedDict([(v, tf.identity(v)) for v in self.state])  # do nothing
        for k, v in init_dictionary.items():
            assert k in self._init_dyn, 'Can set initial dynamics only for state variables in this object, got %s' % k
            self._init_dyn[k] = v

    @property
    def init_dynamics(self):
        """
        :return: The initialization dynamics if it has been set, or `None` otherwise.
        """
        return None if self._init_dyn is None else list(self._init_dyn.items())

    def __lt__(self, other):  # make OptimizerDict sortable
        # TODO be sure that this is consistent
        assert isinstance(other, OptimizerDict)
        return hash(self) < hash(other)

    def __len__(self):
        return len(self._dynamics)


# noinspection PyAbstractClass,PyClassHasNoInit
class Optimizer(tf.train.Optimizer):
    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                 aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):
        """
        Returns an `OptimizerDict` object relative to this minimization. See tf.train.Optimizer.minimize.

        `OptimizerDict` objects notably contain a field `ts` for the training step and
        and a field `dynamics` for the optimization dynamics. The `dynamics` a list of
        var_and_dynamics where var are both variables in `var_list` and also
        additional state (auxiliary) variables, as needed.
        """
        ts, dyn = super(Optimizer, self).minimize(loss, global_step, var_list, gate_gradients, aggregation_method,
                                                  colocate_gradients_with_ops, name, grad_loss)
        return OptimizerDict(ts=ts, dynamics=dyn, objective=loss)

    def _tf_minimize(self, loss, global_step=None, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                     aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):
        return super(Optimizer, self).minimize(loss, global_step, var_list, gate_gradients, aggregation_method,
                                               colocate_gradients_with_ops, name, grad_loss)

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def learning_rate_tensor(self):
        return self._learning_rate_tensor

    @property
    def optimizer_params_tensor(self):
        return [self.learning_rate_tensor]

    @staticmethod
    def tf():
        return tf.train.Optimizer


# noinspection PyClassHasNoInit,PyAbstractClass
class GradientDescentOptimizer(Optimizer, tf.train.GradientDescentOptimizer):
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        ts = super(GradientDescentOptimizer, self).apply_gradients(grads_and_vars, global_step, name)
        dynamics = OrderedDict()
        for g, w in grads_and_vars:
            assert g is not None, GRADIENT_NONE_MESSAGE.format(w)
            wk = w - tf.cast(self._learning_rate_tensor, g.dtype) * g
            dynamics[w] = wk
        return ts, dynamics

    def __str__(self):
        return '{}-lr={}'.format(self._name, self._learning_rate)

    @staticmethod
    def tf():
        return tf.train.GradientDescentOptimizer


class BacktrackingOptimizerDict(OptimizerDict):
    def __init__(self, dynamics, objective, objective_after_step, lr0, m, tau=0.5, c=0.5):
        super(BacktrackingOptimizerDict, self).__init__(None, dynamics, objective)
        self.objective_after_step = objective_after_step
        # assert isinstance(learning_rate, (float, np.float32, np.float64)), 'learning rate must be a float'
        self.lr0 = lr0
        self.tau = tau  # decrease factor
        self.c = c

        self.m = m
        self.armillo_cond = lambda alpha: tf.greater(objective_after_step(alpha), objective + c * alpha * m)

        self.backtrack_body = lambda alpha: alpha * tau

        self.eta_k = tf.while_loop(self.armillo_cond, self.backtrack_body, [self.lr0])

        self._dynamics = OrderedDict([(v, vk1(self.eta_k, v, g)) for v, g, vk1 in dynamics])

    @property
    def ts(self):
        if self._ts is None:
            self._ts = tf.group(*[v.assign(vk1) for v, vk1 in self._dynamics.items()])
        return self._ts

    @property
    def iteration(self):
        if self._iteration is None:
            with tf.control_dependencies([self.ts]):
                self._iteration = self._state_read() + [self.eta_k]  # performs one iteration and returns the
                # value of all variables in the state (ordered according to dyn)
        return self._iteration

    def state_feed_dict(self, his):
        # considers also alpha_k
        if len(his) == len(self._dynamics):
            return {v: his[k] for k, v in enumerate(self.state)}  # for the initialization step
        return utils.merge_dicts({v: his[k] for k, v in enumerate(self.state)}, {self.eta_k: his[-1]})


# noinspection PyAbstractClass
class BackTrackingGradientDescentOptimizer(GradientDescentOptimizer):
    def __init__(self, learning_rate, c=0.5, tau=0.5, use_locking=False, name="GradientDescent"):
        super(BackTrackingGradientDescentOptimizer, self).__init__(learning_rate, use_locking, name)
        self.c = c
        self.tau = tau

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        super(BackTrackingGradientDescentOptimizer, self)._prepare()
        with tf.name_scope(name, self.get_name()):
            m = 0.
            dynamics = OrderedDict()

            def _wk(_eta, _w, _g):
                return _w - _eta * _g

            for g, w in grads_and_vars:
                assert g is not None, GRADIENT_NONE_MESSAGE.format(w)

                dynamics[w] = (g, _wk)
                m -= utils.dot(g, g)

        return dynamics, m

    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                 aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):

        assert callable(loss)

        if var_list is None:
            var_list = tf.trainable_variables()
        curr_loss = loss(var_list)

        dynamics, m = super(BackTrackingGradientDescentOptimizer,
                            self)._tf_minimize(curr_loss, global_step, var_list, gate_gradients, aggregation_method,
                                               colocate_gradients_with_ops, name, grad_loss)

        loss_after_step = lambda eta: loss([dyn(eta, v, g) for v, g, dyn in dynamics])

        return BacktrackingOptimizerDict(dynamics, curr_loss, loss_after_step, self._learning_rate_tensor,
                                         m, self.tau, self.c)

    @property
    def optimizer_params_tensor(self):
        return []

    @staticmethod
    def tf():
        return None


class MomentumOptimizer(Optimizer, tf.train.MomentumOptimizer):
    def __init__(self, learning_rate, momentum, use_locking=False, name="Momentum",
                 use_nesterov=False):
        assert use_nesterov is False, 'Nesterov momentum not implemented yet...'
        super(MomentumOptimizer, self).__init__(learning_rate, momentum, use_locking, name, use_nesterov)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        #         filter_hypers

        ts = super(MomentumOptimizer, self).apply_gradients(grads_and_vars, global_step, name)

        # builds up the dynamics here
        mn = self.get_slot_names()[0]
        dynamics = OrderedDict()
        for g, w in grads_and_vars:
            assert g is not None, GRADIENT_NONE_MESSAGE.format(w)

            m = self.get_slot(w, mn)
            mk = tf.cast(self._momentum_tensor, m.dtype) * m + g
            wk = w - tf.cast(self._learning_rate_tensor, mk.dtype) * mk
            dynamics[w] = wk
            dynamics[m] = mk

        return ts, dynamics

    def __str__(self):
        return '{}-lr={}-m={}'.format(self._name, self._learning_rate, self._momentum)

    @property
    def optimizer_params_tensor(self):
        return super(MomentumOptimizer, self).optimizer_params_tensor + [self._momentum_tensor]

    @staticmethod
    def tf():
        return tf.train.MomentumOptimizer


# noinspection PyClassHasNoInit
class AdamOptimizer(Optimizer, tf.train.AdamOptimizer):
    # changed the default value of epsilon  due to numerical stability of hypergradient computation
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-5, use_locking=False, name="Adam"):
        super(AdamOptimizer, self).__init__(learning_rate, beta1, beta2, epsilon, use_locking, name)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        ts = super(AdamOptimizer, self).apply_gradients(grads_and_vars, global_step, name)

        mn, vn = self.get_slot_names()
        dynamics = OrderedDict()

        with tf.name_scope(name, 'Adam_Dynamics'):
            try:
                b1_pow, b2_pow = self._beta1_power, self._beta2_power
            except AttributeError:  # for newer versions of tensorflow..
                b1_pow, b2_pow = self._get_beta_accumulators()
            lr_k = self._lr_t * tf.sqrt(1. - b2_pow) / (1. - b1_pow)

            lr_k = tf.cast(lr_k, grads_and_vars[0][0].dtype)
            self._beta1_t = tf.cast(self._beta1_t, grads_and_vars[0][0].dtype)
            self._beta2_t = tf.cast(self._beta2_t, grads_and_vars[0][0].dtype)
            self._epsilon_t = tf.cast(self._epsilon_t, grads_and_vars[0][0].dtype)

            for g, w in grads_and_vars:
                assert g is not None, GRADIENT_NONE_MESSAGE.format(w)

                m = self.get_slot(w, mn)
                v = self.get_slot(w, vn)
                mk = tf.add(self._beta1_t * m, (1. - self._beta1_t) * g, name=m.op.name)
                vk = tf.add(self._beta2_t * v, (1. - self._beta2_t) * g * g, name=v.op.name)

                wk = tf.subtract(w, lr_k * mk / (tf.sqrt(vk + self._epsilon_t ** 2)), name=w.op.name)
                # IMPORTANT NOTE: epsilon should be outside sqrt as from the original implementation,
                # but this brings to numerical instability of the hypergradient.

                dynamics[w] = wk
                dynamics[m] = mk
                dynamics[v] = vk

            b1_powk = b1_pow * self._beta1_t
            b2_powk = b2_pow * self._beta2_t

            dynamics[b1_pow] = b1_powk
            dynamics[b2_pow] = b2_powk

        return ts, dynamics

    def __str__(self):
        return '{}-lr={}-b1={}-b=2{}-ep={}'.format(self._name, self._lr, self._beta1, self._beta2, self._epsilon)

    @property
    def learning_rate(self):
        return self._lr

    @property
    def learning_rate_tensor(self):
        return self._lr_t

    @property
    def optimizer_params_tensor(self):
        return super(AdamOptimizer, self).optimizer_params_tensor + [self._beta1_t, self._beta2_t]

    @staticmethod
    def tf():
        return tf.train.AdamOptimizer

import tensorflow as tf
from far_ho.utils import GraphKeys


class OptimizerDict:

    def __init__(self, ts, dynamics, ddyn_dhypers=None):
        self._ts = ts
        self._dynamics = dynamics
        self.ddyn_dhypers = ddyn_dhypers  # this is used in ForwardHg
        self._iteration = None
        self._initialization = None

    @property
    def ts(self):
        """
        Descent step, as returned by `tf.train.Optimizer.apply_gradients`.
        :return:
        """
        return self._ts

    @property
    def iteration(self):
        if self._iteration is None:
            with tf.control_dependencies([self._ts]):
                self._iteration = self._state_read()  # performs an iteration and returns the
                # value of all variables in the state (ordered according to dyn)
        return self._iteration

    @property
    def initialization(self):
        if self._initialization is None:
            with tf.control_dependencies([tf.variables_initializer(self.state)]):
                self._initialization = self._state_read()  # initialize state variables and return the initialized value
        return self._initialization

    @property
    def dynamics(self):
        return self._dynamics

    @property
    def state(self):
        return [v for (v, d) in self.dynamics]  # overridden in Adam

    def _state_read(self):
        return [v.read_value() for v in self.state]  # not sure about read_value vs value

    def state_feed_dict_generator(self, history):
        state = self.state
        for t, his in enumerate(history):
            yield t, {v: his[k] for k, v in enumerate(state)}

    def __lt__(self, other):  # make OptimizerDict sortable
        assert isinstance(other, OptimizerDict)
        return hash(self) < hash(other)


# noinspection PyAbstractClass,PyClassHasNoInit
class Optimizer(tf.train.Optimizer):
    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                 aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None,
                 compute_ddyn_dhypers=False, hyperparameters=None):
        """
        Returns a training step operation and the training dynamics in the form of
        list of var_and_dynamics where var are both variables in `var_list` and also additional state (auxiliary)
        variables needed

        See tf.train.Optimizer.minimize.  Adds the computation of B_t if forward_hg is `True`

        :param loss:
        :param global_step:
        :param var_list:
        :param gate_gradients:
        :param aggregation_method:
        :param colocate_gradients_with_ops:
        :param name:
        :param grad_loss:
        :param compute_ddyn_dhypers:
        :param hyperparameters:
        :return:
        """
        ts, dyn = super().minimize(loss, global_step, var_list, gate_gradients, aggregation_method,
                                   colocate_gradients_with_ops, name, grad_loss)
        ddyn_dhypers = self._compute_ddyn_dhypers(loss, colocate_gradients_with_ops, aggregation_method,
                                                  gate_gradients, hyperparameters) if compute_ddyn_dhypers else None
        return OptimizerDict(ts=ts, dynamics=dyn, ddyn_dhypers=ddyn_dhypers)

    def _compute_ddyn_dhypers(self, loss, colocate_gradients_with_ops=None, aggregation_method=None,
                              gate_gradients=None, hyperparameters=None):
        hypers = hyperparameters or tf.get_collection(GraphKeys.HYPERPARAMETERS)
        assert all([hy.get_shape().ndims == 1 for hy in hypers]), 'only scalar hyperparameters for now'
        grad_and_hypers = self.compute_gradients(loss, hypers, gate_gradients, aggregation_method,
                                                 colocate_gradients_with_ops)
        # TODO filter for algorithmic hyperparameters (grad would be None here!)
        return [
            (tf.gradients(g, hypers, name='d_dyn_d_hypers'), v) for (g, v) in grad_and_hypers
        ]


# noinspection PyClassHasNoInit,PyAbstractClass
class GradientDescentOptimizer(Optimizer, tf.train.GradientDescentOptimizer):

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        ts = super().apply_gradients(grads_and_vars, global_step, name)
        dynamics = []
        for g, w in grads_and_vars:
            wk = w - self._learning_rate_tensor * g
            dynamics += [(w, wk)]
        return ts, dynamics


class MomentumOptimizer(Optimizer, tf.train.MomentumOptimizer):
    def __init__(self, learning_rate, momentum, use_locking=False, name="Momentum",
                 use_nesterov=False):
        assert use_nesterov is False, 'Nesterov momentum not implemented yet...'
        super().__init__(learning_rate, momentum, use_locking, name, use_nesterov)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        #         filter_hypers

        ts = super().apply_gradients(grads_and_vars, global_step, name)

        # builds up the dynamics here
        mn = self.get_slot_names()[0]
        dynamics = []
        for g, w in grads_and_vars:
            m = self.get_slot(w, mn)
            mk = self._momentum_tensor * m + g
            wk = w - self._learning_rate_tensor * mk
            dynamics += [(w, wk), (m, mk)]

        return ts, dynamics


# noinspection PyClassHasNoInit
class AdamOptimizer(Optimizer, tf.train.AdamOptimizer):
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        # TODO
        return super().apply_gradients(grads_and_vars, global_step, name)

    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                 aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None,
                 compute_ddyn_dhypers=False, hyperparameters=None):
        # TODO extend OptimizerDict to take into account also
        # self._beta1_power and self._beta2_power
        return super().minimize(loss, global_step, var_list, gate_gradients, aggregation_method,
                                colocate_gradients_with_ops, name, grad_loss, compute_ddyn_dhypers, hyperparameters)

# variables.PartitionedVariable

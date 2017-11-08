from collections import defaultdict

import tensorflow as tf

try:
    from ordered_set import OrderedSet
except ImportError:
    print('ordered-set package not found..')
    OrderedSet = set

from far_ho.optimizer import Optimizer
from far_ho.hypergradients import ReverseHg, HyperGradient
from far_ho.utils import GraphKeys, maybe_call, maybe_eval

HYPERPARAMETERS_COLLECTIONS = (GraphKeys.HYPERPARAMETERS, GraphKeys.GLOBAL_VARIABLES)


def get_hyperparameter(name, initializer=None, shape=None, dtype=None):
    """
    Creates an hyperparameter variable, which is a GLOBAL_VARIABLE
    and HYPERPARAMETER. Mirrors the behavior of `tf.get_variable`.

    :param name: name of this hyperparameter
    :param initializer: initializer or initial value (can be np.array or float)
    :param shape: optional shape, may be not needed depending on initializer
    :param dtype: optional type,  may be not needed depending on initializer
    :return: the newly created variable.
    """
    return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                           collections=HYPERPARAMETERS_COLLECTIONS)


class HyperOptimizer:
    """
    Wrapper for performing hyperparameter optimization
    """

    def __init__(self, hypergradient=None):
        assert hypergradient is None or isinstance(hypergradient, HyperGradient)
        self._hypergradient = hypergradient or ReverseHg()
        self._fin_hts = None
        self._global_step = None
        self._h_optim_dict = defaultdict(lambda: OrderedSet())

    def set_dynamics(self, inner_objective, inner_objective_optimizer, var_list=None, **minimize_kwargs):
        """
        Set the dynamics Phi: a descent procedure on some inner_objective, can be called multiple times, for instance
        for batching inner optimization problems.

        :param inner_objective: a loss function for the inner optimization problem
        :param inner_objective_optimizer: an instance of some `far.Optimizer` (optimizers from Tensorflow must be
                                            extended to include tensors for the dynamics)
        :param var_list: optional list of variables (of the inner optimization problem)
        :param minimize_kwargs: optional arguments to pass to `optimizer.minimize`
        :return: `OptimizerDict` from optimizer.
        """
        assert isinstance(inner_objective_optimizer, Optimizer)
        optim_dict = inner_objective_optimizer.minimize(
            inner_objective,
            var_list=var_list,
            **minimize_kwargs
        )
        return optim_dict

    def set_problem(self, outer_objective, optim_dict, outer_objective_optimizer,
                 hyper_list=None, global_step=None):
        """
        Set the outer optimization problem and the descent procedure for the optimization of the
        hyperparameters. Can be called at least once for every call of set_dynamics, passing the resulting
         `OptimizerDict`. It can be called multiple times with different objective, optimizers and hyper_list s.

        :param outer_objective: scalar tensor for the outer objective
        :param optim_dict: `OptimizerDict`
        :param outer_objective_optimizer: Optimizer (may be tensorflow optimizer) for the hyperparameters
        :param hyper_list: optional list of hyperparameters
        :param global_step: optional global step.
        :return: itself
        """
        hyper_list = self._hypergradient.compute_gradients(outer_objective, optim_dict, hyper_list=hyper_list)
        self._h_optim_dict[outer_objective_optimizer].update(hyper_list)
        self._global_step = global_step
        return self

    def minimize(self, outer_objective, outer_objective_optimizer, inner_objective, inner_objective_optimizer,
                 hyper_list=None, var_list=None, global_step=None):
        """
        Single function for calling once `set_dynamics` and `set_problem`. Returns

        :param outer_objective:
        :param outer_objective_optimizer:
        :param inner_objective:
        :param inner_objective_optimizer:
        :param hyper_list:
        :param var_list:
        :param global_step:
        :return:
        """
        optim_dict = self.set_dynamics(inner_objective, inner_objective_optimizer, var_list)
        self.set_problem(outer_objective, optim_dict, outer_objective_optimizer, hyper_list, global_step)
        return self.finalize()

    def finalize(self):
        """
        To be called when no more dynamics or problems are added, computes the updates
        for the hyperparameters. This is to behave nicely with global_variables_initializer.

        :return: the run method of this object.
        """
        self._hyperit()
        return self.run

    @property
    def hypergradient(self):
        """
        :return: the hypergradient object underlying this wrapper.
        """
        return self._hypergradient

    def _hyperit(self):  # TODO add names
        """
        iteration of minimization of outer objective(s).
        :return:
        """
        if self._fin_hts is None:
            # in this way also far.optimizer can be used
            _maybe_first_arg = lambda _v: _v[0] if isinstance(_v, tuple) else _v
            # apply updates to each optimizer for outer objective minimization.
            # each optimizer might have more than one group of hyperparameters to optimize
            # and conversely different hyperparameters might be optimized with different optimizers.
            self._fin_hts = tf.group(*[_maybe_first_arg(opt.apply_gradients(
                self.hypergradient.hgrads_hvars(hyper_list=hll)))
                                       for opt, hll in self._h_optim_dict.items()])
            if self._global_step:
                with tf.control_dependencies([self._fin_hts]):
                    self._fin_hts = self._global_step.assign_add(1).op
        return self._fin_hts

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, session=None, _skip_hyper_ts=False):
        """
        Run an hyper-iteration (i.e. train the model(s) and compute hypergradients) and updates the hyperparameters.

        :param T_or_generator: int or generator (that yields an int), number of iteration (or stopping condition)
                                for the inner optimization (training) dynamics
        :param inner_objective_feed_dicts: an optional feed dictionary for the inner problem. Can be a function of
                                            step, which accounts for, e.g. stochastic gradient descent.
        :param outer_objective_feed_dicts: an optional feed dictionary for the outer optimization problem
                                            (passed to the evaluation of outer objective). Can be a function of
                                            hyper-iterations steps (i.e. global variable), which may account for, e.g.
                                            stochastic evaluation of outer objective.
        :param initializer_feed_dict:  an optional feed dictionary for the initialization of inner problems variables.
                                            Can be a function of
                                            hyper-iterations steps (i.e. global variable), which may account for, e.g.
                                            stochastic initialization.
        :param session: optional session
        :param _skip_hyper_ts: if `True` does not perform hyperparameter optimization step.
        """
        self._hypergradient.run(T_or_generator, inner_objective_feed_dicts,
                                maybe_call(outer_objective_feed_dicts, maybe_eval(self._global_step)),
                                maybe_call(initializer_feed_dict, maybe_eval(self._global_step)),
                                session=session, global_step=self._global_step)
        if not _skip_hyper_ts:
            ss = session or tf.get_default_session()
            ss.run(self._hyperit())

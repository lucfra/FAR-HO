from collections import defaultdict

import tensorflow as tf
try:
    from ordered_set import OrderedSet
except ImportError:
    print('ordered-set package not found..')
    OrderedSet = set

from far_ho.optimizer import Optimizer
from far_ho.hypergradients import ReverseHg, HyperGradient
from far_ho.utils import GraphKeys


def get_hyperparameter(name, initializer=None, shape=None, dtype=None):
    return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                           collections=[GraphKeys.HYPERPARAMETERS,
                                        GraphKeys.GLOBAL_VARIABLES])


class HyperOptimizer:
    def __init__(self, hypergradient=None):
        assert hypergradient is None or isinstance(hypergradient, HyperGradient)
        self._hypergradient = hypergradient or ReverseHg()
        self._fin_hts = None
        self._global_step = None
        self._h_optim_dict = defaultdict(lambda: OrderedSet())

    def set_dynamics(self, inner_objective, inner_objective_optimizer, var_list=None, **minimize_kwargs):
        assert isinstance(inner_objective_optimizer, Optimizer)
        optim_dict = inner_objective_optimizer.minimize(
            inner_objective,
            var_list=var_list,
            compute_ddyn_dhypers=self._hypergradient.need_ddyn_dhypers(),
            **minimize_kwargs
        )
        return optim_dict

    def set_problem(self, outer_objective, optim_dict, outer_objective_optimizer,
                    hyper_list=None, global_step=None):
        hyper_list = self._hypergradient.compute_gradients(outer_objective, optim_dict, hyper_list=hyper_list)
        self._h_optim_dict[outer_objective_optimizer].update(hyper_list)
        self._global_step = global_step
        return self

    def minimize(self, outer_objective, outer_objective_optimizer, inner_objective,  inner_objective_optimizer,
                 hyper_list=None, var_list=None, global_step=None):
        optim_dict = self.set_dynamics(inner_objective, inner_objective_optimizer, var_list)
        self.set_problem(outer_objective, optim_dict, outer_objective_optimizer, hyper_list, global_step)
        return self.finalize()

    def finalize(self):
        """
        To be called when no more dynamics or problems are added, computes the updates
        for the hyperparameters. This is to behave nicely with global_variables_initializer.
        :return:
        """
        self._hyperit()
        return self.run

    @property
    def hypergradient(self):
        return self._hypergradient

    def _hyperit(self):  # TODO add names
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
        self._hypergradient.run(T_or_generator, inner_objective_feed_dicts, outer_objective_feed_dicts,
                                initializer_feed_dict, session=session, global_step=self._global_step)
        if not _skip_hyper_ts:
            ss = session or tf.get_default_session()
            ss.run(self._hyperit())

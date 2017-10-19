import tensorflow as tf

from far_ho.optimizer import Optimizer, OptimizerDict
from far_ho.hypergradients import ReverseHg, HyperGradient
from far_ho.utils import GraphKeys


def get_hyperparameter(name, initializer=None, shape=None, dtype=None):
    return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                           collections=[GraphKeys.HYPERPARAMETERS,
                                        GraphKeys.GLOBAL_VARIABLES])


class HyperOptimizer:
    def __init__(self, hypergradient_cls=None):
        assert hypergradient_cls is None or issubclass(hypergradient_cls, HyperGradient)
        self._hypergradient_cls = hypergradient_cls or ReverseHg
        self._hypergradient = None
        # self._optim_dict = None
        self._hts = []
        self._fin_hts = None
        self._global_step = None

    def set_dynamics(self, inner_objective_optimizer, inner_objective, var_list=None,
                     **minimize_kwargs):
        # TODO should be possible to call this multiple times for parallel evaluation...
        # or maybe not really since you can just make a big dynamics if oyu want parallel
        # execution...
        assert isinstance(inner_objective_optimizer, Optimizer)
        optim_dict = inner_objective_optimizer.minimize(
            inner_objective,
            var_list=var_list,
            compute_ddyn_dhypers=self._hypergradient_cls.need_ddyn_dhypers(),
            **minimize_kwargs
        )
        self._hypergradient = self._hypergradient_cls(optim_dict)
        return self

    def set_problem(self, outer_objective_optimizer, outer_objective, hyper_list=None,
                    global_step=None):  # shout work with multiple calls
        # (different objective functions for different hyperparameters....)
        # but must also implement multiple calls of set dynamics (for e.g. mini-batches of episodes)

        self._hypergradient.compute_gradients(outer_objective, hyper_list=hyper_list)
        #   FIXME I don't think this works with multiple calls
        ts_or_optim_dict = outer_objective_optimizer.apply_gradients(self._hypergradient.hgrads_hvars(hyper_list))
        # prev line should be called just once
        self._hts.append(
            ts_or_optim_dict.ts if isinstance(ts_or_optim_dict, OptimizerDict) else ts_or_optim_dict
        )
        self._global_step = global_step
        return self

    @property
    def _hyperit(self):  # TODO add names
        if self._fin_hts is None:
            self._fin_hts = tf.group(*self._hts)
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
            ss.run(self._hyperit)

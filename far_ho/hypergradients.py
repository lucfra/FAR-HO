from collections import defaultdict

from tensorflow.python.training import slot_creator
import tensorflow as tf
from far_ho import utils
from far_ho.optimizer import OptimizerDict


class HyperGradient:
    def __init__(self, optimizer_dict):
        assert isinstance(optimizer_dict, OptimizerDict)
        self._optimizer_dict = optimizer_dict

    def compute_gradients(self, outer_objective, hyper_list=None):
        # """
        # Returns variables that store the values of the hypergradients
        #
        # :param outer_objective: a loss function for the hyperparameters
        # :param hyper_list: list of hyperparameters. If `None`
        # :return: list of pairs (hyperparameter, hypergradient) to be used with the method apply gradients!
        # """
        pass

    @staticmethod
    def need_ddyn_dhypers():
        pass

    def hgrads_hvars(self):
        return None

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None):
        pass


def _maybe_add(a, b):
    return a if b is None else a + b


def _val_or_zero(a, b):
    return a if a is not None else tf.zeros_like(b)


def _remove_from_global_variables(lst):
    """
    Remove variables form GLOBAL_VARIABLES to prevent initialization with tf.global_variables_initializer
    :param lst: a list of variables
    :return: None
    """
    # noinspection PyProtectedMember
    [tf.get_default_graph()._collections[utils.GraphKeys.GLOBAL_VARIABLES].remove(e) for e in lst]


class ReverseHg(HyperGradient):

    def __init__(self, optimizer_dict, history=None):
        super().__init__(optimizer_dict)
        self._hypergrad_dictionary = defaultdict(lambda: [])
        self._alpha_iter = tf.no_op()
        self._reverse_initializer = tf.no_op()
        self._history = history or []
        self._computed_hgrads_hvars = None

    # noinspection SpellCheckingInspection
    def compute_gradients(self, outer_objective, hyper_list=None):
        """
        Returns variables that store the values of the hypergradients

        :param outer_objective: a loss function for the hyperparameters
        :param hyper_list: list of hyperparameters. If `None`
        :return: list of pairs (hyperparameter, hypergradient) to be used with the method apply gradients!
        """
        scope = tf.get_variable_scope()
        # get hyperparameters
        if hyper_list is None:
            hyper_list = utils.hyperparameters(scope.name)

        # derivative of outer objective w.r.t. state
        doo_ds = tf.gradients(outer_objective, self._optimizer_dict.state)

        alphas = self._create_lagrangian_multipliers(doo_ds)

        alpha_vec = utils.vectorize_all(alphas)
        dyn_vec = utils.vectorize_all([d for (s, d) in self._optimizer_dict.dynamics])
        lag_part1 = utils.dot(alpha_vec, dyn_vec, name='iter_wise_lagrangian_part1')
        # TODO outer_objective might be a list... handle this case

        # iterative computation of hypergradients
        dlag_dhypers = tf.gradients(lag_part1, hyper_list)
        doo_dypers = tf.gradients(outer_objective, hyper_list)  # (direct) derivative of outer objective w.r.t. hyp.
        hyper_grad_vars = self._create_hypergradient(hyper_list, doo_dypers)
        hyper_grad_step = tf.group(*[hgv.assign(hgv + dl_dh) for hgv, dl_dh in
                                     zip(hyper_grad_vars, dlag_dhypers)])

        with tf.control_dependencies([hyper_grad_step]):  # first update hypergradinet then alphas.
            _alpha_iter = tf.group(*[alpha.assign(dl_ds) for alpha, dl_ds
                                   in zip(alphas, tf.gradients(lag_part1, self._optimizer_dict.state))])
        self._alpha_iter = tf.group(self._alpha_iter, _alpha_iter)

        # self._hgrad_hvar = (hgv, h) for h, hgv in zip(hyper_list, hyper_grad_vars)]
        [self._hypergrad_dictionary[h].append(hg) for h, hg in zip(hyper_list, hyper_grad_vars)]

        self._reverse_initializer = tf.group(self._reverse_initializer,
                                             tf.variables_initializer(alphas + hyper_grad_vars))

        return self

    def hgrads_hvars(self, aggregation_op=None, hyper_list=None):
        if hyper_list is None:
            hyper_list = utils.hyperparameters(tf.get_variable_scope().name)

        print(hyper_list)
        print()
        print(self._hypergrad_dictionary)
        print()
        print([h in self._hypergrad_dictionary for h in hyper_list])
        assert all([h in self._hypergrad_dictionary for h in hyper_list]), 'FINAL ERROR!'

        if self._computed_hgrads_hvars is None:
            if aggregation_op is None:
                aggregation_op = lambda hgrad_list: tf.reduce_mean(hgrad_list, axis=0)
            self._computed_hgrads_hvars = [(aggregation_op(self._hypergrad_dictionary[h]), h) for h in hyper_list]

        return self._computed_hgrads_hvars

    def _create_lagrangian_multipliers(self, doo_ds):
        lag_mul = [slot_creator.create_slot(v, _val_or_zero(der, v), 'alpha') for v, der
                   in zip(self._optimizer_dict.state, doo_ds)]
        [tf.add_to_collection(utils.GraphKeys.LAGRANGIAN_MULTIPLIERS, lm) for lm in lag_mul]
        _remove_from_global_variables(lag_mul)
        return lag_mul

    @staticmethod
    def _create_hypergradient(hyper_list, doo_dhypers):
        hgs = [slot_creator.create_slot(h, _val_or_zero(doo_dh, h), 'hypergradient') for h, doo_dh
               in zip(hyper_list, doo_dhypers)]
        [tf.add_to_collection(utils.GraphKeys.HYPERGRADIENTS, hg) for hg in hgs]
        _remove_from_global_variables(hgs)
        return hgs

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None):

        ss = session or tf.get_default_session()

        self._history.clear()
        self._history.append(ss.run(self._optimizer_dict.initialization, feed_dict=initializer_feed_dict))
        for t in range(T_or_generator) if isinstance(T_or_generator, int) else T_or_generator:
            self._history.append(ss.run(self._optimizer_dict.iteration,
                                        feed_dict=utils.maybe_call(inner_objective_feed_dicts, t)))
        # initialization of support variables (supports stochastic evaluation of outer objective via global_step
        # variable
        ss.run(self._reverse_initializer, feed_dict=utils.maybe_call(outer_objective_feed_dicts,
                                                                     utils.maybe_eval(global_step, ss)))
        for pt, state_feed_dict in self._optimizer_dict.state_feed_dict_generator(reversed(self._history[:-1])):
            t = len(self._history) - pt - 2  # if T is int then len(self.history) is T + 1 and this numerator
            # shall start at T-1  (99.99 sure its correct)
            ss.run(self._alpha_iter, feed_dict=utils.merge_dicts(
                state_feed_dict,
                utils.maybe_call(inner_objective_feed_dicts, t) if inner_objective_feed_dicts else {}))

    @staticmethod
    def need_ddyn_dhypers():
        return False

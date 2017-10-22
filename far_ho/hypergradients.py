from collections import defaultdict

import tensorflow as tf
from tensorflow.python.training import slot_creator

from far_ho import utils
from far_ho.optimizer import OptimizerDict


class HyperGradient:
    def __init__(self):
        self._optimizer_dicts = set()
        self._hypergrad_dictionary = defaultdict(lambda: [])

    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        # """
        # Returns variables that store the values of the hypergradients
        #
        # :param outer_objective: a loss function for the hyperparameters
        # :param hyper_list: list of hyperparameters. If `None`
        # :return: list of pairs (hyperparameter, hypergradient) to be used with the method apply gradients!
        # """
        raise NotImplementedError()

    @staticmethod
    def need_ddyn_dhypers():
        raise NotImplementedError()

    @property
    def initialization(self):
        return utils.flatten_list([opt_dict.initialization for opt_dict in sorted(self._optimizer_dicts)])

    @property
    def iteration(self):
        return utils.flatten_list([opt_dict.iteration for opt_dict in sorted(self._optimizer_dicts)])

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None):
        raise NotImplementedError()

    def hgrads_hvars(self, aggregation_op=None, hyper_list=None):
        if hyper_list is None:
            hyper_list = utils.hyperparameters(tf.get_variable_scope().name)

        assert all([h in self._hypergrad_dictionary for h in hyper_list]), 'FINAL ERROR!'

        if aggregation_op is None:
            aggregation_op = lambda hgrad_list: tf.reduce_mean(hgrad_list, axis=0)

        def _aggregate_and_manage_collection(_hg_lst):
            if len(_hg_lst) == 1:  # avoid useless operations...
                aggr = _hg_lst[0]
            else:
                with tf.name_scope(_hg_lst[0].name.split(':')[0]):
                    aggr = aggregation_op(_hg_lst) if len(_hg_lst) > 1 else _hg_lst[0]
            tf.add_to_collection(utils.GraphKeys.HYPERGRADIENTS, aggr)
            return aggr

        return [(_aggregate_and_manage_collection(self._hypergrad_dictionary[h]),
                 h) for h in hyper_list]


class ReverseHg(HyperGradient):

    def __init__(self, history=None):
        super().__init__()
        self._alpha_iter = tf.no_op()
        self._reverse_initializer = tf.no_op()
        self._history = history or []

    # noinspection SpellCheckingInspection
    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        """
        Returns variables that store the values of the hypergradients

        :param optimizer_dict:
        :param outer_objective: a loss function for the hyperparameters
        :param hyper_list: list of hyperparameters. If `None`
        :return: list of hyperparameters
        """
        assert isinstance(optimizer_dict, OptimizerDict)
        self._optimizer_dicts.add(optimizer_dict)

        scope = tf.get_variable_scope()
        # get hyperparameters
        if hyper_list is None:
            hyper_list = utils.hyperparameters(scope.name)

        # derivative of outer objective w.r.t. state
        with tf.ops.colocate_with(optimizer_dict.state[0]):  # this should put all the relevant ops in
            # the right gpu... I presume...
            doo_ds = tf.gradients(outer_objective, optimizer_dict.state)

            alphas = self._create_lagrangian_multipliers(optimizer_dict, doo_ds)

            alpha_vec = utils.vectorize_all(alphas)
            dyn_vec = utils.vectorize_all([d for (s, d) in optimizer_dict.dynamics])
            lag_part1 = utils.dot(alpha_vec, dyn_vec, name='iter_wise_lagrangian_part1')
            # TODO outer_objective might be a list... handle this case

            # iterative computation of hypergradients
            doo_dypers = tf.gradients(outer_objective, hyper_list)  # (direct) derivative of outer objective w.r.t. hyp.
            hyper_grad_vars = self._create_hypergradient(hyper_list, doo_dypers)

            dlag_dhypers = tf.gradients(lag_part1, hyper_list)
            hyper_grad_step = tf.group(*[hgv.assign(hgv + dl_dh) for hgv, dl_dh in
                                         zip(hyper_grad_vars, dlag_dhypers)])

            with tf.control_dependencies([hyper_grad_step]):  # first update hypergradinet then alphas.
                _alpha_iter = tf.group(*[alpha.assign(dl_ds) for alpha, dl_ds
                                         in zip(alphas, tf.gradients(lag_part1, optimizer_dict.state))])
            self._alpha_iter = tf.group(self._alpha_iter, _alpha_iter)

            [self._hypergrad_dictionary[h].append(hg) for h, hg in zip(hyper_list, hyper_grad_vars)]

            self._reverse_initializer = tf.group(self._reverse_initializer,
                                                 tf.variables_initializer(alphas + hyper_grad_vars))

            return hyper_list

    @staticmethod
    def _create_lagrangian_multipliers(optimizer_dict, doo_ds):
        lag_mul = [slot_creator.create_slot(v, utils.val_or_zero(der, v), 'alpha') for v, der
                   in zip(optimizer_dict.state, doo_ds)]
        [tf.add_to_collection(utils.GraphKeys.LAGRANGIAN_MULTIPLIERS, lm) for lm in lag_mul]
        utils.remove_from_collection(utils.GraphKeys.GLOBAL_VARIABLES, *lag_mul)
        # this prevents the 'automatic' initialization with tf.global_variables_initializer.
        return lag_mul

    @staticmethod
    def _create_hypergradient(hyper_list, doo_dhypers):
        hgs = [slot_creator.create_slot(h, utils.val_or_zero(doo_dh, h), 'hypergradient') for h, doo_dh
               in zip(hyper_list, doo_dhypers)]
        utils.remove_from_collection(utils.GraphKeys.GLOBAL_VARIABLES, *hgs)
        return hgs

    def state_feed_dict_generator(self, history):
        state = utils.flatten_list([opt_dict.state for opt_dict in sorted(self._optimizer_dicts)])
        for t, his in enumerate(history):
            yield t, {v: his[k] for k, v in enumerate(state)}

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None):

        ss = session or tf.get_default_session()

        self._history.clear()
        self._history.append(ss.run(self.initialization, feed_dict=initializer_feed_dict))
        for t in range(T_or_generator) if isinstance(T_or_generator, int) else T_or_generator:
            self._history.append(ss.run(self.iteration,
                                        feed_dict=utils.maybe_call(inner_objective_feed_dicts, t)))
        # initialization of support variables (supports stochastic evaluation of outer objective via global_step
        # variable
        ss.run(self._reverse_initializer, feed_dict=utils.maybe_call(outer_objective_feed_dicts,
                                                                     utils.maybe_eval(global_step, ss)))
        for pt, state_feed_dict in self.state_feed_dict_generator(reversed(self._history[:-1])):
            t = len(self._history) - pt - 2  # if T is int then len(self.history) is T + 1 and this numerator
            # shall start at T-1  (99.99 sure its correct)
            ss.run(self._alpha_iter, feed_dict=utils.merge_dicts(
                state_feed_dict,
                utils.maybe_call(inner_objective_feed_dicts, t) if inner_objective_feed_dicts else {}))

    @staticmethod
    def need_ddyn_dhypers():
        return False

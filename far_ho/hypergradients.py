from collections import defaultdict

import tensorflow as tf
from tensorflow.python.training import slot_creator

from far_ho import utils
from far_ho.optimizer import OptimizerDict


class HyperGradient:
    def __init__(self):
        self._optimizer_dicts = set()
        self._hypergrad_dictionary = defaultdict(lambda: [])  # dictionary (hyperarameter, list of hypergradients)

    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        # """
        # Returns variables that store the values of the hypergradients
        #
        # :param outer_objective: a loss function for the hyperparameters
        # :param hyper_list: list of hyperparameters. If `None`
        # :return: list of pairs (hyperparameter, hypergradient) to be used with the method apply gradients!
        # """
        assert isinstance(optimizer_dict, OptimizerDict)
        self._optimizer_dicts.add(optimizer_dict)

        scope = tf.get_variable_scope()
        # get hyperparameters
        if hyper_list is None:
            hyper_list = utils.hyperparameters(scope.name)
        return hyper_list

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
                with tf.name_scope(_hg_lst[0].op.name):
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
        hyper_list = super().compute_gradients(outer_objective, optimizer_dict, hyper_list)

        # derivative of outer objective w.r.t. state
        with tf.variable_scope(outer_objective.op.name):
            doo_ds = tf.gradients(outer_objective, optimizer_dict.state)

            alphas = self._create_lagrangian_multipliers(optimizer_dict, doo_ds)

            alpha_vec = utils.vectorize_all(alphas)
            dyn_vec = utils.vectorize_all([d for (s, d) in optimizer_dict.dynamics])
            lag_phi_t = utils.dot(alpha_vec, dyn_vec, name='iter_wise_lagrangian_part1')
            # TODO outer_objective might be a list... handle this case

            # iterative computation of hypergradients
            doo_dypers = tf.gradients(outer_objective, hyper_list)  # (direct) derivative of outer objective w.r.t. hyp.
            alpha_dot_B = tf.gradients(lag_phi_t, hyper_list)
            # check that optimizer_dict has initial ops (phi_0)
            if optimizer_dict.init_dynamics is not None:
                lag_phi0 = utils.dot(alpha_vec, utils.vectorize_all([d for (s, d) in optimizer_dict.init_dynamics]))
                alpha_dot_B0 = tf.gradients(lag_phi0, hyper_list)
            else:
                alpha_dot_B0 = [None]*len(hyper_list)

            # here is some of this is None it may mean that the hyperparameter compares inside phi_0: check that and
            # if it is not the case return error...
            hyper_grad_vars, hyper_grad_step = [], tf.no_op()
            for dl_dh, doo_dh, a_d_b0, hyper in zip(alpha_dot_B, doo_dypers, alpha_dot_B0, hyper_list):
                assert dl_dh is not None or a_d_b0 is not None, 'Hyperparameter %s is detached from ' \
                                                                'this dyamics' % hyper
                hgv = None
                if dl_dh is not None:  # "normal hyperparameter"
                    hgv = self._create_hypergradient(hyper, doo_dh)

                    hyper_grad_step = tf.group(hyper_grad_step, hgv.assign_add(dl_dh))
                if a_d_b0 is not None:
                    hgv = hgv + a_d_b0 if hgv is not None else a_d_b0
                    # here hyper_grad_step has nothing to do...
                hyper_grad_vars.append(hgv)  # save these...

            with tf.control_dependencies([hyper_grad_step]):  # first update hypergradinet then alphas.
                _alpha_iter = tf.group(*[alpha.assign(dl_ds) for alpha, dl_ds
                                         in zip(alphas, tf.gradients(lag_phi_t, optimizer_dict.state))])
            self._alpha_iter = tf.group(self._alpha_iter, _alpha_iter)

            [self._hypergrad_dictionary[h].append(hg) for h, hg in zip(hyper_list, hyper_grad_vars)]

            self._reverse_initializer = tf.group(self._reverse_initializer,
                                                 tf.variables_initializer(alphas),
                                                 tf.variables_initializer([h for h in hyper_grad_vars
                                                                           if hasattr(h, 'initializer')]))  # some ->
            # hypergradients might be just tensors and not variables...

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
    def _create_hypergradient(hyper, doo_dhypers):
        """
        Creates one hyper-gradient as a variable..

        :param hyper:  the relative hyperparameter
        :param doo_dhypers:  initialization, that is the derivative of the outer objective w.r.t this hyper
        :return:
        """
        hgs = slot_creator.create_slot(hyper, utils.val_or_zero(doo_dhypers, hyper), 'hypergradient')
        utils.remove_from_collection(utils.GraphKeys.GLOBAL_VARIABLES, hgs)
        return hgs

    def _state_feed_dict_generator(self, history):
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
        for pt, state_feed_dict in self._state_feed_dict_generator(reversed(self._history[:-1])):
            t = len(self._history) - pt - 2  # if T is int then len(self.history) is T + 1 and this numerator
            # shall start at T-1  (99.99 sure its correct)
            ss.run(self._alpha_iter, feed_dict=utils.merge_dicts(
                state_feed_dict,
                utils.maybe_call(inner_objective_feed_dicts, t) if inner_objective_feed_dicts else {}))


class ForwardHG(HyperGradient):
    def __init__(self):
        super().__init__()
        self._forw_initializer = tf.no_op()
        self._z_iter = tf.no_op()

    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        hyper_list = super().compute_gradients(outer_objective, optimizer_dict, hyper_list)

        return hyper_list

    # @staticmethod
    # def _create_z(optimizer_dict, hyper):
    #     zs = [slot_creator.create_slot(v, tf.zero(der, v), 'z_%s' % hyper.op.name) for v, der
    #                in zip(optimizer_dict.state)]
    #     [tf.add_to_collection(utils.GraphKeys.LAGRANGIAN_MULTIPLIERS, lm) for lm in lag_mul]
    #     utils.remove_from_collection(utils.GraphKeys.GLOBAL_VARIABLES, *lag_mul)
    #     # this prevents the 'automatic' initialization with tf.global_variables_initializer.
    #     return lag_mul

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None):
        pass

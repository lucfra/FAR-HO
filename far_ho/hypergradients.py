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
        # Doesn't do anything useful here. To be overridden.
        """
        Function overridden by specific methods.

        :param optimizer_dict: OptimzerDict object resulting from the inner objective optimization.
        :param outer_objective: A loss function for the hyperparameters (scalar tensor)
        :param hyper_list: Optional list of hyperparameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.

        :return: list of hyperparameters involved in the computation
        """
        assert isinstance(optimizer_dict, OptimizerDict)
        self._optimizer_dicts.add(optimizer_dict)

        if hyper_list is None:  # get default hyperparameters
            hyper_list = utils.hyperparameters(tf.get_variable_scope().name)
        return hyper_list

    @property
    def initialization(self):
        return utils.flatten_list([opt_dict.initialization for opt_dict in sorted(self._optimizer_dicts)])

    @property
    def iteration(self):
        return utils.flatten_list([opt_dict.iteration for opt_dict in sorted(self._optimizer_dicts)])

    @property
    def ts(self):
        return tf.group(*[opt_dict.ts for opt_dict in sorted(self._optimizer_dicts)])

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False):
        raise NotImplementedError()

    def hgrads_hvars(self, hyper_list=None, aggregation_fn=None, process_fn=None):
        """
        Method for getting hypergradient and hyperparameters as required by apply_gradient methods from tensorflow 
        optimizers.
        
        :param hyper_list: Optional list of hyperparameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.
        :param aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
                                by default reduce_mean
        :param process_fn: Optional operation like clipping to be applied.
        :return: 
        """
        if hyper_list is None:
            hyper_list = utils.hyperparameters(tf.get_variable_scope().name)

        assert all([h in self._hypergrad_dictionary for h in hyper_list]), 'FINAL ERROR!'

        if aggregation_fn is None:
            aggregation_fn = lambda hgrad_list: tf.reduce_mean(hgrad_list, axis=0)

        def _aggregate_process_manage_collection(_hg_lst):
            if len(_hg_lst) == 1:  # avoid useless operations...
                aggr = _hg_lst[0]
            else:
                with tf.name_scope(_hg_lst[0].op.name):
                    aggr = aggregation_fn(_hg_lst) if len(_hg_lst) > 1 else _hg_lst[0]
            if process_fn is not None:
                with tf.name_scope('process_gradients'):
                    aggr = process_fn(aggr)
            tf.add_to_collection(utils.GraphKeys.HYPERGRADIENTS, aggr)
            return aggr

        return [(_aggregate_process_manage_collection(self._hypergrad_dictionary[h]),
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
        Function that adds to the computational graph all the operations needend for computing
        the hypergradients in a "dynamic" way, without unrolling the entire optimization graph.
        The resulting computation, while being roughly 2x more expensive then unrolling the
        optimizaiton dynamics, requires much less (GPU) memory and is more flexible, allowing
        to set a termination condition to the parameters optimizaiton routine.

        :param optimizer_dict: OptimzerDict object resulting from the inner objective optimization.
        :param outer_objective: A loss function for the hyperparameters (scalar tensor)
        :param hyper_list: Optional list of hyperparameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.

        :return: list of hyperparameters involved in the computation
        """
        hyper_list = super().compute_gradients(outer_objective, optimizer_dict, hyper_list)

        # derivative of outer objective w.r.t. state
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
            with tf.name_scope('Phi_0_hypergradient'):
                lag_phi0 = utils.dot(alpha_vec, utils.vectorize_all([d for (s, d) in optimizer_dict.init_dynamics]),
                                     name='lagrangian_Phi0')
                alpha_dot_B0 = tf.gradients(lag_phi0, hyper_list)
        else:
            alpha_dot_B0 = [None] * len(hyper_list)

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
            if a_d_b0 is not None:  # add or set hyper-gradient of Phi_0 (initial dynamics)
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
            # hypergradients (those coming form initial dynamics) might be just tensors and not variables...

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
            initializer_feed_dict=None, global_step=None, session=None, online=False):

        ss = session or tf.get_default_session()

        self._history.clear()
        if not online:
            self._history.append(ss.run(self.initialization, feed_dict=utils.maybe_call(
                initializer_feed_dict, utils.maybe_eval(global_step, ss))))

        for t in range(T_or_generator) if isinstance(T_or_generator, int) else T_or_generator:
            self._history.append(ss.run(self.iteration,
                                        feed_dict=utils.maybe_call(inner_objective_feed_dicts, t)))
        # initialization of support variables (supports stochastic evaluation of outer objective via global_step ->
        # variable)
        ss.run(self._reverse_initializer, feed_dict=utils.maybe_call(outer_objective_feed_dicts,
                                                                     utils.maybe_eval(global_step, ss)))
        for pt, state_feed_dict in self._state_feed_dict_generator(reversed(self._history[:-1])):
            # this should be fine also for truncated reverse... but check again the index t
            t = len(self._history) - pt - 2  # if T is int then len(self.history) is T + 1 and this numerator
            # shall start at T-1  (99.99 sure its correct)
            ss.run(self._alpha_iter,
                   feed_dict=utils.merge_dicts(state_feed_dict,
                                               utils.maybe_call(inner_objective_feed_dicts, t)
                                               if inner_objective_feed_dicts else {}))


class UnrolledReverseHG(HyperGradient):
    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False):
        return NotImplemented()

    # maybe... it would require a certain effort...


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
            initializer_feed_dict=None, global_step=None, session=None, online=False):
        pass

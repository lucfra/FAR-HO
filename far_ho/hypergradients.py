from tensorflow.python.training import slot_creator
import tensorflow as tf
from far_ho import utils


class HyperGradient:
    def __init__(self, optimizer_dict):
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


def _maybe_add(a, b):
    return a if b is None else a + b


def _val_or_zero(a):
    return a if a is not None else tf.zeros_like(a)


def _remove_from_global_variables(lst):
    """
    Remove variables form GLOBAL_VARIABLES to prevent initialization with tf.global_variables_initializer
    :param lst: a list of variables
    :return: None
    """
    # noinspection PyProtectedMember
    [tf.get_default_graph()._collections[utils.GraphKeys.GLOBAL_VARIABLES].remove(e) for e in lst]


class ReverseHg(HyperGradient):
    def __init__(self, optimizer_dict, history=None, _is_derived=False):
        super().__init__(optimizer_dict)
        self._hyper_and_hyper_grad = None
        self._alpha_iter = None  # these could become dictionaries
        self._reverse_initializer = None
        self._history = history or []
        self._is_derived = _is_derived
        # flag that if true signals that this object has an ancestor  (maybe put this thing into a graph). Note
        # that the run of the not_derived should be executed first!

        # self._s_his = []
        # with tf.name_scope('s_history', values=[v for (v, d) in dynamics]):
        #     self._s_placeholders = [tf.placeholder(v.dtype, v.get_shape()) for (v, d) in dynamics]
        #     self._s_revert = tf.group(*[v.assign(his) for ((v, d), his) in zip(dynamics, self._s_placeholders)])

    # noinspection SpellCheckingInspection
    def compute_gradients(self, outer_objective, hyper_list=None):
        """
        Returns variables that store the values of the hypergradients

        :param outer_objective: a loss function for the hyperparameters
        :param hyper_list: list of hyperparameters. If `None`
        :return: list of pairs (hyperparameter, hypergradient) to be used with the method apply gradients!
        """
        scope = tf.get_variable_scope()
        if self._alpha_iter is not None:  # multiple outer objectives!
            return ReverseHg(self._optimizer_dict, self._history, _is_derived=True).compute_gradients(
                outer_objective, hyper_list=hyper_list)
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

        # TODO HERE MIGHT WANT TO to the resulting computations inside a dictionary
        with tf.control_dependencies([hyper_grad_step]):  # first update hypergradinet then alphas.
            self._alpha_iter = tf.group(*[alpha.assign(dl_ds) for alpha, dl_ds
                                    in zip(alphas, tf.gradients(lag_part1, self._optimizer_dict.state))])
        self._hyper_and_hyper_grad = [(h, hgv) for h, hgv in (hyper_list, hyper_grad_vars)]

        self._reverse_initializer = tf.variables_initializer(utils.lagrangian_multipliers(scope.name) +
                                                             utils.hypergradients(scope.name))

        return self

        # return self._hyper_and_hyper_grad, self._alpha_iter
        # TODO this will not work because of initialization... must find a better way...

    @property
    def hypers_and_hypergrads(self):
        return self._hyper_and_hyper_grad

    def _create_lagrangian_multipliers(self, doo_ds):
        lag_mul = [slot_creator.create_slot(v, _val_or_zero(v), 'alpha') for ((v, d), der)
                   in zip(self._optimizer_dict, doo_ds)]
        [tf.add_to_collection(utils.GraphKeys.LAGRANGIAN_MULTIPLIERS, lm) for lm in lag_mul]
        _remove_from_global_variables(lag_mul)
        return lag_mul

    @staticmethod
    def _create_hypergradient(hyper_list, doo_dhypers):
        hgs = [slot_creator.create_slot(h, _val_or_zero(doo_dh), 'hypergradient') for h, doo_dh
                in zip(hyper_list, doo_dhypers)]
        [tf.add_to_collection(utils.GraphKeys.HYPERGRADIENTS, hg) for hg in hgs]
        _remove_from_global_variables(hgs)
        return hgs

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None):
        # forward
        # tf.Session().run()
        ss = session or tf.get_default_session()

        assert self._is_derived is False or self._history != [], 'Run first the ReverseHg instance that was ' \
                                                                 'created at first!'

        if self._is_derived is False:
            self._history.clear()
            self._history.append(ss.run(self._optimizer_dict.initialize, feed_dict=initializer_feed_dict))
            for t in range(T_or_generator) if isinstance(T_or_generator, int) else T_or_generator:
                self._history.append(ss.run(self._optimizer_dict.iteration,
                                            feed_dict=utils.maybe_call(inner_objective_feed_dicts, t)))
        # initialization of support variables (supports stochastic evaluation of outer objective via global_step
        # variable
        ss.run(self._reverse_initializer, feed_dict=utils.maybe_call(outer_objective_feed_dicts,
                                                           ss.run(global_step) if global_step else None))
        for pt, state_feed_dict in self._optimizer_dict.state_feed_dict_generator(reversed(self._history[:-1])):
            t = len(self._history) - pt - 2  # if T is int then len(self.history) is T + 1 and this numerator
            # shall start at T-1  (99.99 sure its correct)
            ss.run(self._alpha_iter, feed_dict=utils.merge_dicts(
                state_feed_dict,
                utils.maybe_call(inner_objective_feed_dicts, t) if inner_objective_feed_dicts else {}))

# def compute_hypergradient()

from tensorflow.python.training import slot_creator
import tensorflow as tf
from far import utils


class HyperGradient:
    def __init__(self, dynamics):
        self._dynamics = dynamics

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


class ReverseHg(HyperGradient):
    def __init__(self, dynamics):
        super().__init__(dynamics)

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
        # get hyperparameters
        if hyper_list is None:
            hyper_list = tf.get_collection(utils.GraphKeys.HYPERPARAMETERS)

        # derivative of outer objective w.r.t. state
        doo_ds = tf.gradients(outer_objective, [v for (v, w) in self._dynamics])

        alphas = self._create_lagrangian_multipliers(doo_ds)

        alpha_vec = utils.vectorize_all(alphas)
        dyn_vec = utils.vectorize_all([d for (s, d) in self._dynamics])
        lag_part1 = utils.dot(alpha_vec, dyn_vec, name='iter_wise_lagrangian_part1')
        # TODO outer_objective might be a list... handle this case

        # iterative computation of hypergradients
        dlag_dhypers = tf.gradients(lag_part1, hyper_list)
        doo_dypers = tf.gradients(outer_objective, hyper_list)  # (direct) derivative of outer objective w.r.t. hyp.
        hyper_grad_vars = self._create_hypergradient(hyper_list, doo_dypers)
        hyper_grad_step = tf.group(*[hgv.assign(hgv + dl_dh) for hgv, dl_dh in
                                     zip(hyper_grad_vars, dlag_dhypers)])

        with tf.control_dependencies([hyper_grad_step]):  # first update hypergradinet then alphas.
            alpha_step = tf.group(*[alpha.assign(dl_ds) for alpha, dl_ds
                                    in zip(alphas, tf.gradients(lag_part1, [v for (v, d) in self._dynamics]))])

        return [(h, hgv) for h, hgv in (hyper_list, hyper_grad_vars)], alpha_step
        # TODO this will not work because of initialization... must find a better way...

    def _create_lagrangian_multipliers(self, doo_ds):
        return [slot_creator.create_slot(v, _val_or_zero(v), 'alpha') for ((v, d), der)
                in zip(self._dynamics, doo_ds)]

    @staticmethod
    def _create_hypergradient(hyper_list, doo_dhypers):
        return [slot_creator.create_slot(h, _val_or_zero(doo_dh),
                                         'hypergradient') for h, doo_dh
                in zip(hyper_list, doo_dhypers)]

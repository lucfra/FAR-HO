import tensorflow as tf
from far.utils import GraphKeys


class Optimizer(tf.train.Optimizer):

    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                 aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None,
                 forward_hg=False, hyperparameters=None):
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
        :param forward_hg:
        :param hyperparameters:
        :return:
        """
        ts, dyn = super().minimize(loss, global_step, var_list, gate_gradients, aggregation_method,
                                colocate_gradients_with_ops, name, grad_loss)

        if not forward_hg:
            return ts, dyn
        else:
            hypers = hyperparameters or tf.get_collection(GraphKeys.HYPERPARAMETERS)
            assert all([hy.get_shape().ndims == 1 for hy in hypers]), 'only scalar hyperparameters for now'
            grad_and_hypers = self.compute_gradients(loss, hypers, gate_gradients, aggregation_method,
                                   colocate_gradients_with_ops)
            # TODO filter for algorithmic hyperparameters (grad would be None here!)
            d_dyn_d_hypers = [
                (tf.gradients(g, hypers, name='d_dyn_d_hypers'), v) for (g, v) in grad_and_hypers
            ]

            return ts, dyn, d_dyn_d_hypers


class MomentumOptimizer(tf.train.MomentumOptimizer):
    def __init__(self, learning_rate, momentum, use_locking=False, name="Momentum",
                 use_nesterov=False):
        super().__init__(learning_rate, momentum, use_locking, name, use_nesterov)

    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=Optimizer.GATE_OP,
                 aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None, grad_loss=None,
                 forward_hg=False):
        ts, dyn = super().minimize(loss, global_step, var_list, gate_gradients, aggregation_method,
                                colocate_gradients_with_ops, name, grad_loss)
        if not forward_hg:
            return ts, dyn
        else:
            hypers = tf.get_collection('hyper')
            assert all([hy.get_shape().ndims == 1 for hy in hypers]), 'only scalar hyperparameters for now'
            grad_and_hypers = self.compute_gradients(loss, hypers, gate_gradients, aggregation_method,
                                   colocate_gradients_with_ops)
            # TODO filter for algorithmic hyperparameters (grad would be None here!)
            d_dyn_d_hypers = [
                (tf.gradients(g, hypers, name='d_dyn_d_hypers'), v) for (g, v) in grad_and_hypers
            ]

            return ts, dyn, d_dyn_d_hypers
    #     def compute_

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

# variables.PartitionedVariable

from __future__ import absolute_import, print_function, division

from collections import defaultdict

import sys
import tensorflow as tf
from tensorflow.python.training import slot_creator

from far_ho import utils
from far_ho.utils import dot, maybe_add, reduce_all_sums
from far_ho.optimizer import OptimizerDict

RAISE_ERROR_ON_DETACHED = False


class HyperGradient(object):
    def __init__(self):
        self._optimizer_dicts = set()
        self._hypergrad_dictionary = defaultdict(lambda: [])  # dictionary (hyperarameter, list of hypergradients)
        self._ts = None
        self.inner_losses = []

    _ERROR_NOT_OPTIMIZER_DICT = """
    Looks like {} is not an `OptimizerDict`. Use optimizers in far_ho.optimizers for obtaining an OptimizerDict.
    """

    _ERROR_HYPER_DETACHED = """
    Hyperparameter {} is detached from this optimization dynamics.
    """

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
        assert isinstance(optimizer_dict, OptimizerDict), HyperGradient._ERROR_NOT_OPTIMIZER_DICT.format(optimizer_dict)
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

    def _init_ts(self):
        return tf.group(*[opt_dict.ts for opt_dict in sorted(self._optimizer_dicts)])

    @property
    def ts(self):
        if self._ts is None:
            self._ts = self._init_ts()
        return self._ts

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, inner_objective=None):
        """
        Runs the inner optimization dynamics for T iterations (T_or_generator can be indeed a generator) and computes
        in the meanwhile.

        :param T_or_generator: integer or generator that should yield a step. Total number of iterations of
                                inner objective optimization dynamics
        :param inner_objective_feed_dicts: Optional feed dictionary for the inner objective
        :param outer_objective_feed_dicts: Optional feed dictionary for the outer objective
                                            (note that this is not used in ForwardHG since hypergradients are not
                                            variables)
        :param initializer_feed_dict: Optional feed dictionary for the inner objective
        :param global_step: Optional global step for the
        :param session: Optional session (otherwise will take the default session)
        :param online: Performs the computation of the hypergradient in the online (or "real time") mode. Note that
                        `ReverseHG` and `ForwardHG` behave differently.
        :param inner_objective: Tensor, inner objective will be evaluated for all the T iterations
                                and its values will be stored in self.inner_losses

        """
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
        super(ReverseHg, self).__init__()
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
        hyper_list = super(ReverseHg, self).compute_gradients(outer_objective, optimizer_dict, hyper_list)

        # derivative of outer objective w.r.t. state
        with tf.variable_scope(outer_objective.op.name):  # for some reason without this there is a cathastrofic
            # failure...
            doo_ds = tf.gradients(outer_objective, optimizer_dict.state)

            alphas = self._create_lagrangian_multipliers(optimizer_dict, doo_ds)

            alpha_vec = utils.vectorize_all(alphas)
            dyn_vec = utils.vectorize_all(optimizer_dict.dynamics)
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
                alpha_dot_B0 = [None] * len(hyper_list)

            # here is some of this is None it may mean that the hyperparameter compares inside phi_0: check that and
            # if it is not the case return error...
            hyper_grad_vars, hyper_grad_step = [], tf.no_op()
            for dl_dh, doo_dh, a_d_b0, hyper in zip(alpha_dot_B, doo_dypers, alpha_dot_B0, hyper_list):
                assert dl_dh is not None or a_d_b0 is not None, HyperGradient._ERROR_HYPER_DETACHED.format(hyper)
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
            self._alpha_iter = tf.group(self._alpha_iter, _alpha_iter)  # put all the backward iterations toghether

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
            initializer_feed_dict=None, global_step=None, session=None, online=False, inner_objective=None):

        ss = session or tf.get_default_session()

        self._history = []
        if not online:
            self.inner_losses = []
            self._history.append(ss.run(self.initialization, feed_dict=utils.maybe_call(
                initializer_feed_dict, utils.maybe_eval(global_step, ss))))

        for t in range(T_or_generator) if isinstance(T_or_generator, int) else T_or_generator:
            s_t, inner_loss = ss.run([self.iteration, inner_objective],
                                     feed_dict=utils.maybe_call(inner_objective_feed_dicts, t))

            self.inner_losses.append(inner_loss)
            self._history.append(s_t)

        #self.inner_losses.append(ss.run(inner_objective, feed_dict=utils.maybe_call(inner_objective_feed_dicts, t)))


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
            initializer_feed_dict=None, global_step=None, session=None, online=False, inner_objective=None):
        return NotImplemented()

        # maybe... it would require a certain effort...


class ForwardHG(HyperGradient):
    def __init__(self):
        super(ForwardHG, self).__init__()
        self._forward_initializer = tf.no_op()
        self._z_iter = tf.no_op()
        self._iteration = None
        self.A_dot_zs = {}

    _HYPER_RANK_ERROR_MESSAGE = """
    ForwardHG: Only scalar hyperparameters accepted.\n
     Hyperparameter tensor {} has rank {}.\n
     Use keyword argument far_ho.get_hyperparameter(..., scalar=True) on hyperparameter creation.
    """

    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        hyper_list = super(ForwardHG, self).compute_gradients(outer_objective, optimizer_dict, hyper_list)

        # scalar_hyper_list

        with tf.variable_scope(outer_objective.op.name):
            # dynamics_vec = vectorize_all(optimizer_dict.dynamics)  # in the new implementation there's no need of
            # vectorizing... it might be more efficient since it's better to avoid too many reshaping operations...
            d_oo_d_state = tf.gradients(outer_objective, optimizer_dict.state)

            # d_oo_d_state = [_v if _v is not None else tf.zeros_like(_s)
            # for _v, _s in zip(d_oo_d_state, optimizer_dict.state)]

            with tf.name_scope('DUMMY'):  # variables to compute forward propagation
                # TODO avoid this computation if optimizer_dict has already been seen.
                aux_v = [tf.zeros_like(v) for v in optimizer_dict.state]
                # aux_v_vec = vectorize_all(aux_v)
                # dynamics_dot_aux_v = dot(dynamics_vec, aux_v_vec)  # old impl
                dynamics_dot_aux_v = reduce_all_sums(optimizer_dict.dynamics, aux_v)

                der_dynamics_dot_aux_v = tf.gradients(dynamics_dot_aux_v, optimizer_dict.state)
                # this is a list of jacobians times aux_v that have the same dimension of states variables.

                init_dynamics_dot_aux_v = None
                if optimizer_dict.init_dynamics:
                    # init_dynamics_dot_aux_v = dot(vectorize_all(optimizer_dict.init_dynamics), aux_v_vec)  # old impl
                    init_dynamics_dot_aux_v = reduce_all_sums(
                        optimizer_dict.init_dynamics, aux_v)

            for hyp in hyper_list:
                assert hyp.shape.ndims == 0, ForwardHG._HYPER_RANK_ERROR_MESSAGE.format(hyp, hyp.shape.ndims)

                d_init_dyn_d_hyp = None if init_dynamics_dot_aux_v is None else \
                    tf.gradients(init_dynamics_dot_aux_v, hyp)[0]
                d_dyn_d_hyp = tf.gradients(dynamics_dot_aux_v, hyp)[0]
                d_oo_d_hyp = tf.gradients(outer_objective, hyp)[0]

                # ------------------------------------------------------------
                # check detached hyperparameters (for which hypergradient would be always null)
                hyper_ok = d_init_dyn_d_hyp is not None or d_dyn_d_hyp is not None or d_oo_d_hyp is not None
                if RAISE_ERROR_ON_DETACHED:
                    # try:
                    assert hyper_ok, HyperGradient._ERROR_HYPER_DETACHED.format(hyp)
                    # ex
                else:
                    if not hyper_ok:
                        print(HyperGradient._ERROR_HYPER_DETACHED.format(hyp), file=sys.stderr)
                        hyper_list.remove(hyp)
                # -------------------------------------------------------------

                # UPDATE OF TOTAL DERIVATIVE OF STATE W.R.T. HYPERPARAMETER
                zs = ForwardHG._create_z(
                    optimizer_dict, hyp, None if d_init_dyn_d_hyp is None else tf.gradients(d_init_dyn_d_hyp, aux_v)
                )
                # dyn_dot_zs = dot(dynamics_vec, vectorize_all(zs))
                Bs = tf.gradients(d_dyn_d_hyp, aux_v)  # this looks right...
                # A_dot_zs = tf.gradients(dyn_dot_zs, optimizer_dict.state)  # I guess the error is here!
                # the error is HERE! this operation computes d Phi/ d w * z for each w instead of d Phi_i / d s * z
                # for each i

                # A_dot_zs = tf.gradients(dot(vectorize_all(der_dynamics_dot_aux_v), vectorize_all(zs)), aux_v)  # old
                A_dot_zs = tf.gradients(reduce_all_sums(der_dynamics_dot_aux_v, zs), aux_v)

                self.A_dot_zs[hyp] = A_dot_zs

                _z_iter = tf.group(*[
                    z.assign(maybe_add(A_dot_z, B)) for z, A_dot_z, B
                    in zip(zs, A_dot_zs, Bs)
                ])
                self._z_iter = tf.group(self._z_iter, _z_iter)

                # HYPERGRADIENT
                # d_E_T = dot(vectorize_all(d_oo_d_state), vectorize_all(zs))
                d_E_T = [dot(d_oo_d_s, z) for d_oo_d_s, z in zip(d_oo_d_state, zs)
                         if d_oo_d_s is not None and z is not None]
                hg = maybe_add(tf.reduce_sum(d_E_T), d_oo_d_hyp)  # this is right... the error is not here!
                # hg = maybe_add(d_E_T, d_oo_d_hyp)

                self._hypergrad_dictionary[hyp].append(hg)

                self._forward_initializer = tf.group(self._forward_initializer,
                                                     tf.variables_initializer(zs))

        return hyper_list

    @staticmethod
    def _create_z(optimizer_dict, hyper, d_init_dynamics_d_hyper):
        if d_init_dynamics_d_hyper is None: d_init_dynamics_d_hyper = [None] * len(optimizer_dict.state)
        with tf.variable_scope('Z'):
            z = [slot_creator.create_slot(v, utils.val_or_zero(der, v), hyper.op.name) for v, der
                 in zip(optimizer_dict.state, d_init_dynamics_d_hyper)]
            [tf.add_to_collection(utils.GraphKeys.ZS, lm) for lm in z]
            # utils.remove_from_collection(utils.GraphKeys.GLOBAL_VARIABLES, *z)
            # in this case it is completely fine to keep zs into the global variable...
            return z

    # @property
    # def iteration(self):
    #     if self._iteration is None:
    #         with tf.control_dependencies([self._z_iter]):
    #             self._iteration = utils.flatten_list(
    # [opt_dict.iteration for opt_dict in sorted(self._optimizer_dicts)])
    #     return self._iteration

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, inner_objective=None):

        ss = session or tf.get_default_session()

        if not online:
            ss.run(self.initialization, feed_dict=utils.maybe_call(
                initializer_feed_dict, utils.maybe_eval(global_step, ss)))
            ss.run(self._forward_initializer)
            self.inner_losses = []

        for t in range(T_or_generator) if isinstance(T_or_generator, int) else T_or_generator:
            # ss.run()
            ss.run(self._z_iter, utils.maybe_call(inner_objective_feed_dicts, t))
            _, inner_loss = ss.run([self.iteration, inner_objective], utils.maybe_call(inner_objective_feed_dicts, t))
            self.inner_losses.append(inner_loss)

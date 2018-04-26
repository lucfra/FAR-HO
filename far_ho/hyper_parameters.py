from __future__ import absolute_import, print_function, division


from collections import defaultdict
# from functools import reduce

import tensorflow as tf
import numpy as np
import sys

from far_ho.utils import maybe_call, maybe_eval, merge_dicts, as_list

try:
    from ordered_set import OrderedSet
except ImportError:
    print('ordered-set package not found..')
    OrderedSet = set

from far_ho.optimizer import Optimizer
from far_ho.hyper_gradients import ReverseHg, HyperGradient
from far_ho.utils import GraphKeys

HYPERPARAMETERS_COLLECTIONS = (GraphKeys.HYPERPARAMETERS, GraphKeys.GLOBAL_VARIABLES)


# noinspection PyArgumentList,PyTypeChecker
def get_hyperparameter(name, initializer=None, shape=None, dtype=None, collections=None,
                       scalar=False):
    """
    Creates an hyperparameter variable, which is a GLOBAL_VARIABLE
    and HYPERPARAMETER. Mirrors the behavior of `tf.get_variable`.

    :param name: name of this hyperparameter
    :param initializer: initializer or initial value (can be also np.array or float)
    :param shape: optional shape, may be not needed depending on initializer
    :param dtype: optional type,  may be not needed depending on initializer
    :param collections: optional additional collection or list of collections, which will be added to
                        HYPERPARAMETER and GLOBAL_VARIABLES
    :param scalar: default False, if True splits the hyperparameter in its scalar components, i.e. each component
                    will be a single scalar hyperparameter. In this case the method returns a tensor which of the
                    desired shape (use this option with `ForwardHG`)

    :return: the newly created variable, or, if `scalar` is `True` a tensor composed by scalar variables.
    """
    _coll = HYPERPARAMETERS_COLLECTIONS
    if collections:
        _coll += as_list(collections)
    if not scalar:
        return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                               collections=_coll)
    else:
        with tf.variable_scope(name + '_components'):
            _shape = shape or initializer.shape
            if isinstance(_shape, tf.TensorShape):
                _shape = _shape.as_list()
            # _tmp_lst = reduce(lambda a, v: [a]*v, shape[::-1], None)
            _tmp_lst = np.empty(_shape, object)
            for k in range(np.multiply.reduce(_shape)):
                indices = np.unravel_index(k, _shape)
                # print(indices)
                _ind_name = '_'.join([str(ind) for ind in indices])
                _tmp_lst[indices] = tf.get_variable(_ind_name, (), dtype,
                                                    initializer if callable(initializer) else initializer[indices],
                                                    trainable=False, collections=_coll)
        return tf.convert_to_tensor(_tmp_lst.tolist(), name=name)


class HyperOptimizer(object):
    """
    Wrapper for performing hyperparameter optimization
    """

    def __init__(self, hypergradient=None):
        self.inner_objective = None
        assert hypergradient is None or isinstance(hypergradient, HyperGradient)
        self._hypergradient = hypergradient or ReverseHg()
        self._fin_hts = None
        self._global_step = None
        self._h_optim_dict = defaultdict(lambda: OrderedSet())

        self.inner_objective = None
        self.inner_losses = []

    # noinspection PyMethodMayBeStatic
    def inner_problem(self, inner_objective, inner_objective_optimizer, var_list=None, init_dynamics_dict=None,
                      **minimize_kwargs):
        """
        Set the dynamics Phi: a descent procedure on some inner_objective, can be called multiple times, for instance
        for batching inner optimization problems.

        :param inner_objective: a loss function for the inner optimization problem
        :param inner_objective_optimizer: an instance of some `far.Optimizer` (optimizers from Tensorflow must be
                                            extended to include tensors for the dynamics)
        :param var_list: optional list of variables (of the inner optimization problem)
        :param init_dynamics_dict: optional dictrionary that defines Phi_0 (see `OptimizerDict.set_init_dynamics`)
        :param minimize_kwargs: optional arguments to pass to `optimizer.minimize`
        :return: `OptimizerDict` from optimizer.
        """
        assert isinstance(inner_objective_optimizer, Optimizer)
        optim_dict = inner_objective_optimizer.minimize(
            inner_objective,
            var_list=var_list,
            **minimize_kwargs
        )
        self.inner_objective = optim_dict.objective if hasattr(optim_dict, 'objective') else inner_objective  # first
        # part is true for BacktrackingGD
        if init_dynamics_dict:
            optim_dict.set_init_dynamics(init_dynamics_dict)
        return optim_dict

    def outer_problem(self, outer_objective, optim_dict, outer_objective_optimizer,
                      hyper_list=None, global_step=None):
        """
        Set the outer optimization problem and the descent procedure for the optimization of the
        hyperparameters. Can be called at least once for every call of inner_problem, passing the resulting
         `OptimizerDict`. It can be called multiple times with different objective, optimizers and hyper_list s.

        :param outer_objective: scalar tensor for the outer objective
        :param optim_dict: `OptimizerDict` obtained by calling minimize on an instance of `far.Optimizer`
        :param outer_objective_optimizer: Optimizer (may be tensorflow optimizer) for the hyperparameters
        :param hyper_list: optional list of hyperparameters
        :param global_step: optional global step.
        :return: itself
        """
        hyper_list = self._hypergradient.compute_gradients(outer_objective, optim_dict, hyper_list=hyper_list)
        self._h_optim_dict[outer_objective_optimizer].update(hyper_list)
        self._global_step = global_step
        return self

    def minimize(self, outer_objective, outer_objective_optimizer, inner_objective, inner_objective_optimizer,
                 hyper_list=None, var_list=None, init_dynamics_dict=None, global_step=None,
                 aggregation_fn=None, process_fn=None):
        """
        Single method for calling once `inner_problem`, `outer_problem` and `finalize`, and optionally
        set an initial dynamics.  For more complex uses (like inner problems batching) use the methods separately.

        Returns method `HyperOptimizer.run`, that runs one hyperiteration.
        """
        optim_dict = self.inner_problem(inner_objective, inner_objective_optimizer, var_list, init_dynamics_dict)
        self.outer_problem(outer_objective, optim_dict, outer_objective_optimizer, hyper_list, global_step)
        return self.finalize(aggregation_fn=aggregation_fn, process_fn=process_fn)

    def finalize(self, aggregation_fn=None, process_fn=None):
        """
        To be called when no more dynamics or problems will be added, computes the updates
        for the hyperparameters. This behave nicely with global_variables_initializer.

        :param aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
                                by (default: reduce_mean)
        :param process_fn: Optional operation like normalizing to be applied to hypergradients before performing
                            a descent step (default: nothing).

        :return: the run method of this object.
        """
        if self._fin_hts is None:
            # in this way also far.optimizer can be used
            _maybe_first_arg = lambda _v: _v[0] if isinstance(_v, tuple) else _v
            # apply updates to each optimizer for outer objective minimization.
            # each optimizer might have more than one group of hyperparameters to optimize
            # and conversely different hyperparameters might be optimized with different optimizers.
            self._fin_hts = tf.group(*[_maybe_first_arg(opt.apply_gradients(
                self.hypergradient.hgrads_hvars(hyper_list=hll, aggregation_fn=aggregation_fn, process_fn=process_fn)))
                for opt, hll in self._h_optim_dict.items()])
            if self._global_step:
                with tf.control_dependencies([self._fin_hts]):
                    self._fin_hts = self._global_step.assign_add(1).op
        else:
            print('HyperOptimizer WARNING:, finalize has already been called on this object, ' +
                  'further calls have no effect', file=sys.stderr)
        return self.run

    @property
    def hypergradient(self):
        """
        :return: the hypergradient object underlying this wrapper.
        """
        return self._hypergradient

    @property
    def _hyperit(self):
        """
        iteration of minimization of outer objective(s), assuming the hyper-gradients are already computed.

        :return: an operation
        """
        assert self._fin_hts is not None, 'Must call HyperOptimizer.finalize before performing optimization.'
        return self._fin_hts

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, optimization_step_feed_dict=None, session=None, online=False,
            _skip_hyper_ts=False, _only_hyper_ts=False):
        """
        Run an hyper-iteration (i.e. train the model(s) and compute hypergradients) and updates the hyperparameters.

        :param _only_hyper_ts: just execute the update of the hyperparameters
        :param T_or_generator: int or generator (that yields an int), number of iteration (or stopping condition)
                                for the inner optimization (training) dynamics
        :param inner_objective_feed_dicts: an optional feed dictionary for the inner problem. Can be a function of
                                            step, which accounts for, e.g. stochastic gradient descent.
        :param outer_objective_feed_dicts: an optional feed dictionary for the outer optimization problem
                                            (passed to the evaluation of outer objective). Can be a function of
                                            hyper-iterations steps (i.e. global variable), which may account for, e.g.
                                            stochastic evaluation of outer objective.
        :param initializer_feed_dict:  an optional feed dictionary for the initialization of inner problems variables.
                                            Can be a function of
                                            hyper-iterations steps (i.e. global variable), which may account for, e.g.
                                            stochastic initialization.
        :param optimization_step_feed_dict: an optional feed dict for the iteration of the hyperparameter optimizer.
        :param session: optional session
        :param online: default `False` if `True` performs the online version of the algorithms (i.e. does not
                            reinitialize the state after at each run).
        :param _skip_hyper_ts: if `True` does not perform hyperparameter optimization step.
        """
        if not _only_hyper_ts:
            if not online:
                self.inner_losses = []
            self._hypergradient.run(T_or_generator, inner_objective_feed_dicts,
                                    outer_objective_feed_dicts,
                                    initializer_feed_dict,
                                    session=session,
                                    online=online, global_step=self._global_step,
                                    inner_objective=self.inner_objective)
            self.inner_losses = self._hypergradient.inner_losses

        if not _skip_hyper_ts:
            ss = session or tf.get_default_session()

            def _opt_fd():
                _od = maybe_call(optimization_step_feed_dict, maybe_eval(self._global_step)) \
                    if optimization_step_feed_dict else {}  # e.g. hyper-learning rate is a placeholder
                _oo_fd = maybe_call(outer_objective_feed_dicts, maybe_eval(self._global_step)) \
                    if outer_objective_feed_dicts else {}  # this is used in ForwardHG. In ReverseHG should't be needed
                # but it doesn't matter
                return merge_dicts(_od, _oo_fd)

            ss.run(self._hyperit, _opt_fd())

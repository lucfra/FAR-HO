from __future__ import absolute_import, print_function, division


import sys
from functools import reduce

try:
    import experiment_manager as em
    from experiment_manager import models
except ImportError:
    print('Install the package ExperimentManager first: https://github.com/lucfra/ExperimentManager',
          file=sys.stderr)
    em = models = None

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import far_ho as far
import numpy as np


def conv_layer(net, filters=32, hyperparameter=False, activation=tf.nn.relu,
               stride=1, max_pool=True, var_coll=far.HYPERPARAMETERS_COLLECTIONS,
               conv_initialization=tf.contrib.layers.xavier_initializer_conv2d(tf.float32)):
    max_pool_stride = [1, 2, 2, 1]

    bn = lambda _inp: tcl.batch_norm(_inp, variables_collections=var_coll)

    net + tcl.conv2d(net.out, num_outputs=filters, stride=stride,
                     kernel_size=3, normalizer_fn=bn, activation_fn=None,
                     trainable=not hyperparameter,
                     variables_collections=var_coll, weights_initializer=conv_initialization)
    net + activation(net.out)
    if max_pool:
        net + tf.nn.max_pool(net.out, max_pool_stride, max_pool_stride, 'VALID')


class HRMiniImagenetConv(models.Network):

    def __init__(self, _input, name='HyperRepresentationNet',
                 conv_block=conv_layer, reuse=False):
        self.conv_block = conv_block

        super(HRMiniImagenetConv, self).__init__(_input, name, False, reuse=reuse)

        # variables from batch normalization
        self.betas = self.filter_vars('beta')
        # moving mean and variance (these variables should be used at inference time... so must save them)
        self.moving_means = self.filter_vars('moving_mean')
        self.moving_variances = self.filter_vars('moving_variance')

        if not reuse:  # these calls might print a warning... it's not a problem..
            far.utils.remove_from_collection(far.GraphKeys.MODEL_VARIABLES, *self.moving_means)
            far.utils.remove_from_collection(far.GraphKeys.MODEL_VARIABLES, *self.moving_variances)
        # moving mean and averages are really poorly managed with tf....
        # would be the best to avoid using batch normalization altogether
        # remove moving avarages from hyperparameter collections (looks like there is no other way......)
        far.utils.remove_from_collection(far.GraphKeys.HYPERPARAMETERS, *self.moving_means)
        far.utils.remove_from_collection(far.GraphKeys.HYPERPARAMETERS, *self.moving_variances)
        print(name, 'MODEL CREATED')

    def for_input(self, new_input):
        return HRMiniImagenetConv(new_input, self.name, self.conv_block, reuse=True)

    def _build(self):
        for _ in range(4):
            self.conv_block(self)
        flattened_shape = reduce(lambda a, v: a * v, self.layers[-1].get_shape().as_list()[1:])
        self + tf.reshape(self.out, shape=(-1, flattened_shape), name='representation')


def mini_imagenet_model(x, name):
    """
    Kind of standard model for miniimagenet
    """
    return HRMiniImagenetConv(x, name=name)


def omniglot_model(x, name):
    return HRMiniImagenetConv(x, name=name,
                              conv_block=lambda net: conv_layer(net, 64, stride=2, max_pool=False))


def setup(T, seed, n_episodes_testing, MBS):
    T = far.utils.as_tuple_or_list(T)
    print(T)

    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    return T, tf.InteractiveSession(config=em.utils.GPU_CONFIG()), n_episodes_testing // MBS


def _optimizers(lr, mlr0, mlr_decay, learn_lr=True):
    io_optim = far.GradientDescentOptimizer(far.get_hyperparameter('lr', lr) if learn_lr else
                                            tf.constant(lr, name='lr'))
    gs = tf.get_variable('global_step', initializer=0, trainable=False)
    meta_lr = tf.train.inverse_time_decay(mlr0, gs, 1., mlr_decay)
    oo_optim = tf.train.AdamOptimizer(meta_lr)
    farho = far.HyperOptimizer()
    return io_optim, gs, meta_lr, oo_optim, farho


def _helper_function(exs, n_episodes_testing, MBS, ss, farho, T):
    def feed_dicts(dat_lst):
        dat_lst = em.as_list(dat_lst)
        tr_fd = em.utils.merge_dicts(
            *[{_ex.x: dat.train.data, _ex.y: dat.train.target}
              for _ex, dat in zip(exs, dat_lst)])
        val_fd = em.utils.merge_dicts(
            *[{_ex.x: dat.test.data, _ex.y: dat.test.target}
              for _ex, dat in zip(exs, dat_lst)])

        return tr_fd, val_fd

    def just_train_on_dataset(dat):
        """
        :return: list of accuracies or errors on training and test set (pair)
        """
        _trfd, _vfd = feed_dicts(dat)
        ss.run(farho.hypergradient.initialization)
        for _ in range(T[-1]):
            ss.run(farho.hypergradient.ts, _trfd)
        return (ss.run([_ex.scores['accuracy'] for _ex in exs], _trfd),
                ss.run([_ex.scores['accuracy'] for _ex in exs], _vfd),
                ss.run([_ex.errors['validation'] for _ex in exs], _trfd),
                ss.run([_ex.errors['validation'] for _ex in exs], _vfd))

    def accs_and_errs(metasets):
        results = []
        for meta_dataset, name in zip(metasets, ['train', 'valid', 'test']):
            ac_tr, ac_tst, err_tr, err_ts = [], [], [], []
            for _d in meta_dataset.generate(n_episodes_testing, batch_size=MBS, rand=0):
                jt = just_train_on_dataset(_d)
                ac_tr.extend(jt[0])
                ac_tst.extend(jt[1])
                err_tr.extend(jt[2])
                err_ts.extend(jt[3])

            results.append((
                ('mean accuracy on training set::{}'.format(name), np.mean(ac_tr)),
                ('mean accuracy on test set::{}'.format(name), np.mean(ac_tst)),
                ('mean error on train set::{}'.format(name), np.mean(err_tr)),
                ('mean error on test set::{}'.format(name), np.mean(err_ts)),
                ('HIDE::accuracies and errors::{}'.format(name), (ac_tr, ac_tst, err_tr, err_ts))
            ))
        return results

    return feed_dicts, just_train_on_dataset, accs_and_errs, em.rec.COS('mean accuracy on test set::valid')


def _records(metasets, saver, model, cond, ss, accs_and_errs, ex_name, meta_lr):
    return [em.rec.direct('norm of hyperparameters',
                          lambda: [(np.linalg.norm(h)) for h in ss.run(far.utils.hyperparameters())],
                          'norm of hypergradients',
                          lambda: [(np.linalg.norm(h)) for h in ss.run(far.utils.hypergradients())],
                          'FLAT', lambda: accs_and_errs(metasets),
                          'SKIP::meta_lr', lambda: meta_lr.eval(),
                          'SKIP::exp name', lambda: ex_name,
                          'SKIP::num_classes', lambda: metasets.train.dim_target
                          ),
            cond.rec_best_record(),
            cond.rec_score(),
            em.rec.tensors('lr'),
            em.rec.autoplot(saver, ex_name),
            em.rec.model(model, cond.condition())]


SAVER_EXP = em.Saver.std('L2BRIDGE', 'HYPER_REPR',
                         description='standard experiment for learning meta-representations')


def train(metasets, ex_name, hyper_repr_model_builder, classifier_builder=None, saver=None, seed=0, MBS=4,
          available_devices=('/gpu:0', '/gpu:1'),
          mlr0=.001, mlr_decay=1.e-5, T=4, n_episodes_testing=600,
          print_every=1000, patience=40, restore_model=False,
          lr=0.1, learn_lr=True, process_fn=None):
    """
    Function for training an hyper-representation network.

    :param metasets: Datasets of MetaDatasets
    :param ex_name: name of the experiment
    :param hyper_repr_model_builder: builder for the representation model,
                                        function (input, name) -> `experiment_manager.Network`
    :param classifier_builder: optional builder for classifier model (if None then builds a linear model)
    :param saver: experiment_manager.Saver object
    :param seed:
    :param MBS: meta-batch size
    :param available_devices: distribute the computation among different GPUS!
    :param mlr0: initial meta learning rate
    :param mlr_decay:
    :param T: number of gradient steps for training ground models
    :param n_episodes_testing:
    :param print_every:
    :param patience:
    :param restore_model:
    :param lr: initial ground models learning rate
    :param learn_lr: True for optimizing the ground models learning rate
    :param process_fn: optinal hypergradient process function (like gradient clipping)

    :return: tuple: the saver object, the hyper-representation model and the list of experiments objects
    """
    if saver is None:
        saver = SAVER_EXP(metasets)

    T, ss, n_episodes_testing = setup(T, seed, n_episodes_testing, MBS)
    exs = [em.SLExperiment(metasets) for _ in range(MBS)]

    hyper_repr_model = hyper_repr_model_builder(exs[0].x, name=ex_name)
    if classifier_builder is None: classifier_builder = lambda inp, name: models.FeedForwardNet(
        inp, metasets.train.dim_target, name=name)

    io_optim, gs, meta_lr, oo_optim, farho = _optimizers(lr, mlr0, mlr_decay, learn_lr)

    for k, ex in enumerate(exs):
        with tf.device(available_devices[k % len(available_devices)]):
            ex.model = classifier_builder(hyper_repr_model.for_input(ex.x).out, 'Classifier_%s' % k)
            ex.errors['training'] = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=ex.y, logits=ex.model.out)
            )
            ex.errors['validation'] = ex.errors['training']
            ex.scores['accuracy'] = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(ex.y, 1), tf.argmax(ex.model.out, 1)), tf.float32),
                name='accuracy')

            optim_dict = farho.inner_problem(ex.errors['training'], io_optim, var_list=ex.model.var_list)
            farho.outer_problem(ex.errors['validation'], optim_dict, oo_optim, global_step=gs)

    farho.finalize(process_fn=process_fn)

    feed_dicts, just_train_on_dataset, mean_acc_on, cond = _helper_function(
        exs, n_episodes_testing, MBS, ss, farho, T)

    rand = em.get_rand_state(0)

    with saver.record(*_records(metasets, saver, hyper_repr_model, cond, ss, mean_acc_on, ex_name, meta_lr),
                      where='far', every=print_every, append_string=ex_name):
        tf.global_variables_initializer().run()
        if restore_model:
            saver.restore_model(hyper_repr_model)
        # ADD ONLY TESTING
        for _ in cond.early_stopping_sv(saver, patience):
            trfd, vfd = feed_dicts(metasets.train.generate_batch(MBS, rand=rand))

            farho.run(T[0], trfd, vfd)  # one iteration of optimization of representation variables (hyperparameters)

    return saver, hyper_repr_model, exs

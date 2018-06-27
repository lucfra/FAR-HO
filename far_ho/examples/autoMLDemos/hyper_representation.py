from __future__ import absolute_import, print_function, division
from functools import reduce

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import far_ho as far
from collections import defaultdict


def hyper_conv_layer(x):
    hyper_coll = far.HYPERPARAMETERS_COLLECTIONS
    return tcl.conv2d(x, num_outputs=64, stride=2,
                      kernel_size=3,
                      # normalizer_fn=
                      # lambda z: tcl.batch_norm(z,
                      #   variables_collections=hyper_coll,
                      #   trainable=False),
                      trainable=False,
                      variables_collections=hyper_coll)


def build_hyper_representation(_x, auto_reuse=False):
    reuse = tf.AUTO_REUSE if auto_reuse else False
    with tf.variable_scope('HR', reuse=reuse):
        conv_out = reduce(lambda lp, k: hyper_conv_layer(lp),
                          range(4), _x)
        return tf.reshape(conv_out, shape=(-1, 256))


def classifier(_x, _y):
    return tcl.fully_connected(
        _x, int(_y.shape[1]), activation_fn=None,
        weights_initializer=tf.zeros_initializer)


def get_placeholders():
    _x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    _y = tf.placeholder(tf.float32, (None, 5))
    return _x, _y


def get_data():
    import experiment_manager.datasets.load as load
    return load.meta_omniglot(
        std_num_classes=5,
        std_num_examples=(5, 15*5))


def make_feed_dicts(tasks, mbd):
    train_fd, test_fd = {}, {}
    for task, _x, _y in zip(tasks, mbd['x'], mbd['y']):
        train_fd[_x] = task.train.data
        train_fd[_y] = task.train.target
        test_fd[_x] = task.test.data
        test_fd[_y] = task.test.target
    return train_fd, test_fd


def accuracy(y_true, logits):
    return tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y_true, 1), tf.argmax(logits, 1)),
            tf.float32))


def meta_test(meta_batches, mbd, opt, n_steps):
    ss = tf.get_default_session()
    ss.run(tf.variables_initializer(tf.trainable_variables()))
    sum_loss, sum_acc = 0., 0.
    n_tasks = len(mbd['err'])*len(meta_batches)
    for _tasks in meta_batches:
        _train_fd, _valid_fd = make_feed_dicts(_tasks, mbd)
        mb_err = tf.add_n(mbd['err'])
        mb_acc = tf.add_n(mbd['acc'])
        opt_step = opt.minimize(mb_err)
        for i in range(n_steps):
            ss.run(opt_step, feed_dict=_train_fd)

        mb_loss, mb_acc = ss.run([mb_err, mb_acc], feed_dict=_valid_fd)
        sum_loss += mb_loss
        sum_acc += mb_acc

    return sum_loss/n_tasks, sum_acc/n_tasks


meta_batch_size = 16  # meta-batch size
n_episodes_testing = 10
mb_dict = defaultdict(list)  # meta_batch dictionary
meta_dataset = get_data()

for _ in range(meta_batch_size):
    x, y = get_placeholders()
    mb_dict['x'].append(x)
    mb_dict['y'].append(y)
    hyper_repr = build_hyper_representation(x, auto_reuse=True)
    logits = classifier(hyper_repr, y)
    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=logits))
    mb_dict['err'].append(ce)
    mb_dict['acc'].append(accuracy(y, logits))

L = tf.add_n(mb_dict['err'])
E = L / meta_batch_size
mean_acc = tf.add_n(mb_dict['acc'])/meta_batch_size

inner_opt = far.GradientDescentOptimizer(learning_rate=0.1)
outer_opt = tf.train.AdamOptimizer()

hyper_step = far.HyperOptimizer().minimize(
    E, outer_opt, L, inner_opt)

sess = tf.Session()
n_hyper_steps = 100
with sess.as_default():
    tf.global_variables_initializer().run()
    T = 3
    for meta_batch in meta_dataset.train.generate(n_hyper_steps, batch_size=meta_batch_size):
        train_fd, valid_fd = make_feed_dicts(meta_batch, mb_dict)
        hyper_step(T, train_fd, valid_fd)

        test_optim = tf.train.GradientDescentOptimizer(0.1)
        test_mbs = [mb for mb in meta_dataset.test.generate(n_episodes_testing, batch_size=meta_batch_size, rand=0)]

        print('train_test (loss, acc)', sess.run([E, mean_acc], feed_dict=valid_fd))
        print('test_test (loss, acc)', meta_test(test_mbs, mb_dict, test_optim, T))

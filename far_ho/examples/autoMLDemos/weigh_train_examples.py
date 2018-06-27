import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import far_ho as far
import far_ho.examples as far_ex

sess = tf.InteractiveSession()


def get_data():
    # load a small portion of mnist data
    datasets = far_ex.mnist(folder=os.path.join(os.getcwd(), 'MNIST_DATA'), partitions=(.1, .1,))
    return datasets.train, datasets.validation


def g_logits(x,y):
    with tf.variable_scope('model'):
        h1 = layers.fully_connected(x, 300)
        logits = layers.fully_connected(h1, int(y.shape[1]))
    return logits


x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
logits = g_logits(x,y)
train_set, validation_set = get_data()

lambdas = far.get_hyperparameter('lambdas', tf.zeros(train_set.num_examples))
lr = far.get_hyperparameter('lr', initializer=0.01)

ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
L = tf.reduce_mean(tf.sigmoid(lambdas)*ce)
E = tf.reduce_mean(ce)

inner_optimizer = far.GradientDescentOptimizer(lr)
outer_optimizer = tf.train.AdamOptimizer()
hyper_step = far.HyperOptimizer().minimize(E, outer_optimizer, L, inner_optimizer)

T = 200  # Number of inner iterations
train_set_supplier = train_set.create_supplier(x, y)
validation_set_supplier = validation_set.create_supplier(x, y)
tf.global_variables_initializer().run()

print('inner:', L.eval(train_set_supplier()))
print('outer:', E.eval(validation_set_supplier()))
# print('-'*50)
n_hyper_iterations = 10
for _ in range(n_hyper_iterations):
    hyper_step(T,
               inner_objective_feed_dicts=train_set_supplier,
               outer_objective_feed_dicts=validation_set_supplier)
    print('inner:', L.eval(train_set_supplier()))
    print('outer:', E.eval(validation_set_supplier()))
    print('learning rate', lr.eval())
    print('norm of examples weight', tf.norm(lambdas).eval())
    print('-'*50)

# # plt.plot(tr_accs, label='training accuracy')
# # plt.plot(val_accs, label='validation accuracy')
# # plt.legend(loc=0, frameon=True)
# # # plt.xlim(0, 19)

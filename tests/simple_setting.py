import tensorflow as tf
import far_ho as far


tf.reset_default_graph()
ss = tf.InteractiveSession()

v1 = tf.Variable([10., 3])

v2 = tf.Variable([[-1., -2], [1., -21.]])


# In[17]:

lmbd = far.get_hyperparameter('lambda', 
                              initializer=tf.ones_initializer, shape=v2.get_shape())

reg2 = far.get_hyperparameter('reg2', 0.1)

eta = far.get_hyperparameter('eta', 0.1)
beta1 = far.get_hyperparameter('beta1', 1.)
beta2 = far.get_hyperparameter('beta2', 2.)

# noinspection PyTypeChecker
cost = tf.reduce_mean(v1**2) + tf.reduce_sum(lmbd*v2**2) + reg2*tf.nn.l2_loss(v1)

io_optim = far.AdamOptimizer(eta, tf.nn.sigmoid(beta1), tf.nn.sigmoid(beta2), epsilon=1.e-4)

oo = tf.reduce_mean(v1*v2)

rhg = far.ReverseHG()

optim_oo = tf.train.AdamOptimizer()
# ts_hy = optim_oo.apply_gradients(rhg.hgrads_hvars())
farho = far.HyperOptimizer(rhg)
run = farho.minimize(oo, optim_oo, cost, io_optim)

print(tf.global_variables())

print(far.utils.hyperparameters())


tf.global_variables_initializer().run()
print('hyperparameters:', ss.run(far.utils.hyperparameters()))

farho.run(100)

print(ss.run(tf.trainable_variables()))

print(ss.run(far.utils.hypergradients()))

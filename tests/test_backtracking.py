import far_ho as far
import tensorflow as tf
import numpy as np

run_gd = False  # true for constant step size
right_step = False

tf.reset_default_graph()
ss = tf.InteractiveSession()

L = tf.constant(10.65158)
kappa = .25
lmbd = far.get_hyperparameter('lmbd', .008921)

# L / (1 + lmbd)
sol = L / (1 + lmbd)
# w = tf.get_variable('w', initializer=sol)
w = tf.get_variable('w', initializer=tf.zeros_initializer, shape=(1,))
b = tf.get_variable('b', initializer=tf.ones_initializer, shape=(2,))

outer_obj = (w - 2.) ** 2 / 2.


#  this should be a callable! yeah
def inner_obj(var_list):
    w = var_list[0]
    obj = (w - L) ** 2 / 2. + lmbd * (w) ** 2 / 2 + tf.reduce_sum(var_list[1] ** 2)
    return obj[0]


io_lip = 1. + lmbd

farho = far.HyperOptimizer(far.ReverseHg())
if run_gd:
    inner_obj = inner_obj([w, b])
    if right_step:
        gd = far.GradientDescentOptimizer(2 * kappa / io_lip)
    else: gd = far.GradientDescentOptimizer(1.)
else:
    gd = far.BackTrackingGradientDescentOptimizer(tf.constant(1.))

run = farho.minimize(outer_obj, tf.train.GradientDescentOptimizer(0.01), inner_obj, gd, var_list=[w, b],
                     hyper_list=[lmbd])

tf.global_variables_initializer().run()

rs = []
for t in range(10, 100):
    run(t)
    #     print(farho.hypergradient._history)
    #     print()
    rs.append(ss.run(far.hypergradients())[0])

print(rs)

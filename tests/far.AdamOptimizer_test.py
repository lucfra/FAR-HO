import tensorflow as tf
import far_ho as far

tf.reset_default_graph()

v = tf.Variable([1., 2., 3.])
obj = tf.reduce_sum(tf.pow(v, 2))

iterations = 10
lr = .1

adam = far.AdamOptimizer(lr)
adam_dict = adam.minimize(obj)

with tf.Session().as_default() as ss:
    tf.global_variables_initializer().run()
    for _ in range(iterations):
        print(ss.run(adam_dict.dynamics))
        ss.run(adam_dict.ts)
    res = v.eval()

print('far.Adam:', res)

tf_adam = tf.train.AdamOptimizer(lr).minimize(obj, var_list=[v])

with tf.Session().as_default() as ss:
    tf.global_variables_initializer().run()
    for _ in range(iterations):
        ss.run(tf_adam)
    res2 = v.eval()

print('tf.Adam: ', res2)


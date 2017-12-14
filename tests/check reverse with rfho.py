
# coding: utf-8

# In[ ]:


# In[2]:

import tensorflow as tf
import numpy as np
import experiment_manager as em
import far_ho as far


# In[4]:

far.utils._check()


# In[16]:

try: ss.close()
except: pass
tf.reset_default_graph()
ss = tf.InteractiveSession()

v1 = tf.Variable([1.,3])

v2 = tf.Variable([[-1., -2], [1., 0.]])


# In[17]:

lmbd = far.get_hyperparameter('lambda', 
                              initializer=tf.ones_initializer,
                             shape=v2.get_shape())

cost = tf.reduce_mean(v1**2) + tf.reduce_sum(lmbd*v2**2)

io_optim = far.AdamOptimizer(epsilon=1.e-6)

#io_optim = far.MomentumOptimizer(far.get_hyperparameter('eta', 0.1), far.get_hyperparameter('mu', .9))
io_optim_dict = io_optim.minimize(cost)

oo = tf.reduce_mean(v1*v2)


# In[18]:

rhg = far.ReverseHg()
rhg.compute_gradients(oo, io_optim_dict)

optim_oo = tf.train.AdamOptimizer()
ts_hy = optim_oo.apply_gradients(rhg.hgrads_hvars())


# In[ ]:




# In[19]:

print(tf.global_variables())


# In[20]:

print(far.utils.hyperparameters())


# In[21]:

tf.global_variables_initializer().run()
print('hyperparameters:', ss.run(far.utils.hyperparameters()))


# In[22]:

rhg.run(10)


# In[23]:

print(ss.run(far.utils.hypergradients()))


# In[ ]:




# In[24]:

# ss.run(ts_hy)  # perform an hypergradient descent step....
# print(ss.run(far.utils.hyperparameters()))
# print(ss.run(oo))


# # check with rfho

# In[25]:

import rfho as rf


# In[26]:

w, c, co = rf.vectorize_model([v1, v2], cost, oo, augment=2)


# In[27]:

#dyn = rf.MomentumOptimizer.create(w, 0.1, 0.9, loss=c)
dyn = rf.AdamOptimizer.create(w, loss=c)


# In[28]:

hyperg = rf.HyperOptimizer(dyn, {co: lmbd}, rf.ReverseHG)


# In[29]:

hyperg.initialize()


# In[30]:

hgs = hyperg.run(10, val_feed_dict_suppliers={co:lambda: None},
                 _debug_no_hyper_update=True)


# In[32]:

print('rfho')
print(ss.run(hyperg.hyper_gradients.hyper_gradient_vars))
print('far')
print(ss.run(far.utils.hypergradients()))  # there are also other hyperparameters


# # END... RESULTS ARE (almost the) SAME

# In[ ]:




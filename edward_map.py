
# coding: utf-8

# In[1]:

import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import edward as ed
ed.set_seed(42)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from edward.models import Normal, Categorical, PointMass


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:


N = 100   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.
neurons = int(sys.argv[1])
layers = int(sys.argv[2])
epochs = 100
n_samples = 100
eval_batch_size = 5000
iters = int(mnist.train.num_examples/N*epochs)


# In[4]:


X_val = mnist.validation.images
Y_val = np.argmax(mnist.validation.labels,axis=1)


# In[5]:


def neural_network(X):
    h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
    if layers == 2:
        h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return h

# MODEL
with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([D, neurons]), scale=1e20*tf.ones([D, neurons]), name="W_0")
    b_0 = Normal(loc=tf.zeros(neurons), scale=1e20*tf.ones(neurons), name="b_0")

    if layers == 2:
        W_1 = Normal(loc=tf.zeros([neurons, neurons]), scale=1e20*tf.ones([neurons, neurons]), name="W_1")
        b_1 = Normal(loc=tf.zeros(neurons), scale=1e20*tf.ones(neurons), name="b_1")

    W_2 = Normal(loc=tf.zeros([neurons, K]), scale=1e20*tf.ones([neurons, K]), name="W_2")
    b_2 = Normal(loc=tf.zeros(K), scale=1e20*tf.ones(K), name="b_2")

    X = tf.placeholder(tf.float32, [None, D], name="X")
    y = Categorical(neural_network(X), name="y")
    
# INFERENCE
with tf.name_scope("posterior"):
    with tf.name_scope("qW_0"):
        qW_0 = PointMass(tf.get_variable("qW0", shape=[D, neurons], initializer=tf.contrib.layers.xavier_initializer()))
    with tf.name_scope("qb_0"):
        qb_0 = PointMass(tf.get_variable("qb0", shape=[neurons], initializer=tf.contrib.layers.xavier_initializer()))
        
    if layers == 2:
        with tf.name_scope("qW_1"):
            qW_1 = PointMass(tf.get_variable("qW1", shape=[neurons, neurons], initializer=tf.contrib.layers.xavier_initializer()))
        with tf.name_scope("qb_1"):
            qb_1 = PointMass(tf.get_variable("qb1", shape=[neurons], initializer=tf.contrib.layers.xavier_initializer()))

    with tf.name_scope("qW_2"):
        qW_2 = PointMass(tf.get_variable("qW2", shape=[neurons, K], initializer=tf.contrib.layers.xavier_initializer()))
    with tf.name_scope("qb_2"):
        qb_2 = PointMass(tf.get_variable("qb2", shape=[K], initializer=tf.contrib.layers.xavier_initializer()))

# In[9]:


def eval_acc_auc(dataset):
    prob_lst = []
    pred_lst = []
    Y = []

    for _ in tqdm(range(dataset.num_examples//eval_batch_size)):
        X, y = mnist.test.next_batch(eval_batch_size, shuffle=False)
        prob_lst_temp = []
        W0_samp = qW_0.sample()
        b0_samp = qb_0.sample()
        if layers == 2:
            W1_samp = qW_1.sample()
            b1_samp = qb_1.sample()
        W2_samp = qW_2.sample()
        b2_samp = qb_2.sample()

        h_samp = tf.nn.relu(tf.matmul(X, W0_samp) + b0_samp)
        if layers == 2:
            h_samp = tf.nn.relu(tf.matmul(h_samp, W1_samp) + b1_samp)
        h_samp = tf.matmul(h_samp, W2_samp) + b2_samp
        prob = tf.nn.softmax(h_samp)

        # Also compue the probabiliy of each class for each (w,b) sample.
        prob_lst_temp.append(prob.eval())
        pred_lst.append(np.argmax(np.mean(prob_lst_temp,axis=0),axis=1))
        prob_lst.append(np.max(np.mean(prob_lst_temp,axis=0),axis=1))
        Y.append(y)

    Y_pred = np.concatenate(pred_lst)
    Y_prob = np.concatenate(prob_lst)
    Y_actual = np.argmax(np.concatenate(Y), axis=1)

    acc = (Y_pred == Y_actual).mean()*100

    auc = roc_auc_score(Y_pred == Y_actual, Y_prob)

    return acc, auc


# In[7]:


y_ph = tf.placeholder(tf.int32, [N])

if layers == 1:
    inference = ed.MAP({W_0: qW_0, b_0: qb_0,
                         W_2: qW_2, b_2: qb_2}, data={y: y_ph})
elif layers == 2:
    inference = ed.MAP({W_0: qW_0, b_0: qb_0,
                         W_1: qW_1, b_1: qb_1,
                         W_2: qW_2, b_2: qb_2}, data={y: y_ph})
    
inference.initialize(n_iter=iters, n_print=100, optimizer='adam', scale={y: float(mnist.train.num_examples) / N})

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(1, inference.n_iter+1):
    X_batch, Y_batch = mnist.train.next_batch(N)
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={X: X_batch, y_ph: Y_batch})
    #inference.print_progress(info_dict)
    #if i*N % mnist.train.num_examples == 0: # epoch
    #    print('  Val acc:', eval_acc(X_val, Y_val))


# In[10]:


X_test = mnist.test.images
Y_test = np.argmax(mnist.test.labels,axis=1)
print('MAP', neurons, layers, epochs, n_samples, eval_acc_auc(mnist.test))

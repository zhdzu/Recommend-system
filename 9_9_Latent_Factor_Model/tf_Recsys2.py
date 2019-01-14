# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:54:44 2018

@author: CynthiaWang
"""

# Imports for data io operations
from collections import deque
from six import next
import readers

# Main imports for training
import tensorflow as tf
import numpy as np

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(41)

#u_num = 6040 # Number of users in the dataset
#i_num = 3963# Number of movies in the dataset
u_num = 999 # Number of users in the dataset
i_num = 2131# Number of movies in the dataset

batch_size = 10 # Number of samples per batch
dims = 5          # Dimensions of the data, 15
max_epochs = 50  # Number of times the network sees all the training data

# Device used for all computations
place_device = "/cpu:0"

def get_data():
    # Reads file using the demiliter :: form the ratings file
    # Columns are user ID, item ID, rating, and timestamp
    # Sample data - 3::1196::4::978297539
    #df = readers.read_file("./ml-1m/ratings.dat", sep="::")
    #df = readers.read_file("./ml-1m/dfnew2.dat", sep=",")
    df = readers.read_file("tf.dat", sep=",")
    #df.item = df.item+1
    rows = len(df)
    # Purely integer-location based indexing for selection by position
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    # Separate data into train and test, 90% for train and 10% for test
    split_index = int(rows * 0.9)
    # Use indices to separate the data
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    
    return df_train, df_test

def clip(x):
    return np.clip(x, 1.0, 5.0)

def model(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        with tf.variable_scope('lsi',reuse=tf.AUTO_REUSE):
            # Using a global bias term
            bias_global = tf.get_variable("bias_global", shape=[])
            # User and item bias variables
            # get_variable: Prefixes the name with the current variable scope 
            # and performs reuse checks.
            w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
            w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
            # embedding_lookup: Looks up 'ids' in a list of embedding tensors
            # Bias embeddings for user and items, given a batch
            bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
            bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
            # User and item weight variables
            w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            # Weight embeddings for user and items, given a batch
            embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
            embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    
    with tf.device(device):
        # reduce_sum: Computes the sum of elements across dimensions of a tensor
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        # l2_loss: Computes half the L2 norm of a tensor without the sqrt
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), 
                             name="svd_regularizer")
    return infer, regularizer

def loss(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        # Use L2 loss to compute penalty
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        # 'Follow the Regularized Leader' optimizer
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return cost, train_op

# Read data from ratings file to build a TF model
df_train, df_test = get_data()

samples_per_batch = len(df_train) // batch_size
print("Number of train samples %d, test samples %d, samples per batch %d" % 
      (len(df_train), len(df_test), samples_per_batch))

# Using a shuffle iterator to generate random batches, for training
iter_train = readers.ShuffleIterator([df_train["user"],
                                     df_train["item"],
                                     df_train["rate"]],
                                     batch_size=batch_size)

# Sequentially generate one-epoch batches, for testing
iter_test = readers.OneEpochIterator([df_test["user"],
                                     df_test["item"],
                                     df_test["rate"]],
                                     batch_size=-1)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer = model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims, device=place_device)
_, train_op = loss(infer, regularizer, rate_batch, learning_rate=0.10, reg=0.05, device=place_device)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

err1=[]
err2=[]
with tf.Session() as sess:
    sess.run(init_op)
    print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()
    for i in range(max_epochs * samples_per_batch):
        users, items, rates = next(iter_train)
        _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                               item_batch: items,
                                                               rate_batch: rates})
        pred_batch = clip(pred_batch)
        errors.append(np.power(pred_batch - rates, 2))
        if i % samples_per_batch == 0:
            train_err = np.sqrt(np.mean(errors))
            test_err2 = np.array([])
            for users, items, rates in iter_test:
                pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                        item_batch: items})
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
            end = time.time()
            print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" % (i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)), end - start))
            err1.append(train_err)
            err2.append(np.sqrt(np.mean(test_err2)))
            start = end

    saver.save(sess, './save/')
    
epoch=list(range(1,50+1))
epoch
err1
import pandas as pd
import seaborn as sns
#sns.set(style="ticks")

#dots = pd.DataFrame({'Train Error':err1,'Test Error':err2})#sns.load_dataset("dots")

import matplotlib.pyplot as plt

#折线图
#x = [5,7,11,17,19,25]#点的横坐标
#k1 = [0.8222,0.918,0.9344,0.9262,0.9371,0.9353]#线1的纵坐标
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(epoch,err1,'s-',color = 'b',label="Train Error")#s-:方形
plt.plot(epoch,err2,'o-',color = 'g',label="Test Error ")#o-:圆形
plt.xlabel("Epoches")#横坐标名字
plt.ylabel("Error")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()


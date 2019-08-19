#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:09:38 2018

@author: zhaoyu
"""

from ReadMNIST import *
import tensorflow as tf

def one_hot_encoding(y, n_class):
    one_hot = np.zeros([len(y), n_class], dtype='float32')
    for i in range(len(y)):
        one_hot[i][y[i]] = 1.0
    return one_hot

if '__main__' == __name__:
    print('tensorflow implemented CNN')

    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file, True)

    # data preparation
    n_samp, n_feat = X_train.shape
    n_class = len(set(y_train))
    y_train_onehot = one_hot_encoding(y_train, n_class)

    # define basic containers (placeholders) computing graphs
    input_X = tf.placeholder(tf.float32, [None, n_feat])
    input_y = tf.placeholder(tf.int32, [None])
    input_y_onehot = tf.placeholder(tf.float32, [None, n_class])

    # preprare for image process
    input_X_image = tf.reshape(input_X, [-1, 28, 28, 1])

    # define layers and parameters
    # The filters in conv layer is defined by myself,
    # Diffing from the MLP, in wich the weight matrix has its own shape fixed by
    # the numbers of neurons of L-th and (L+1)-th layer.
    W1 = tf.Variable(tf.random_normal([5, 5, 1, 32], 0, 1, dtype=tf.float32))
    b1 = tf.Variable(tf.random_normal([32, ], 0, 1, dtype=tf.float32))
    h1_conv = tf.nn.sigmoid(tf.nn.conv2d(input_X_image, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    h1_pool = tf.nn.max_pool(h1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully-connect layer 1
    h1_pool_flat = tf.reshape(h1_pool, [-1, 14 * 14 * 32])
    W_fc = tf.Variable(tf.random_normal([14 * 14 * 32, 1024], 0, 1))
    b_fc = tf.Variable(tf.random_normal([1024], 0, 1))
    h_fc = tf.nn.sigmoid(tf.matmul(h1_pool_flat, W_fc) + b_fc)

    # fully-connect layer 2
    W_fc2 = tf.Variable(tf.random_normal([1024, 10], 0, 1))
    b_fc2 = tf.Variable(tf.random_normal([10], 0, 1))
    f = tf.matmul(h_fc, W_fc2) + b_fc2

    # define output layer, prediction, and loss function
    prob = tf.nn.softmax(f)
    y_pred = tf.cast(tf.argmax(prob, axis=1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(input_y, y_pred), tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f, labels=input_y_onehot))

    # define optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    s = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    loops_cnt = 10
    batch_size = 32
    batch_num = int(np.ceil(n_samp/batch_size))
    for i in range(loops_cnt):
        seq = np.arange(n_samp)
        np.random.shuffle(seq)
        for epoch in range(100):
            scope = seq[range(epoch * batch_size, min((epoch + 1) * batch_size, n_samp))]
            X_batch = X_train[scope, :]
            y_batch = y_train[scope]
            y_batch_onehot = y_train_onehot[scope, :]
            s.run(train_step, {input_X: X_batch, input_y_onehot: y_batch_onehot})
        loss_i, acc_i = s.run([loss, accuracy], {input_X: X_batch, input_y: y_batch, input_y_onehot: y_batch_onehot})
#        print(float(0) in prob) # check NaN in logits
        print(i, ' : ', loss_i, acc_i)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:45:42 2018

@author: zhaoyu
"""

from ReadMNIST import *
import tensorflow as tf
import numpy as np
import keras.utils as ku

class MultilayerPerceptron:
    def __init__(self, train_data):
        self.X, self.y = train_data
        self.n_samp, self.n_feat = self.X.shape
        self.n_class = len(set(self.y))

if __name__ == '__main__':
    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file, True)
    y_train = ku.to_categorical(y_train)
    
    n_samp, n_feat = X_train.shape
    n_class = len(set(y_train))

    X_input = tf.placeholder(tf.float32, [None, n_feat])
    y_input = tf.placeholder(tf.float32, [None])

    weight = tf.Variable(tf.random_normal([n_feat, n_class], dtype=tf.float32))
    b = tf.Variable(tf.zeros([n_class]))

    s = tf.Session()
    s.run(tf.global_variables_initializer())
    
    print(s.run(X_input, {X_input:X_train}).shape)
    
    pr = tf.nn.softmax(tf.matmul(X_input, weight) + b)
    loss = tf.reduce_mean(tf.reduce_sum(-tf.log(pr)*y_input, axis=1))
    
    
    

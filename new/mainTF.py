# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:43:10 2018

@author: ZhaoYu

More practice makes higher proficiency

"""

import tensorflow as tf
from ReadMNIST import *

def one_hot_coding(y, n_class):
    one_hot = np.zeros([y.shape[0], n_class])
    for i in range(y.shape[0]):
        one_hot[i][y[i]] = 1
    return one_hot

def forward(X, weight, b):
    pr = tf.nn.softmax(tf.matmul(X, weight)+b)
    return pr

def forward_2(X, weight0, b0, weight1, b1):
    tmp = tf.nn.sigmoid(tf.matmul(X, weight0)+b0)
    pr = tf.nn.softmax(tf.matmul(tmp, weight1)+b1)
    return pr

def cross_ent_loss(y_onehot, pr):
    return tf.reduce_mean(tf.reduce_sum(-tf.log(pr)*y_onehot, axis=1))

def predict(pr):
    return tf.argmax(pr, axis=1)

def accuracy(y, y_hat):
    return tf.reduce_mean(tf.cast(tf.equal(y, y_hat), tf.float32))

class MultilayerPerceptron:
    def __init__(self, train_data):
        self.X, self.y = train_data
        self.n_samp, self.n_feat = self.X.shape
        self.n_class = len(set(self.y))


#class LogisticRegression():
#    '''
#    The Logistic Regression Class,
#    member variables: X_train, y_train
#    member function: forward, loss_function, predict, accuracy
#    '''
#    def __init__(self, X_train, y_train, n_class):
#        self.X = X_train
#        self.y = y_train
#        self.n_case, self.n_feat = X_train.shape
#        self.n_class = n_class
#        
#    def pretrain(self):
#        print('prepare for TenforFlow')
#        self.input_X = tf.placeholder(dtype=tf.float32, shape=[None, n_feat])
#        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None])
#        self.input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
#        self.weight = tf.Variable(tf.zeros([n_feat, n_class]), name='weights')
#        self.b = tf.Variable(tf.zeros([n_class]), name='bias')
#        self.s = tf.Session()
#        self.s.run(tf.global_variables_initializer())        
#        
#    def forward(self, X):
#        self.pr = tf.nn.softmax(tf.matmul(self.input_X, self.weight)+self.b)
##        print(self.s.run(self.pr, {self.input_X:X}))
#        return self.pr
#    
#    def loss_function(self):
#        self.loss = tf.reduce_mean(tf.reduce_sum(-tf.log(self.pr)*self.y, axis=1))

if __name__ == '__main__':
    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file, True)
    
    n_case, n_feat = X_train.shape
    n_class = 10
    y_train_onehot = one_hot_coding(y_train, n_class)
    
    print('tensorflow implemented LR')
    
#    lr = LogisticRegression(X_train, y_train, 10)
#    lr.pretrain()
#    lr.forward()
#    
#    weight0 = tf.Variable(tf.zeros([n_feat, n_class]), name='weights')
#    b0 = tf.Variable(tf.zeros([n_class]), name='bias')
#    
#    input_X = tf.placeholder(dtype=tf.float32, shape=[None, n_feat])
#    input_y = tf.placeholder(dtype=tf.int64, shape=[None])
#    input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
#    
#    pr = forward(input_X, weight0, b0)
#    loss = cross_ent_loss(input_y_onehot, pr)
#    y_predict = predict(pr)
#    acc = accuracy(input_y, y_predict)
#    gama = 1
#    optimizer = tf.train.GradientDescentOptimizer(gama).minimize(loss)
#
#    s = tf.Session()
#    s.run(tf.global_variables_initializer())
#
#    batch_size = 32
#    batch_num = int(np.ceil(n_case/batch_size))
#    for j in range(batch_num):
#        X_batch = X_train[j*batch_size:min((j+1)*batch_size, n_case), :]
#        y_batch_onehot = y_train_onehot[j*batch_size:min((j+1)*batch_size, n_case), :]
#        y_batch = y_train[j*batch_size:min((j+1)*batch_size, n_case)]        
#        loss_i, acc_i = s.run([loss, acc], 
#                {input_X:X_train, input_y:y_train, input_y_onehot:y_train_onehot})
#        print(j, ' : ', loss_i, acc_i)
#        s.run(optimizer, {input_X:X_batch, input_y_onehot:y_batch_onehot})
    
    n_neuron = 50
    
    weight0 = tf.Variable(tf.random_normal([n_feat, n_neuron], 0, 1), name='w0')
    weight1 = tf.Variable(tf.random_normal([n_neuron, n_class], 0, 1), name='w1')
    b0 = tf.Variable(tf.zeros([n_neuron]), name='b0')
    b1 = tf.Variable(tf.zeros([n_class]), name='b1')
    
    input_X = tf.placeholder(dtype=tf.float32, shape=[None, n_feat])
    input_y = tf.placeholder(dtype=tf.int64, shape=[None])
    input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
    
    pr = forward_2(input_X, weight0, b0, weight1, b1)
    loss = cross_ent_loss(input_y_onehot, pr)
    y_predict = predict(pr)
    acc = accuracy(input_y, y_predict)
    gama = 1e-2
    optimizer = tf.train.AdamOptimizer(gama, beta1=0.5).minimize(loss)

    s = tf.Session()
    s.run(tf.global_variables_initializer())

    batch_size = 1000
    batch_num = int(np.ceil(n_case/batch_size))
    for loop in range(10):
        permutation = list(np.random.permutation(n_case))
        X_train_tmp = X_train[permutation, :]
        y_train_tmp = y_train[permutation]
        y_train_onehot_tmp = y_train_onehot[permutation, :]
        for j in range(batch_num):
            X_batch = X_train_tmp[j*batch_size:min((j+1)*batch_size, n_case), :]
            y_batch_onehot = y_train_onehot_tmp[j*batch_size:min((j+1)*batch_size, n_case), :]
            y_batch = y_train_tmp[j*batch_size:min((j+1)*batch_size, n_case)]        
            loss_i, acc_i = s.run([loss, acc], 
                    {input_X:X_train, input_y:y_train, input_y_onehot:y_train_onehot})
            print(j, ' : ', loss_i, acc_i)
            s.run(optimizer, {input_X:X_batch, input_y_onehot:y_batch_onehot})    
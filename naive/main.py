# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:03:49 2018

@author: ZhaoYu
"""

from ReadMNIST import *
from GradientDecent import *
from mlp import *

if __name__ == '__main__':
    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file, True)
    
    n_feat = X_train.shape[1]
    n_class = 10
    
##    beta = np.random.normal(0, 1, [n_feat, n_class])
##    beta0 = np.random.normal(0, 1, n_class)
#    
#    beta = np.zeros([n_feat, n_class], dtype='float32')
#    beta0 = np.zeros([n_class,], dtype='float32')
#    
##    grad_descent(X_train, y_train, beta, beta0)
##    sgd(X_train, y_train, beta, beta0)
#    mini_batch(X_train, y_train, beta, beta0)
#    
#    predict_y = np.argmax(softmax(np.dot(X_train, beta)+beta0), axis=1)
#    print(accuracy(X_train, y_train, beta, beta0))
    
    n_neuron = 50
    
    weight0 = np.random.normal(0, 1, [n_feat, n_neuron])
    b0 = np.random.normal(0, 1, [n_neuron])
#    h = full_connect_sigmoid(X_train, weight0, b0)
    weight1 = np.random.normal(0, 1, [n_neuron, n_class])
    b1 = np.random.normal(0, 1, [n_class])
#    pr = full_connect_softmax(h, weight1, b1)
    
    for i in range(10):
      mlp_mini_batch(X_train, y_train, weight0, b0, weight1, b1, 2000)
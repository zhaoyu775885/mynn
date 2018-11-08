#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:08:24 2018

@author: zhaoyu
"""

from GradientDecent import *

def sigmoid(x):
    return 1./(1+np.exp(-x))

def full_connect_sigmoid(X, beta, beta0):
    return sigmoid(np.matmul(X, beta)+beta0)

def full_connect_softmax(X, beta, beta0):
    return softmax(np.matmul(X, beta)+beta0)

def forward(X, weight0, b0, weight1, b1):
    h = full_connect_sigmoid(X, weight0, b0)
    pr = full_connect_softmax(h, weight1, b1)
    return h, pr

def backward(X, y, weight0, b0, weight1, b1):
    n_case = X.shape[0]
    n_class = 10
    n_hidden = weight0.shape[1]
    
    [h, pr] = forward(X, weight0, b0, weight1, b1)
    
    weight1_mean = np.matmul(weight1, pr.T)
    
    pd_loss_h = np.zeros([n_case, n_hidden])
    for i in range(n_case):
        pd_loss_h[i] = -(weight1[:, y[i]]-weight1_mean[:, i])
    
    for i in range(n_case):
        pr[i][y[i]] -= 1
    
    pd_loss_weight1 = 1/n_case * np.matmul(h.T, pr)
    pd_loss_b1 = 1/n_case * np.matmul(np.ones([n_case]), pr)
    
    tmp= pd_loss_h * h * (1-h)
    
    pd_loss_weight0 = 1/n_case * np.matmul(X.T, tmp)
    pd_loss_b0 = 1/n_case * np.matmul(np.ones([n_case]), tmp)
    
    return pd_loss_weight0, pd_loss_b0, pd_loss_weight1, pd_loss_b1

def mlp_cross_ent(y, pr):
    ent = 0
    for i in range(y.shape[0]):
        ent -= np.log(pr[i][y[i]])
    return ent

def mlp_accuracy(y, predict_y):
    return np.sum(y==predict_y)/y.shape[0]

def mlp_mini_batch(X, y, weight0, b0, weight1, b1, batch_size=32):
    n_case = X.shape[0]
    
    batch_num = int(np.ceil(n_case/batch_size))
    gama_0 = 5
    gama_1 = 2
    
    for i in range(batch_num):
        h, pr = forward(X, weight0, b0, weight1, b1)
        predict_y = np.argmax(pr, axis=1)
        loss = 1/n_case * mlp_cross_ent(y, pr)
        acc = mlp_accuracy(y, predict_y)
        print('{0}: loss={1:.6f} acc={2:.6f}'.format(i, loss, acc))
        
        X_batch = X[i*batch_size:min((i+1)*batch_size, n_case), :]
        y_batch = y[i*batch_size:min((i+1)*batch_size, n_case)]
        
        gd_weight0, gd_b0, gd_weight1, gd_b1 = backward(X_batch, y_batch, 
            weight0, b0, weight1, b1)
        weight0 -= gd_weight0*gama_0
        b0 -= gd_b0*gama_0
        weight1 -= gd_weight1*gama_1
        b1 -= gd_b1*gama_1

if __name__ == '__main__':
    print(sigmoid(1))
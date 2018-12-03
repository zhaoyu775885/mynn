# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:04:01 2018

@author: ZhaoYu
"""
import numpy as np

def softmax(y):
    if y.ndim == 1:
        return np.exp(y)/ np.sum(np.exp(y))
    if y.ndim == 2:
        z = np.zeros(y.shape)
        for row in range(y.shape[0]):
            z[row, :] = softmax(y[row, :])
        return z
    
def grad_softmax(x, y_label, weight, b):
    if x.ndim == 1:
        grad_b = -softmax(np.matmul(x, weight)+b)
        grad_b[y_label] += 1
        grad_weight = np.matmul(np.mat(x).T, np.mat(grad_b))
        return grad_weight, grad_b
    if x.ndim == 2 and y_label.ndim == 1:
        cnt = y_label.shape[0]
        tmp = -softmax(np.matmul(x, weight)+b)
        for i in range(cnt):
            tmp[i][y_label[i]] += 1
        grad_b = np.matmul(np.ones([cnt]), tmp)
        grad_weight = np.matmul(x.T, tmp)
        return grad_weight, grad_b
    print('parameter input error!')
    print(type(x), type(y_label))
    
def cross_ent(x, y_label, weight, b):
    pr = softmax(np.matmul(x, weight)+b)
    if x.ndim == 1 and type(y_label)==int:
        return -np.log(predict_y[y_label])
    if x.ndim == 2 and y_label.ndim == 1:
        loss = 0
        for i in range(y_label.shape[0]):
            loss -= np.log(pr[i][y_label[i]])
        return loss

def loss(x, y, weight, b):
    return 1/y.shape[0] * cross_ent(x, y, weight, b)

def accuracy(x, y, weight, b):
    y_predict = np.argmax(softmax(np.matmul(x, weight)+b), axis=1)
    return np.sum(y_predict == y)/y.shape[0]

def grad_descent(x, y_label, weight, b):
    gama = 1
    n_case = y_label.shape[0]
    for step in range(100):
        print('loss = ', loss(x, y_label, weight, b), ', acc = ', accuracy(x, y_label, weight, b))
        grad_weight, grad_b = grad_softmax(x, y_label, weight, b)
        weight += 1/n_case * gama * grad_weight
        b += 1/n_case * gama * grad_b
    return weight, b

def sgd(x, y_label, weight, b):
    gama = 1
    for i in range(y_label.shape[0]):
        print('loss = ', loss(x, y_label, weight, b), ', acc = ', accuracy(x, y_label, weight, b))
        grad_weight, grad_b = grad_softmax(x[i, :], y_label[i], weight, b)
        weight += gama * grad_weight
        b += gama * grad_b
    return weight, b

def mini_batch(x, y_label, weight, b):
    n_case = y_label.shape[0]
    batch_size = 32
    batch_num = int(np.ceil(n_case/batch_size))
    gama = 1
    for i in range(batch_num):
        print('{0} : {1:.6f}, {2:.6f}'.format(i, loss(x, y_label, weight, b), accuracy(x, y_label, weight, b)))
        x_batch = x[i*batch_size:min(i*batch_size+batch_size, n_case), :]
        y_batch = y_label[i*batch_size:min(i*batch_size+batch_size, n_case)]
        tmp_batch_size = min(i*batch_size+batch_size, n_case) - i*batch_size
        grad_weight, grad_b = grad_softmax(x_batch, y_batch, weight, b)
        weight += 1/tmp_batch_size * gama * grad_weight
        b += 1/tmp_batch_size * gama * grad_b
    return weight, b
    
if __name__ == '__main__':
    a = np.array([[1, 1, 1], [2, 2, 1]])
    print(softmax(a).ndim)
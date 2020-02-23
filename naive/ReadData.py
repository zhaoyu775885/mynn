# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:36:17 2018

@author: ZhaoYu
"""

import numpy as np
import pandas as pd

def read_mnist(file, normal_flag=True, conv_flag=True):
    pf = pd.read_csv(file)
    X_data = np.array(pf.iloc[:, 1:], dtype='float32')
    if conv_flag:
        X_data = X_data.reshape(-1, 28, 28, 1)
    print(X_data.shape)
    y_data = np.array(pf.iloc[:, 0], dtype='int32')
    if normal_flag:
        X_data = normalize(X_data)
    return X_data, y_data

def read_mnist_test(file, normal_flag=False):
    pf = pd.read_csv(file)
    X_data = np.array(pf.iloc[:, :], dtype='float32')
    if normal_flag:
        X_data = normalize(X_data)
    return X_data

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_cifar10(filename, normal_flag=True):
    data = unpickle(filename)
    img = data[b'data']
    labels = np.array(data[b'labels'])
    n_samp = len(labels)
    img.resize([n_samp, 3, 32, 32])
    if normal_flag:
        img = normalize(img)
    img = np.swapaxes(img, 1, -1)
    return img, labels

def normalize(data):
#    data = (data-np.min(data))/(np.max(data)-np.min(data))
    data = data / 255 - 0.5
    return data

def read_mnist_naive(file, normal_flag=False):
    fh = open(file, 'r')
    train_data = []
    for line in fh:
        try:
            line_list = [int(it) for it in line.strip('\n').split(',')]
            train_data.append(line_list)
        except:
            continue
    train_data = np.array(train_data)
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    if normal_flag:
        X_train = normalize(X_train)
    return X_train, y_train

if __name__ == '__main__':
    file = './data/train.csv'
    X, y = read_mnist_naive(file, True)

    

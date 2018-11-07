# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:36:17 2018

@author: ZhaoYu
"""

import numpy as np
import pandas as pd

def read_mnist(file, normal_flag=False):
    pf = pd.read_csv(file)
    X_data = np.array(pf.iloc[:, 1:], dtype='float32')
    y_data = np.array(pf.iloc[:, 0], dtype='int32')
    if normal_flag:
        X_data = normalize(X_data)
    return X_data, y_data

def normalize(data):
    data = (data-np.min(data))/(np.max(data)-np.min(data))
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

    

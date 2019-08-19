# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:43:10 2018

@author: ZhaoYu

More practice makes higher proficiency

"""

import tensorflow as tf
#import keras as kr
from cnnTF_OO import *
from ReadData import *

if __name__ == '__main__':
    '''
    # 1
    # for MNIST
    print('tensorflow implemented CNN')
    train_file = './data/train.csv'
    test_file = './data/test.csv'
    X_train, y_train= read_mnist(train_file, normal_flag=True)
    X_test = read_mnist_test(test_file)
    cnn2 = ConvNet2()
    cnn2.addlayer('conv', {'ksz': [3, 3], 'ssz': [1, 1], 'kn': 16})
    cnn2.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    cnn2.addlayer('conv', {'ksz': [3, 3], 'ssz': [1, 1], 'kn': 32})
    cnn2.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    cnn2.addlayer('dense', {'nn': 128})
    cnn2.addlayer('softmax', {'nn': 10})

    cnn2.fit(X_train, y_train)
    y_pred = cnn2.pred(X_test)
    d = {'ImageId':np.arange(1,len(y_pred)+1), 'Label': y_pred}
    pf = pd.DataFrame(d)
    pf.to_csv('./data/predict.csv', index=False)
    '''
    
    # 2
    # for CIFAR-10
    train_file_1 = './data/cifar-10/data_batch_1'
    train_file_2 = './data/cifar-10/data_batch_2'
    train_file_3 = './data/cifar-10/data_batch_3'
    train_file_4 = './data/cifar-10/data_batch_4'
    train_file_5 = './data/cifar-10/data_batch_5'
    X_data_1, y_data_1 = read_cifar10(train_file_1, True)
    X_data_2, y_data_2 = read_cifar10(train_file_2, True)
    X_data_3, y_data_3 = read_cifar10(train_file_3, True)
    X_data_4, y_data_4 = read_cifar10(train_file_4, True)
    X_data_5, y_data_5 = read_cifar10(train_file_5, True)
    X_data= np.concatenate((X_data_1, X_data_2, X_data_3, 
                            X_data_4, X_data_5), axis=0)
    y_data= np.concatenate((y_data_1, y_data_2, y_data_3, 
                            y_data_4, y_data_5), axis=0)
    cnn2 = ConvNet2()
    cnn2.addlayer('conv', {'ksz': [5, 5], 'ssz': [1, 1], 'kn': 16, 'std':1e-3})
    cnn2.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    cnn2.addlayer('conv', {'ksz': [5, 5], 'ssz': [1, 1], 'kn': 32, 'std':1e-3})
    cnn2.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    cnn2.addlayer('dense', {'nn': 256, 'std':1e-3})
    cnn2.addlayer('softmax', {'nn': 10, 'std':1e-3})
    
    cnn2.fit(X_data, y_data, 1e-3, epoch=50, bsz=64)
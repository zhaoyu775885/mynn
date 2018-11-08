#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:45:42 2018

@author: zhaoyu
"""

from ReadMNIST import *
import tensorflow as tf
import numpy as np
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Activation

if __name__ == '__main__':
    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file, True)
    y_train = kr.utils.to_categorical(y_train)
    
    model = Sequential()
    model.add(Dense(50, input_dim=784))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    
    for loop in range(10):
        model.fit(X_train, y_train, epochs=5)
        
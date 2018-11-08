# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:40:52 2018

@author: ZhaoYu
"""
#
#import tensorflow as tf
#import numpy as np
#
#def mini_batch(loss, X_train, y_train, batch_size=32):
#    n_case = X_train.shape[0]
#    batch_num = np.ceil(n_case/batch_size)
#    
#    for epoch in range(batch_num):
#        X_batch = X_train[j*batch_size:min((j+1)*batch_size, n_case), :]
#        y_batch_onehot = y_train_onehot[j*batch_size:min((j+1)*batch_size, n_case), :]
#        y_batch = y_train[j*batch_size:min((j+1)*batch_size, n_case)]        
#        loss_i, acc_i = s.run([loss, acc], 
#                {input_X:X_train, input_y:y_train, input_y_onehot:y_train_onehot})
#        print(j, ' : ', loss_i, acc_i)
#        s.run(optimizer, {input_X:X_batch, input_y_onehot:y_batch_onehot})        
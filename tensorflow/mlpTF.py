# -*- coding: utf-8 -*-

"""
Created on Tue Nov 13 14:22:16 2018
Multilayer Perceptron Implementation
with Tensorflow Framework
@author: zhaoyu
"""

import tensorflow as tf
import numpy as np

def loss_cross_entropy(pred, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

def one_hot_encoding(y, n_class):
    one_hot = np.zeros([len(y), n_class], dtype='float32')
    for i in range(len(y)):
        one_hot[i][y[i]] = 1.0
    return one_hot

def predict(prob):
    return tf.cast(tf.argmax(prob, axis=1), tf.int32)

def accuracy(y, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))



class MultilayerPerceptron:
    def __init__(self):
        pass

    def init_params(self):
        self.init_input_params()
        self.init_tf_param()

    def init_input_params(self):
        self.n_samp, self.n_feat = self.X.shape
        self.n_class = len(set(self.y))
        self.n_layers = len(self.n_hiddens) + 1 #zero hidden layer is just 1-layer mlp, that is LR
        self.y_onehot = one_hot_encoding(self.y, self.n_class)
        self.w_list = []
        self.b_list = []

    def init_tf_param(self):
        '''
        # one-hot encoding in tf.float32 is too wierd!
        '''
        self.input_X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_feat])       
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, self.n_class])
        '''
        # n_hiddens is a list containing the number of neurons on each layer
        # thus a list of weight matrices and bias vectors are required to represent each layer
        '''
        nrow = self.n_feat
        for each in self.n_hiddens:
            w = tf.Variable(tf.random_normal([nrow, each], 0, 1), name='w')
            b = tf.Variable(tf.random_normal([each], 0, 1), name='b')
            self.w_list.append(w)
            self.b_list.append(b)
            nrow = each
        '''
        # the output layer is diff from the hidden layer
        # and need to be specially treated.
        '''
        w = tf.Variable(tf.random_normal([nrow, self.n_class], 0, 1), name='w')
        b = b = tf.Variable(tf.random_normal([self.n_class], 0, 1), name='b')
        self.w_list.append(w)
        self.b_list.append(b)

    def forward(self):
        '''
        # forward-propagation procedure for prediction
        '''
        f = self.input_X
        for (w, b) in zip(self.w_list[:-1], self.b_list[:-1]):
            f = tf.nn.sigmoid(tf.matmul(f, w)+b)
        f = tf.matmul(f, self.w_list[-1])+self.b_list[-1]
#        f = tf.nn.softmax(tf.matmul(f, self.w_list[-1])+self.b_list[-1])
        return f

    def def_graph(self):
        '''
        # define placeholder (contained for input parameters), variables, 
        # loss functions, esitamation functions and predictions.
        # Symbols and logics defined for computin graph in Tensorflow.
        '''
        self.pred = self.forward()
        self.prob = tf.nn.softmax(self.pred)
        self.loss = loss_cross_entropy(self.pred, self.input_y_onehot)
        self.y_pred = predict(self.prob)
        self.acc = accuracy(self.input_y, self.y_pred)
        self.gama = 1
        self.optimizer = tf.train.GradientDescentOptimizer(self.gama).minimize(self.loss)

    def fit(self, X_data, y_data, hidden_list=[]):
        '''
        # Just mimic the fitting procedure in sklearn
        # here, it is back-propagation.
        # Details are shown in the naive-numpy version of MLP
        '''
        self.X, self.y, self.n_hiddens = X_data, y_data, hidden_list
        self.init_params()
        self.def_graph()
        self.s = tf.Session()
        self.s.run(tf.global_variables_initializer())
        self.mini_batch()     
    
    def mini_batch(self):
        '''
        # Stochatic Gradient Descent
        # I will implement all versions of optimizer in the naive version
        '''
        loops_cnt = 100
        batch_size = 32
        for i in range(loops_cnt):
            seq = np.arange(self.n_samp)
            np.random.shuffle(seq)
#            batch_num = int(np.ceil(self.n_samp/batch_size))
            for epoch in range(100):
                scope = seq[range(epoch*batch_size, min((epoch+1)*batch_size, self.n_samp))]
                X_batch = self.X[scope, :]
                y_batch_onehot = self.y_onehot[scope, :]
                self.s.run(self.optimizer, 
                           {self.input_X:X_batch, self.input_y_onehot:y_batch_onehot})
            loss_i, acc_i, prob = self.s.run([self.loss, self.acc, self.prob],
                {self.input_X:self.X, self.input_y:self.y, self.input_y_onehot:self.y_onehot})
#            print(float(0) in prob) # check NaN in logits
            print(i, ' : ', loss_i, acc_i)

    def test(self, X_test, y_test):
        y_test_onehot = one_hot_encoding(y_test, self.n_class)
        loss, acc, prob = self.s.run([self.loss, self.acc, self.prob], {self.input_X:X_test, self.input_y:y_test, self.input_y_onehot:y_test_onehot})
        print(loss, acc)

'''
# procedural-oriented programming
def mlp_tf(X, y):
    n_samp, n_feat = X.shape
    n_class = len(set(y))
    y_onehot = one_hot_encoding(y, n_class)
    input_X = tf.placeholder(dtype=tf.float32, shape=[None, n_feat])
    input_y = tf.placeholder(dtype=tf.int32, shape=[None])
    input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
#    w = tf.Variable(tf.random_normal([n_feat, n_class], 0, 1), name='w')
#    b = tf.Variable(tf.random_normal([n_class], 0, 1), name='b')
#    prob = tf.nn.softmax(tf.matmul(input_X, w)+b)
    n_neuron = 50
    w0 = tf.Variable(tf.random_normal([n_feat, n_neuron], 0, 1), name='w0')
    b0 = tf.Variable(tf.random_normal([n_neuron], 0, 1), name='b0')
    w1 = tf.Variable(tf.random_normal([n_neuron, n_class], 0, 1), name='w1')
    b1 = tf.Variable(tf.random_normal([n_class], 0, 1), name='b1')
    prob = tf.nn.softmax(tf.matmul(tf.nn.sigmoid(tf.matmul(input_X, w0)+b0), w1)+b1)
    L = loss_cross_entropy(input_y_onehot, prob)
    y_pred = predict(prob)
    acc = accuracy(input_y, y_pred)
    gama = 1
    optimizer = tf.train.GradientDescentOptimizer(gama).minimize(L) 
    s = tf.Session()
    s.run(tf.global_variables_initializer())
    batch_size = 32
    batch_num = int(np.ceil(n_samp/batch_size))
    for epoch in range(batch_num):
        scope = range(epoch*batch_size, min((epoch+1)*batch_size, n_samp))
        X_batch = X[scope, :]
        y_batch_onehot = y_onehot[scope, :]
        loss_i, acc_i = s.run([L, acc],
                {input_X:X, input_y:y, input_y_onehot:y_onehot})
        print(epoch, ' : ', loss_i, acc_i)
        s.run(optimizer, {input_X:X_batch, input_y_onehot:y_batch_onehot})
'''

if __name__ == '__main__':
    print('tensorflow implemented mlp')
    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file, True)   
    mlp = MultilayerPerceptron()
    mlp.fit(X_train, y_train, [100, 50])
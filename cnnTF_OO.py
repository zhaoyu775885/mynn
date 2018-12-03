#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 22:25:27 2018

@author: zhaoyu
"""

from ReadMNIST import *
import tensorflow as tf

def one_hot_encoding(y, n_class):
    one_hot = np.zeros([len(y), n_class], dtype='float32')
    for i in range(len(y)):
        one_hot[i][y[i]] = 1.0
    return one_hot

def chk(layer, argu):
    if layer == 'dense' or layer=='softmax':
        if 'nn' not in argu.keys():
            print('Error: nn keywords are required!')
            return False
    elif layer=='conv':
        if 'ksz' not in argu.keys():
            print('Error: ksz keywords are required!')
            return False
        if 'ssz' not in argu.keys():
            print('Error: ssz keywords are required!')
            return False
        if 'kn' not in argu.keys():
            print('Error: kn keywords are required!')
            return False
    elif layer == 'pool':
        if 'type' not in argu.keys():
            print('Error: type of pooling are required!')
            return False
    return True

class DepNet:
    def __init__(self):
        self.dep = []

    def addlayer(self, lyr, arg):
        '''
        argu 本身就是一个字典
        'ksz': kernel size
        'ssz': stride size
        'cn' : channel number
        'kn' : kernel number
        '''

        if chk(lyr, arg):
            tmp_dict = {}
            tmp_dict['lyr'] = lyr
            tmp_dict['arg'] = arg
            self.dep.append(tmp_dict)
        else:
            exit(1)

    def get_dep(self):
        return self.dep



class ConvNet:
    def __init__(self):
        pass

    def init_params(self, image_size):
        self.init_input_params(image_size)
        self.init_tf_params()

    def init_input_params(self, image_size):
        self.n_samp, self.n_feat = self.X.shape
        self.img_width, self.img_height = image_size
        self.n_class = len(set(self.y))
        # self.n_layers = len(self.hiddens) + 1
        self.y_onehot = one_hot_encoding(self.y, self.n_class)
        self.w_list = []
        self.b_list = []

    def init_tf_params(self):
        self.input_X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_feat])
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, self.n_class])
        self.input_X_img = tf.reshape(self.input_X, [-1, self.img_width, self.img_height, 1])

        self.w1 = tf.Variable(tf.random_normal([5, 5, 1, 32], 0, 1, dtype=tf.float32))
        self.b1 = tf.Variable(tf.random_normal([32, ], 0, 1, dtype=tf.float32))

        self.w_fc = tf.Variable(tf.random_normal([14*14*32, 1024], 0, 1))
        self.b_fc = tf.Variable(tf.random_normal([1024, ], 0, 1))

        self.keep_prob = tf.placeholder(tf.float32)

        self.w_fc2 = tf.Variable(tf.random_normal([1024, self.n_class], 0, 1))
        self.b_fc2 = tf.Variable(tf.random_normal([self.n_class], 0, 1))

    def def_graph(self):
        a1 = tf.nn.relu(tf.nn.conv2d(self.input_X_img, self.w1, strides=[1, 1, 1, 1], padding='SAME') + self.b1)
        h1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h1_flat = tf.reshape(h1, [-1, 14 * 14 * 32])
        h_fc = tf.nn.relu(tf.matmul(h1_flat, self.w_fc) + self.b_fc)
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)
        f = tf.matmul(h_fc_drop, self.w_fc2) + self.b_fc2
        self.prob = tf.nn.softmax(f)
        self.y_pred = tf.cast(tf.argmax(self.prob, axis=1), tf.int32)
        self.accu = tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.y_pred), tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=f, labels=self.input_y_onehot))
        self.optimizer = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    def fit(self, X_data, y_data, image_size=[28, 28], hiddens=[]):
        self.X, self.y, self.hiddens = X_data, y_data, hiddens
        self.init_params(image_size)
        self.def_graph()
        self.s = tf.InteractiveSession()
        self.s.run(tf.global_variables_initializer())
        self.mini_batch()

    def mini_batch(self):
        loops_cnt = 10
        batch_size = 128
        batch_num = int(np.ceil(self.n_samp / batch_size))
        for i in range(loops_cnt):
            seq = np.arange(self.n_samp)
            np.random.shuffle(seq)
            for epoch in range(batch_num):
                scope = seq[range(epoch * batch_size, min((epoch + 1) * batch_size, self.n_samp))]
                X_batch = self.X[scope, :]
                y_batch = self.y[scope]
                y_batch_onehot = self.y_onehot[scope, :]
                self.s.run(self.optimizer,
                    {self.input_X: X_batch, self.input_y: y_batch,
                    self.input_y_onehot: y_batch_onehot, self.keep_prob:0.5})
                if epoch % 50 == 0:
                    loss_i, acc_i = self.s.run([self.loss, self.accu],
                                        {self.input_X: X_batch, self.input_y: y_batch,
                                        self.input_y_onehot: y_batch_onehot, self.keep_prob:1})
                    print(loss_i, acc_i)


class ConvNet2:
    def __init__(self):
        pass

    def init_params(self):
        self.init_input_params()
        self.init_tf_params()

    def init_input_params(self):
        self.n_samp, self.n_feat = self.X.shape
        self.img_width, self.img_height = self.img_sz
        self.n_class = len(set(self.y))
        self.y_onehot = one_hot_encoding(self.y, self.n_class)
        # self.n_layers = len(self.hiddens) + 1
        self.w_list = []
        self.b_list = []

    def init_tf_params(self):
        self.input_X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_feat])
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, self.n_class])
        self.input_X_img = tf.reshape(self.input_X, [-1, self.img_width, self.img_height, 1])
        width, height = self.img_width, self.img_height
        prev_lyr_type = ''
        cn = 1
        row, col = 0, 1
        for each in self.lyrs.get_dep():
            #            print(each)
            cur_lyr_type = each['lyr']
            if cur_lyr_type == 'conv':
                ksz_w, ksz_h = each['arg']['ksz']
                ssz_w, ssz_h = each['arg']['ssz']
                kn = each['arg']['kn']
                w = tf.Variable(tf.random_normal([ksz_w, ksz_h, cn, kn], 0, 1, dtype=tf.float32))
                b = tf.Variable(tf.random_normal([kn], 0, 1, dtype=tf.float32))
                self.w_list.append(w)
                self.b_list.append(b)
                print('{4} layer, kernel size: [{0}, {1}, {2}, {3}]'.format(ksz_w, ksz_h, cn, kn, cur_lyr_type))
                cn = kn
            elif cur_lyr_type == 'pool':
                ksz_w, ksz_h = each['arg']['ksz']
                ssz_w, ssz_h = each['arg']['ssz']
                width = int(np.ceil((width - ksz_w) / ssz_w)) + 1
                height = int(np.ceil((height - ksz_h) / ssz_h)) + 1
                print('{0} layer'.format(cur_lyr_type))
                self.w_list.append('no params')
                self.b_list.append('no params')
            elif cur_lyr_type == 'dense' or cur_lyr_type == 'softmax':
                if prev_lyr_type != 'dense':
                    row = width * height * cn
                else:
                    row = col
                col = each['arg']['nn']
                w = tf.Variable(tf.random_normal([row, col], 0, 1, dtype=tf.float32))
                b = tf.Variable(tf.random_normal([col], 0, 1, dtype=tf.float32))
                self.w_list.append(w)
                self.b_list.append(b)
                print('{2} layer, tensorflow size: [{0}, {1}]'.format(row, col, cur_lyr_type))
            prev_lyr_type = cur_lyr_type

    def forward(self):
        dep = self.lyrs.get_dep()
        f = self.input_X_img
        for ilyr in range(len(dep)):
            lyr = dep[ilyr]
            lyr_type = lyr['lyr']
            w = self.w_list[ilyr]
            b = self.b_list[ilyr]
            if lyr_type == 'conv':
                ssz_w, ssz_h = dep[ilyr]['arg']['ssz']
                f = tf.nn.relu(tf.nn.conv2d(f, w, strides=[1, ssz_w, ssz_h, 1],
                                            padding='SAME') + b)
            elif lyr_type == 'pool':
                ksz_w, ksz_h = lyr['arg']['ksz']
                ssz_w, ssz_h = lyr['arg']['ssz']
                f = tf.nn.max_pool(f, ksize=[1, ksz_w, ksz_h, 1],
                                strides=[1, ssz_w, ssz_h, 1], padding='SAME')
            elif lyr_type == 'dense':
                row, col = w.shape
                f = tf.reshape(f, [-1, row])
                f = tf.nn.relu(tf.matmul(f, w) + b)
            elif lyr_type == 'softmax':
                row, col = w.shape
                f = tf.reshape(f, [-1, row])
                f = tf.matmul(f, w) + b
            print(ilyr, lyr_type, w, b, f.shape)
        return f

    def def_graph(self):
        pred = self.forward()
        self.prob = pred
        self.y_pred = tf.cast(tf.argmax(self.prob, axis=1), tf.int32)
        self.accu = tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.y_pred), tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.input_y_onehot))
        self.optimizer = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    def fit(self, X_data, y_data, lyr_dep, image_size=[28, 28]):
        self.X, self.y, self.img_sz, self.lyrs = X_data, y_data, image_size, lyr_dep
        self.init_params()
        self.def_graph()
        self.s = tf.InteractiveSession()
        self.s.run(tf.global_variables_initializer())
        self.mini_batch()
        self.s.close()

    def mini_batch(self):
        loops_cnt = 1
        batch_size = 128
        batch_num = int(np.ceil(self.n_samp / batch_size))
        for i in range(loops_cnt):
            seq = np.arange(self.n_samp)
            np.random.shuffle(seq)
            for epoch in range(batch_num):
                scope = seq[range(epoch * batch_size, min((epoch + 1) * batch_size, self.n_samp))]
                X_batch = self.X[scope, :]
                y_batch = self.y[scope]
                y_batch_onehot = self.y_onehot[scope, :]
                self.s.run(self.optimizer,
                        {self.input_X: X_batch, self.input_y: y_batch,
                            self.input_y_onehot: y_batch_onehot})
                if epoch % 50 == 0:
                    loss_i, acc_i = self.s.run([self.loss, self.accu],
                                            {self.input_X: X_batch, self.input_y: y_batch,
                                                self.input_y_onehot: y_batch_onehot})
                    print('epoch: ', loss_i, acc_i)

    def _debug_(self):
        pred = self.s.run(self.prob, {self.input_X: self.X, self.input_y: self.y, self.input_y_onehot: self.y_onehot})
        return pred

if '__main__' == __name__:
    print('tensorflow implemented CNN')
    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file)

    # cnn = ConvNet()
    # cnn.fit(X_train, y_train, image_size=[28, 28])

    dep = DepNet()
    '''
    # 暂时未加入conv layer和pool layer的类型选择，'SAME' or 'VALID'
    # 后期的设计主要围绕：
    #  1. 如何兼容更多类型的layer
    #  2. 如何规范化约束各个layer
    #  3. 如何提高layer定义的自由度 
    '''
    dep.addlayer('conv', {'ksz': [3, 3], 'ssz': [1, 1], 'kn': 10})
    dep.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    dep.addlayer('conv', {'ksz': [3, 3], 'ssz': [1, 1], 'kn': 20})
    dep.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    dep.addlayer('dense', {'nn': 256})
    dep.addlayer('dense', {'nn': 64})
    dep.addlayer('softmax', {'nn': 10})
    # tst = dep.get_dep()

    cnn2 = ConvNet2()
    cnn2.fit(X_train, y_train, dep, image_size=[28, 28])

    '''
    idx = 2
    im = X_train[idx].reshape([28,28])
    print(y_train[idx])
    plt.imshow(im)
    '''
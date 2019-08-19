# -*- coding: utf-8 -*-

from ReadData import *
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

class ConvNet:
    def __init__(self):
        pass

    def init_params(self):
        self.init_input_params()
        self.init_tf_params()

    def init_input_params(self):
        self.n_samp, self.img_width, self.img_height, self.cn = self.X.shape
        self.n_class = len(set(self.y))
        self.y_onehot = one_hot_encoding(self.y, self.n_class)
        self.w_list = []
        self.b_list = []

    def init_tf_params(self):
        self.input_X = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.img_width, self.img_height, self.cn])
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, self.n_class])

        self.w1 = tf.Variable(tf.random_normal([5, 5, 1, 32], 0, 1, dtype=tf.float32))
        self.b1 = tf.Variable(tf.random_normal([32, ], 0, 1, dtype=tf.float32))

        self.w_fc = tf.Variable(tf.random_normal([14*14*32, 1024], 0, 1))
        self.b_fc = tf.Variable(tf.random_normal([1024, ], 0, 1))

        self.keep_prob = tf.placeholder(tf.float32)

        self.w_fc2 = tf.Variable(tf.random_normal([1024, self.n_class], 0, 1))
        self.b_fc2 = tf.Variable(tf.random_normal([self.n_class], 0, 1))

    def def_graph(self):
        a1 = tf.nn.relu(tf.nn.conv2d(self.input_X, self.w1, strides=[1, 1, 1, 1], padding='SAME') + self.b1)
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

    def fit(self, X_data, y_data, hiddens=[]):
        self.X, self.y, self.hiddens = X_data, y_data, hiddens
        self.init_params()
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
    '''
    # 暂时未加入conv layer和pool layer的类型选择，'SAME' or 'VALID'
    # 后期的设计主要围绕：
    #  1. 如何兼容更多类型的layer
    #  2. 如何规范化约束各个layer
    #  3. 如何提高layer定义的自由度
    #  4. 加入正则化方法：BN, GN,
    '''
    def __init__(self):
        self.lyrs = []

    def addlayer(self, lyr, arg):
        '''
        # argu 本身就是一个字典
        # 'ksz': kernel size
        # 'ssz': stride size
        # 'cn' : channel number
        # 'kn' : kernel number
        '''
        if chk(lyr, arg):
            tmp_dict = {}
            tmp_dict['lyr'] = lyr
            tmp_dict['arg'] = arg
            self.lyrs.append(tmp_dict)
        else:
            exit(1)

    def get_lyrs(self):
        return self.lyrs

    def init_params(self):
        self.init_input_params()
        self.init_tf_params()

    def init_input_params(self):
        self.n_samp, self.img_width, self.img_height, self.cn = self.X.shape
        self.n_class = len(set(self.y))
        self.y_onehot = one_hot_encoding(self.y, self.n_class)
        self.w_list = []
        self.b_list = []

    def init_tf_params(self):
        self.input_X = tf.placeholder(dtype=tf.float32,
            shape=[None, self.img_width, self.img_height, self.cn])
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_y_onehot = tf.placeholder(dtype=tf.float32, shape=[None, self.n_class])
        width, height = self.img_width, self.img_height
        prev_lyr_type = ''
        cn = self.cn
        row, col = 0, 1
        for each in self.lyrs:
            #            print(each)
            cur_lyr_type = each['lyr']
            cur_lyr_arg = each['arg']
            if cur_lyr_type == 'conv':
                ksz_w, ksz_h = each['arg']['ksz']
                ssz_w, ssz_h = each['arg']['ssz']
                kn = each['arg']['kn']
                w = tf.Variable(tf.random_normal([ksz_w, ksz_h, cn, kn], 0, cur_lyr_arg['std'], dtype=tf.float32))
                b = tf.Variable(tf.random_normal([kn], 0, cur_lyr_arg['std'], dtype=tf.float32))
                self.w_list.append(w)
                self.b_list.append(b)
                width = int(np.ceil(width/ssz_w))
                height = int(np.ceil(height/ssz_h))
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
                w = tf.Variable(tf.random_normal([row, col], 0, cur_lyr_arg['std'], dtype=tf.float32))
                b = tf.Variable(tf.random_normal([col], 0, cur_lyr_arg['std'], dtype=tf.float32))
                self.w_list.append(w)
                self.b_list.append(b)
                print('{2} layer, tensorflow size: [{0}, {1}]'.format(row, col, cur_lyr_type))
            prev_lyr_type = cur_lyr_type

    def forward(self):
        lyrs = self.lyrs
        f = self.input_X
        for ilyr in range(len(lyrs)):
            lyr = lyrs[ilyr]
            lyr_type = lyr['lyr']
            w = self.w_list[ilyr]
            b = self.b_list[ilyr]
            if lyr_type == 'conv':
                ssz_w, ssz_h = lyrs[ilyr]['arg']['ssz']
                f = tf.nn.relu(tf.nn.conv2d(f, w, strides=[1, ssz_w, ssz_h, 1],
                    padding='SAME') + b)
            elif lyr_type == 'pool':
                ksz_w, ksz_h = lyr['arg']['ksz']
                ssz_w, ssz_h = lyr['arg']['ssz']
                pool_type = lyr['arg']['type']
                if pool_type == 'max':
                    f = tf.nn.max_pool(f, ksize=[1, ksz_w, ksz_h, 1],
                        strides=[1, ssz_w, ssz_h, 1], padding='SAME')
                elif pool_type == 'avg':
                    f = tf.nn.avg_pool(f, ksize=[1, ksz_w, ksz_h, 1],
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
        self.prob = tf.nn.softmax(pred)
        self.y_pred = tf.cast(tf.argmax(self.prob, axis=1), tf.int32)
        self.accu = tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.y_pred), tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.input_y_onehot))
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def fit(self, X_data, y_data, rate, epoch=50, bsz=32):
        self.X, self.y= X_data, y_data
        self.init_params()
        print('==========================')
        self.def_graph()
        # writer = tf.summary.FileWriter('./tf.log', tf.get_default_graph())
        # writer.close()
        self.s = tf.Session()
        self.s.run(tf.global_variables_initializer())
        self.mini_batch(epoch, bsz)
        self.s.close()

    def close(self):
        self.s.close()

    def mini_batch(self, epochs, batch_size=32):
        batch_num = int(np.ceil(self.n_samp / batch_size))
        for epoch in range(epochs):
            seq = np.arange(self.n_samp)
            np.random.shuffle(seq)
            for it in range(batch_num):
                scope = seq[it * batch_size: min((it + 1) * batch_size, self.n_samp)]
                X_batch = self.X[scope, :]
                y_batch = self.y[scope]
                y_batch_onehot = self.y_onehot[scope, :]
                self.s.run(self.optimizer,
                           {self.input_X: X_batch, self.input_y: y_batch,
                            self.input_y_onehot: y_batch_onehot})
                if it == 0:
                    loss_i, acc_i = self.s.run([self.loss, self.accu],
                                               {self.input_X: X_batch, self.input_y: y_batch,
                                                self.input_y_onehot: y_batch_onehot})
                    print('epoch ', epoch, ' iteration ', it, ": ", loss_i, acc_i)

    def test(self, X_test, y_test):
        y_test_onehot = one_hot_encoding(y_test, self.n_class)
        loss, acc = self.s.run([self.loss, self.accu],
                               {self.input_X: X_test, self.input_y: y_test,
                                self.input_y_onehot: y_test_onehot})
        print(loss, acc)

    def _debug_(self):
        pred = self.s.run(self.prob, {self.input_X: self.X, self.input_y: self.y, self.input_y_onehot: self.y_onehot})
        return pred

if '__main__' == __name__:
    print('tensorflow implemented CNN')
    train_file = './data/train.csv'
    X_data, y_data = read_mnist(train_file, True)
    # train_file = './data/cifar-10/data_batch_1'
    # X_data, y_data = read_cifar10(train_file, True)
    # n_samp = X_data.shape[0]
    # seq = np.arange(n_samp)
    # np.random.shuffle(seq)
    # seq_train = seq[:35000]
    # seq_test = seq[35000:]
    # X_train = X_data[seq_train, :]
    # y_train = y_data[seq_train]
    # X_test = X_data[seq_test, :]
    # y_test = y_data[seq_test]

    # cnn = ConvNet()
    # cnn.fit(X_data, y_data)

    cnn2 = ConvNet2()
    cnn2.addlayer('conv', {'ksz': [3, 3], 'ssz': [1, 1], 'kn': 32})
    cnn2.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    cnn2.addlayer('conv', {'ksz': [3, 3], 'ssz': [1, 1], 'kn': 64})
    cnn2.addlayer('pool', {'ksz': [2, 2], 'ssz': [2, 2], 'type': 'max'})
    cnn2.addlayer('dense', {'nn': 256})
    cnn2.addlayer('softmax', {'nn': 10})
    cnn2.fit(X_data, y_data)
    # # cnn2.test(X_test, y_test)
    # cnn2.close()
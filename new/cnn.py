##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Sun Nov 18 22:09:38 2018
#
#@author: zhaoyu
#"""
#
#from ReadMNIST import *
#import tensorflow as tf
#
#if '__main__' == __name__:
#    print('tensorflow implemented CNN')
#    
#    train_file = './data/train.csv'
#    X_train, y_train = read_mnist(train_file, True)
#    
#    # data preparation
#    n_samp, n_feat = X_train.shape
#    n_class = len(set(y_train))
#    y_train_onehot = one_hot_encoding(y_train, n_class)
#    
#    # define basic containers (placeholders) computing graphs
#    input_X = tf.placeholder(tf.float32, [None, n_feat])
#    input_y = tf.placeholder(tf.int32, [None])
#    input_y_onehot = tf.placeholder(tf.float32, [None])
#    
#    # preprare for image process
#    input_X_image=tf.reshape(input_X, [-1,28,28,1])
#    
#    #define layers and parameters
#    # The filters in conv layer is defined by myself,
#    # Diffing from the MLP, in wich the weight matrix has its own shape fixed by
#    # the numbers of neurons of L-th and (L+1)-th layer.
#    W1 = tf.Variable(tf.random_normal([5, 5, 1, 32], 0, 1, dtype=tf.float32))
#    b1 = tf.Variable(tf.random_normal([32, ], 0, 1, dtype=tf.float32))
#    h1_conv = tf.nn.sigmoid(tf.nn.conv2d(input_X_image, W1, [1, 1, 1, 1], 'SAME')+b1)
#    h1_pool = tf.nn.max_pool(h1_conv, ksize=[1, 2, 2, 1], 'SAME')
#    
#    # fully-connected layer 1
#    h1_pool_flat = tf.reshape(h1_pool, [-1, 14*14*32])
#    W_fc = tf.Variable(tf.random_normal([14*14*32, 1024], 0, 1))
#    b_fc = tf.Variable(tf.random_normal([1024], 0, 1))
#    h_fc = tf.nn.sigmoid(tf.matmul(h1_pool_flat, W_fc)+b_fc)
#    
#    # fully-connected layer 2
#    W_fc2 = tf.Variable(tf.random_normal([1024, 10], 0, 1))
#    b_fc2 = tf.Variable(tf.random_normal([10], 0, 1))
#    f = tf.matmul(h_fc, W_fc2) + b_fc2
#    
#    prob = tf.nn.softmax(f)
#    y_pred = tf.cast(tf.argmax(prob, axis=1), tf.int32)
#    
#    accuracy = tf.reduce_mean(tf.cast(tf.equal(input_y, y_pred), tf.float32))
#    loss = tf.nn.softmax_cross_entropy_with_logits(logits=f, labels=input_y_onehot)
#    
#    tf.global_variables_initializer().run()
#    seq = np.arange(n_samp)
#    np.random.shuffle(seq)
#    
#    batch_size = 32
#    batch_num = int(np.ceil(n_samp/batch_size))
#    for epoch in range(batch_num):
#        scope = seq[range(epoch*batch_size, min((epoch+1)*batch_size, n_samp))]
#        X_batch = X_train[scope, :]
#        y_batch_onehot = y_train_onehot[scope, :]
#        
#        if epoch%100==0:  # 每进行100次训练，对准确率进行一次评测。评测时keep_prob设为1
#            train_accuracy=accuracy.eval(feed_dict={input_X:X_batch,input_y:y_batch_onehot,keep_prob:1.0})
#            print("step %d,training accurancy %g"%(epoch,train_accuracy))
#
#        train_step.run(feed_dict={input_X:X_batch,input_y:y_batch_onehot,keep_prob:0.5})     


# -*- coding: utf-8 -*-

# 载入MNIST数据集
#from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow as tf
#mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#sess=tf.InteractiveSession()
## 定义权重和偏置的初始化函数
#def weight_varible(shape):
#    # 给权重制造一些随机的噪声来打破完全对称，比如截断的正态分布噪声，标准差设为0.1
#    initial=tf.truncated_normal(shape,stddev=0.1)#stddev表示标准差
#    return tf.Variable(initial)
#def bias_variable(shape):
#    # 给偏置增加一些小的正值(0.1)，来避免死亡节点(dead neurons)
#    initial=tf.constant(0.1,shape=shape)
#    return tf.Variable(initial)
## 定义卷积层和池化层的初始化函数
#def conv2d(x,W):
#    # tf.nn.conv2d是二维卷积函数。参数中x是输入；W是卷积的参数，比如[5,5,1,32]：前两个数字代表卷积核的尺寸，第三个代表有多少个channel，对于灰度单色，channe是1，若是彩色图片，channel是3；最后一个数字代表卷积核的数量。
#    # strides代表卷积模板移动的步长，都是1代表不遗漏地划过图片的每一个点。padding代表边界的处理方式，SAME代表给边界加上padding，让卷积的输出和输入保持同样尺寸。
#    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#def max_pool_2x2(x):
#    # tf.nn.max_pool是最大池化函数，这里使用2*2最大池化，即将一个2*2的像素块降为1*1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征。
#    # 因为希望整体上缩小图片尺寸，所以池化层的strides设为横竖方向以2为步长。如果步长还是1，那么会得到一个尺寸不变的图片。
#    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
## 定义输入的placeholder
#x=tf.placeholder(tf.float32,[None,784])#特征
#y_=tf.placeholder(tf.float32,[None,10])#真是的label
## 因为CNN会利用空间结构信息，因此需要将1D的输入向量转为2D的图片结构，即从1*784转为原始的28*28结构。
## 因为只有一个颜色通道，所以最终尺寸为[-1,28,28,1]，-1代表样本数量不固定，最后的1代表颜色通道数量
#x_image=tf.reshape(x,[-1,28,28,1])#tf.reshape是变形函数
## 定义第一个卷积层，使用前面写好的函数进行参数初始化
#W_conv1=weight_varible([5,5,1,32])#代表卷积核尺寸为5*5,1个颜色通道，32个不同的卷积核
#b_conv1=bias_variable([32])
#h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)  # 使用conv2d函数进行卷积操作，并加上偏置，再使用ReLU激活函数进行非线性处理
#h_pool1=max_pool_2x2(h_conv1)#使用最大池化函数max_pool_2x2函数对卷积的输出结果进行池化操作
## 定义第二个卷积层,与第一个不同的是，卷积核的数量变成了64，也就是说这一层卷积会提取64种特征
#W_conv2=weight_varible([5,5,32,64])  # 书上是5,5,32,64？？
#b_conv2=bias_variable([64])
#h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#h_pool2=max_pool_2x2(h_conv2)
## 经过两次步长为2*2的最大池化，现在边长已经只有1/4了。图片尺寸由28*28变为7*7.而第二个卷积层的卷积核数量为64，其输出的tensor尺寸即为7*7*64.
## 定义一个全连接层
#W_fc1=weight_varible([7*7*64,1024])  # 全连接层隐含节点为1024个
#b_fc1=bias_variable([1024])
#h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])  # 对第二个卷积层的输出tensor进行变形，将其转化为1D的向量
#h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
## 为了减轻过拟合，下面使用一个dropout层
#keep_prob=tf.placeholder(tf.float32)
#h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
## 将dropout层的输出连接一个softmax层，得到最后的概率输出
#W_fc2=weight_varible([1024,10])
#b_fc2=bias_variable([10])
#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
## 定义损失函数为cross entronpy，优化器使用Adam
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
#train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 学习率为1e-4
## 定义评测准确率的操作
#corret_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
#accuracy=tf.reduce_mean(tf.cast(corret_prediction,tf.float32))
## 开始训练过程
#tf.global_variables_initializer().run()
#for i in range(20000):  # 进行20000次迭代训练
#    # 初始化所有参数，设置dropout的keep_drop为0.5.使用大小为50的mini-batch。参与训练的样本总量为50*20000=100万
#    batch=mnist.train.next_batch(50)
#    if i%100==0:  # 每进行100次训练，对准确率进行一次评测。评测时keep_prob设为1
#        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
#        print("step %d,training accurancy %g"%(i,train_accuracy))
#    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
## 全部训练完成后，在最终的测试集上进行全面的测试，得到整体的分类准确率
#print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:09:38 2018

@author: zhaoyu
"""

from ReadMNIST import *
import tensorflow as tf

def one_hot_encoding(y, n_class):
    one_hot = np.zeros([len(y), n_class], dtype='float32')
    for i in range(len(y)):
        one_hot[i][y[i]] = 1.0
    return one_hot

if '__main__' == __name__:
    print('tensorflow implemented CNN')

    train_file = './data/train.csv'
    X_train, y_train = read_mnist(train_file, True)

    # data preparation
    n_samp, n_feat = X_train.shape
    n_class = len(set(y_train))
    y_train_onehot = one_hot_encoding(y_train, n_class)

    # define basic containers (placeholders) computing graphs
    input_X = tf.placeholder(tf.float32, [None, n_feat])
    input_y = tf.placeholder(tf.int32, [None])
    input_y_onehot = tf.placeholder(tf.float32, [None, n_class])

    # preprare for image process
    input_X_image = tf.reshape(input_X, [-1, 28, 28, 1])

    # define layers and parameters
    # The filters in conv layer is defined by myself,
    # Diffing from the MLP, in wich the weight matrix has its own shape fixed by
    # the numbers of neurons of L-th and (L+1)-th layer.
    W1 = tf.Variable(tf.random_normal([5, 5, 1, 32], 0, 1, dtype=tf.float32))
    b1 = tf.Variable(tf.random_normal([32, ], 0, 1, dtype=tf.float32))
    h1_conv = tf.nn.sigmoid(tf.nn.conv2d(input_X_image, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    h1_pool = tf.nn.max_pool(h1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully-connect layer 1
    h1_pool_flat = tf.reshape(h1_pool, [-1, 14 * 14 * 32])
    W_fc = tf.Variable(tf.random_normal([14 * 14 * 32, 1024], 0, 1))
    b_fc = tf.Variable(tf.random_normal([1024], 0, 1))
    h_fc = tf.nn.sigmoid(tf.matmul(h1_pool_flat, W_fc) + b_fc)

    # fully-connect layer 2
    W_fc2 = tf.Variable(tf.random_normal([1024, 10], 0, 1))
    b_fc2 = tf.Variable(tf.random_normal([10], 0, 1))
    f = tf.matmul(h_fc, W_fc2) + b_fc2

    # define output layer, prediction, and loss function
    prob = tf.nn.softmax(f)
    y_pred = tf.cast(tf.argmax(prob, axis=1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(input_y, y_pred), tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f, labels=input_y_onehot))

    # define optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    s = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    loops_cnt = 10
    batch_size = 32
    batch_num = int(np.ceil(n_samp/batch_size))
    for i in range(loops_cnt):
        seq = np.arange(n_samp)
        np.random.shuffle(seq)
        for epoch in range(batch_num):
            scope = seq[range(epoch * batch_size, min((epoch + 1) * batch_size, n_samp))]
            X_batch = X_train[scope, :]
            y_batch = y_train[scope]
            y_batch_onehot = y_train_onehot[scope, :]
            s.run(train_step, {input_X: X_batch, input_y_onehot: y_batch_onehot})
        loss_i, acc_i = s.run([loss, accuracy], {input_X: X_batch, input_y: y_batch, input_y_onehot: y_batch_onehot})
#        print(float(0) in prob) # check NaN in logits
        print(i, ' : ', loss_i, acc_i)
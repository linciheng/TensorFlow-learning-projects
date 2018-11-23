# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 21:42:14 2018

@author: mail
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

minist = input_data.read_data_sets('MINIST_data', one_hot = True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    
    return result

#定义权重变量
def weight_variable(shape):
    initial = tf.truncted_nomal(shape, stddev = 0.1)
    return tf.Variable(initial)

#定义偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
#卷积函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W,strides=[1,1,1,1], padding = 'SAME')
#池化层
def pooling(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1])
#输入
xs = tf.placeholder(tf.float32,[None, 784]) #第一维是batches，第二维是特恒维度
ys = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1,28,28,1])

#conv layer1
W_conv1 = weight_variable([5,5,1,32]) #卷积核的大小5*5，通道1，输出32
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pooling1 = pooling(h_conv1)

#conv layer2
W_conv2 = weight_variable([5,5,32,64]) #卷积核的大小5*5，通道1，输出32
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pooling1,W_conv2) + b_conv2)
h_pooling2 = pooling(h_conv2)

#建立全连接层
h_pool2_flat = tf.reshape(h_pooling2,[-1,7*7*64])

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pooling2, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = weight_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

#损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
#优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#根据TensorFlow的版本选择变量初始化函数
if int((tf.__version__).split('.')[1]) <12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = minist.train.next_batch(100)
    sess.run(train_step, feed_dict = {xs: batch_xs,ys: batch_ys, keep_prob: 0.5})
    if i % 50==0:
        print(compute_accuracy(minist.test.image[:1000], minist.test.labels[:1000]))
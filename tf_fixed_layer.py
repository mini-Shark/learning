from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import time

BATCH_SIZE = 128
ITERATION = 1000
TEST_ITER = 100
LEARNING_RATE = 0.001

mnist = input_data.read_data_sets(r"D:\test_data\mnist", one_hot=True)
x_in = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
training_flag = tf.placeholder(dtype=tf.bool)
keep_prob = tf.placeholder(tf.float32)

x = tf.reshape(tensor=x_in, shape=[-1, 28, 28, 1])

w_conv1 = tf.get_variable(name='w_conv1',
                    shape=[5, 5, 1, 32],
                    initializer=tf.random_normal_initializer(),
                    dtype=tf.float32,
                    trainable=False)
b_conv1 = tf.get_variable(name = 'b_conv1',
                    shape = 32,
                    initializer=tf.random_normal_initializer(),
                    trainable=False)

conv1 = tf.nn.conv2d(input=x, filter=w_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1
conv1 = tf.nn.relu(conv1)


w_conv2 = tf.get_variable(name='w_conv2',
                    shape=[3, 3, 32, 32],
                    initializer=tf.random_normal_initializer(),
                    dtype=tf.float32)
b_conv2 = tf.get_variable(name= 'b_conv2',
                    shape = 32,
                    initializer=tf.random_normal_initializer())
conv2 = tf.nn.conv2d(input=conv1, filter=w_conv2, strides=[1,2,2,1], padding="SAME") + b_conv2
conv2 = tf.nn.relu(conv2)


w_conv3 = tf.get_variable(name='w_conv3',
                    shape=[5, 5, 32, 64],
                    initializer=tf.random_normal_initializer(),
                    dtype=tf.float32)
b_conv3 = tf.get_variable(name = 'b_conv3',
                    shape = 64,
                    initializer=tf.random_normal_initializer())
conv3 = tf.nn.conv2d(input=conv2, filter=w_conv3, strides=[1,1,1,1], padding="SAME") + b_conv3
conv3 = tf.nn.relu(conv3)


w_conv5 = tf.get_variable(name='w_conv5',
                    shape=[5, 5, 32, 64],
                    initializer=tf.random_normal_initializer(),
                    dtype=tf.float32)
b_conv5 = tf.get_variable(name = 'b_conv5',
                    shape = 64,
                    initializer=tf.random_normal_initializer())
conv5 = tf.nn.conv2d(input=conv2, filter=w_conv5, strides=[1,1,1,1], padding="SAME") + b_conv5
conv5 = tf.nn.relu(conv5)

w_cnov4 = tf.get_variable(name='w_cnov4',
                    shape=[3, 3, 64, 64],
                    initializer=tf.random_normal_initializer(),
                    dtype=tf.float32)
b_cnov4 = tf.get_variable(name = 'b_cnov4',
                    shape = 64,
                    initializer=tf.random_normal_initializer())
cnov4 = tf.nn.conv2d(input=conv5, filter=w_cnov4, strides=[1,2,2,1], padding="SAME") + b_cnov4
cnov4 = tf.nn.relu(cnov4)

fea_flatten = tf.reshape(cnov4,shape=[-1,7*7*64])

w_fc1 = tf.get_variable(name='w_fc1',
                  shape=[7*7*64, 1024],
                  initializer=tf.random_normal_initializer())
b_fc1 = tf.get_variable(name = 'b_fc1',
                    shape = 1024,
                    initializer=tf.random_normal_initializer())

fc1 = tf.matmul(fea_flatten,w_fc1) + b_fc1
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep_prob)

w_fc2 = tf.get_variable(name='w_fc2',
                  shape=[1024, 10],
                  initializer=tf.random_normal_initializer())
b_fc2 = tf.get_variable(name = 'b_fc2',
                    shape = 10,
                    initializer=tf.random_normal_initializer())
logits = tf.matmul(fc1,w_fc2) + b_fc2

prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

corrected = tf.equal(tf.arg_max(input=prediction, dimension=1), tf.arg_max(input=y_true, dimension=1))
accuracy = tf.reduce_mean(tf.cast(x=corrected, dtype=tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
  sess.run(init)
  variable_name = [v.name for v in tf.trainable_variables()]
  print(variable_name)
  for i in range(1, ITERATION+1):
    x_ ,y_ = mnist.train.next_batch(batch_size=BATCH_SIZE, shuffle=True)
    _loss_ ,_ = sess.run(fetches=[loss, train_op], feed_dict={x_in: x_, 
                                                              y_true: y_,
                                                              training_flag: True,
                                                              keep_prob:0.8})
    # print("step:{:0>5d}".format(i),"current loss: {:<.10f}".format(_loss_))

    if i%TEST_ITER==0:
      x_, y_ = mnist.test.next_batch(batch_size=BATCH_SIZE, shuffle=True)
      _loss_, _accuracy_ = sess.run(fetches=[loss, accuracy], feed_dict={x_in: x_,
                                                                         y_true: y_,
                                                                         training_flag: False,
                                                                         keep_prob: 1.0})
      print("test step {:0>5d}".format(i), "test loss: {:<.4f}".format(_loss_),
            "Accuracy: {:.2%}".format(_accuracy_))                                                                         



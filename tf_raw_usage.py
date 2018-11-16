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

def conv_layer(fea_map,ksize,name,stride,training):
  '''
  fea_map: feature maps
  ksize: convolution kernel size
  name: name suffix of variable name
  stride: convolution stride
  training: FLAG of training process
  '''
  assert stride[0] == stride[-1] == 1, "stride should be like [1, ..., 1]"
  # assert tf.shape(fea_map)[-1] == ksize[-2], "input feature shape should fit the size of kernel"

  w = tf.get_variable(name='w_'+name,
                      shape=ksize,
                      initializer=tf.random_normal_initializer(),
                      dtype=tf.float32)

  b = tf.get_variable(name='b_'+name,
                      shape=ksize[-1],
                      initializer=tf.random_normal_initializer(),
                      dtype=tf.float32)

  output = tf.nn.conv2d(input=fea_map, filter=w, strides=stride, padding="SAME") + b # same means have same spatial size as input 
  # output = tf.layers.batch_normalization(inputs=output, momentum=0.998, epsilon=0.001, # this is a functional interface of tf.Layers.BatchNormalization
  #                                        center=True, scale=True, trainable=True, training=training)# batch norm should been apply before activation
  return tf.nn.relu(output)

def fc_layers(fea_map, input_size,output_size,name):
  '''
  fea_map: all input feature
  input_size: input neurons
  output_size: output neurons
  name: name suffix of variable
  '''
  w = tf.get_variable(name='w_'+name,
                      shape=[input_size, output_size],
                      initializer=tf.random_normal_initializer())

  b = tf.get_variable(name = 'b_'+name,
                      shape = output_size,
                      initializer=tf.random_normal_initializer())
  
  output = tf.matmul(fea_map,w) + b
  return output

def network(x,keep_prob,training_flag):

  x = tf.reshape(tensor=x, shape=[-1, 28, 28, 1])

  with tf.variable_scope('conv_layer_1'): # variable scope name couldn't have white space
    conv1 = conv_layer(fea_map=x, ksize=[5, 5, 1, 32], name='conv1', stride=[1,1,1,1], training=training_flag) # N x 28 x 28 x 16
  
  pool1 = tf.nn.max_pool(value=conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # N x 14 x 14 x 32

  with tf.variable_scope('conv_layer_2'):
    conv2 = conv_layer(fea_map=pool1, ksize=[5, 5, 32, 64], name='conv2', stride=[1,1,1,1], training=training_flag) # N x 28 x 28 x 32
  
  pool2 = tf.nn.max_pool(value=conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # N x 14 x 14 x 32


  # with tf.variable_scope('conv_layer_3'):
  #   conv3 = conv_layer(fea_map=pool1, ksize=[3,3,32,16], name='conv2', stride=[1,1,1,1], training=training_flag) # N x 14 x 14 x 16

  # pool2 = tf.nn.avg_pool(value=conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # N x 7 x 7 x 16

  # with tf.variable_scope('conv_layer_4'):
  #   conv4 = conv_layer(fea_map=pool2, ksize=[1,1,16,1], name='conv4', stride=[1,1,1,1], training=training_flag) # N x 7 x 7 x 1

  fea_flatten = tf.reshape(pool2,shape=[-1,7*7*64])

  # with tf.variable_scope('fc_layer_1'):
  #   fc1 = tf.nn.relu(fc_layers(fea_map=fea_flatten, input_size=7*7*64, output_size = 1024, name="fc_1"))
  #   fc1 = tf.nn.dropout(fc1, keep_prob)
  #   # debug_op = tf.Print(fc1,[fc1.shape,"fc1"], message="Debug Message", summarize=100)


  # with tf.variable_scope('fc_layer_2'):
  #   logits = fc_layers(fea_map=fea_flatten, input_size=1024, output_size = 10, name="fc_2")

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

  return logits

mnist = input_data.read_data_sets(r"D:\test_data\mnist", one_hot=True)
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
training_flag = tf.placeholder(dtype=tf.bool)
keep_prob = tf.placeholder(tf.float32)

logits = network(x, keep_prob, training_flag)
prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

corrected = tf.equal(tf.arg_max(input=prediction, dimension=1), tf.arg_max(input=y_true, dimension=1))
accuracy = tf.reduce_mean(tf.cast(x=corrected, dtype=tf.float32))

init = tf.global_variables_initializer()

start = time.time()

with tf.Session() as sess:
  sess.run(init)
  variable_name = [v.name for v in tf.trainable_variables()]
  print(variable_name)
  for i in range(1, ITERATION+1):
    x_ ,y_ = mnist.train.next_batch(batch_size=BATCH_SIZE, shuffle=True)
    _loss_ ,_ = sess.run(fetches=[loss, train_op], feed_dict={x: x_, 
                                                              y_true: y_,
                                                              training_flag: True,
                                                              keep_prob:0.8})
    # print("step:{:0>5d}".format(i),"current loss: {:<.10f}".format(_loss_))

    if i%TEST_ITER==0:
      x_, y_ = mnist.test.next_batch(batch_size=BATCH_SIZE, shuffle=True)
      _loss_, _accuracy_ = sess.run(fetches=[loss, accuracy], feed_dict={x: x_,
                                                                         y_true: y_,
                                                                         training_flag: False,
                                                                         keep_prob: 1.0})
      print("test step {:0>5d}".format(i), "test loss: {:<.4f}".format(_loss_),
            "Accuracy: {:.2%}".format(_accuracy_))                                                                         

print("time is{:}".format(time.time()-start))
     

  






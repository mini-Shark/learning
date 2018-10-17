from __future__ import absolute_import, division, print_function 
import numpy as np
import tensorflow as tf
'''
在eager模式下dataset是一个迭代器，可以用迭代器的方法访问数据
在graph模式下不能直接访问,需要创建迭代器后访问
'''
x = np.random.randn(100,3,3)
y = np.zeros((100,),dtype=np.int32)
dataset = tf.data.Dataset.from_tensor_slices((x,y))
dataset = dataset.shuffle(1000).repeat().batch(32) #注意这里的shuffle里的参数应该大于数据的条目个数
iterator = dataset.make_initializable_iterator()#定义可重新初始化迭代器
next_element = iterator.get_next()#定义迭代器访问元素的操作
with tf.Session() as sess:
    sess.run(iterator.initializer)#初始化迭代器
    for _ in range(100):
        tx,ty = sess.run(next_element)#获取元素
        print(tx,ty)


'''
上面的代码段会将 x和 y 数组作为 tf.constant() 指令嵌入在 TensorFlow 图中。
这样非常适合小型数据集，但会浪费内存，因为会多次复制数组的内容，
并可能会达到 tf.GraphDef 协议缓冲区的 2GB 上限,所以建议使用以下形式。
利用place_holder来创建dataset。
'''
# Load the training data into two NumPy arrays, for example using `np.load()`.
features = np.random.randn(100,3,3)
labels = np.zeros((100,),dtype=np.int32)

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
'''
这里又是用get_next()来获取元素，需要run()一次。
但是在实际中又需要把这些元素重新使用feed dict喂给计算图
能不能跳过这个过程，直接喂
'''
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                            labels_placeholder: labels})
    for _ in range(100):
        tx,ty = sess.run(next_element)#获取元素
        print(tx,ty)
import os
import time
from datetime import datetime
import preprocessing
import tensorflow as tf

# input size: 1323000

def cnn(input, training=True):
    feat_len = input.get_shape().as_list()[1]
    conv1 = tf.layers.conv1d(inputs=input, filters=10, kernel_size=100, strides=2,
                             padding='same', activation=tf.nn.relu, name='conv_%d' % 1)
    pool1 = tf.layers.max_pooling1d(inputs=conv1,pool_size=2,strides=2,
                                    padding='same', name ='pool_%d' % 1)

    conv2 = tf.layers.conv1d(inputs=pool1, filters=10, kernel_size=100, strides=1,
                             padding='same', activation=tf.nn.relu, name='conv_%d' % 2)
    pool2= tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2,
                                    padding='same', name='pool_%d' % 2)

    conv3 = tf.layers.conv1d(inputs=pool2, filters=10, kernel_size=100, strides=1,
                             padding='same', activation=tf.nn.relu, name='conv_%d' % 3)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2,
                                    padding='same', name='pool_%d' % 3)

    conv4 = tf.layers.conv1d(inputs=pool3, filters=10, kernel_size=100, strides=1,
                             padding='same', activation=tf.nn.relu, name='conv_%d' % 4)
    pool4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2,
                                    padding='same', name='pool_%d' % 4)
    import math
    final_size = int(math.ceil(feat_len / 2 ** 5))
    final_size = final_size * 10
    flat = tf.reshape(pool4, [-1,final_size])
    flat = tf.nn.dropout(flat,keep_prob=0.6)
    dense = tf.layers.dense(flat,final_size/4)
    dense = tf.nn.dropout(dense,keep_prob=0.6)
    output = tf.layers.dense(dense, 41)
    return output

def main():
    x = tf.placeholder(tf.float32,shape=[None,None],name='x')
    y = tf.placeholder(tf.int32,shape=[None,1],name = 'y')
    mode = tf.placeholder(tf.bool,name='mode')

    logits = cnn(x, training=mode)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits)


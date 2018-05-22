import os
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
import random
random.seed(1)
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

# input size: 1323000

n_epochs = 10
train_size = 2800
batch_size = 8
step_per_epoch = train_size/batch_size
log_dir = 'log/log'

def cnn(input, training=True):
    input = tf.expand_dims(input, axis=-1)
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

    flat = tf.layers.flatten(pool4)
    flat = tf.nn.dropout(flat,keep_prob=0.6)
    dense = tf.layers.dense(flat,128)
    dense = tf.nn.dropout(dense,keep_prob=0.6)
    output = tf.layers.dense(dense, 41)
    return output


def data_loader():
    train_data = np.load('train.npy')
    label_data = np.load('label.npy')
    valid_data = train_data[-(3000-train_size):]
    label_val = label_data[-(3000-train_size):]
    train_data = train_data[:train_size]
    label_data = label_data[:train_size]
    return train_data, label_data, valid_data, label_val


def batch_generator(train_data, label_data, step):
    max_size =int(np.ceil(train_size*1.0/batch_size))
    if step % max_size == 0:
        x_batch = train_data[(max_size - 1) * batch_size:]
        y_batch = label_data[(max_size - 1) * batch_size:]

    else:
        x_batch = train_data[((step % max_size) - 1) * batch_size:step % max_size * batch_size]
        y_batch = label_data[((step % max_size) - 1) * batch_size:step % max_size * batch_size]
    return x_batch, y_batch

def main():
    train_data, train_label, val_data, val_label = data_loader()

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32,shape=[None,600000],name='x')
    y = tf.placeholder(tf.int32,shape=[None],name = 'y')
    mode = tf.placeholder(tf.bool,name='mode')

    logits = cnn(x, training=mode)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits)
    tf.summary.scalar('loss',loss)

    global_step = tf.train.get_or_create_global_step()
    start_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(start_learning_rate,global_step,
                                               step_per_epoch,0.96,staircase=True)
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optim.minimize(loss, global_step)

    y_pred = tf.argmax(logits, axis=1)
    y_pred = tf.cast(y_pred, tf.int32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
    tf.summary.scalar('acc', accuracy)

    summary_op = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_summary_writer = tf.summary.FileWriter(log_dir)
        valid_summary_writer = tf.summary.FileWriter(log_dir + '_val')
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(1,n_epochs*int(np.ceil(train_size*1.0/batch_size))):
            start_time = time.time()
            x_batch, y_batch = batch_generator(train_data,train_label,step)

            _, loss_val, step_summary, global_step_val = sess.run(
                [train_op, loss, summary_op, global_step],
                feed_dict={x:x_batch,
                           y:y_batch,
                           mode: True}
            )
            duration = time.time() - start_time

            format_str = ('%s: step %d, loss = %.2f (%.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), global_step_val, loss_val, duration))

            train_summary_writer.add_summary(step_summary, global_step_val)

            if step == int(np.ceil(train_size*1.0/batch_size)):
                x_batch, y_batch = val_data,val_label
                step_summary,global_step_val = sess.run(
                    [summary_op, global_step], feed_dict={x: x_batch, y: y_batch, mode: False}
                )

                valid_summary_writer.add_summary(step_summary, global_step_val)


if __name__ == '__main__':
    main()
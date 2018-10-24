import scipy.io as scio
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import graph_util  
from tensorflow.python.platform import gfile  


# 定义网络超参数
learning_rate = 0.005  #学习率
training_iters = 200 #训练集样本数量
test_batch_size = 200 #测试集每批的batch大小（测试时，每批都输出一次精度）
testing_iters = 200 #测试集样本数量
batch_size = 10  #训练集每批的batch大小
display_step = 1  #训练时 每多少批数据输出一次精度accuracy
# 定义网络参数
n_input = 256 # 输入的维度
n_input_1 = 128 # 通道1的输入维数，对应大方差
n_input_2 = 128 # 通道2的输入维数，对应小方差
# n_input = n_input_1 + n_input_2
n_classes = 3 # 标签的维度

dropout = 0.8 # Dropout 的概率
epoch_num = 2 #生成batch时的迭代次数


# 导入HRRP数据
file_name = './Train_hrrp.mat'   
# file_name = 'D:/HRRP data/data/2channel/pca256.mat'   
traindata_base =scio.loadmat(file_name)['aa']
# print(data_base.shape)
hrrp = traindata_base[:,3:]
# print(hrrp)
labels = traindata_base[:,0:3]
# print(label)
file_name2 = './Test_hrrp.mat'
testdata_base = scio.loadmat(file_name2)['bb']
test_hrrp = testdata_base[:,3:]
test_labels =testdata_base[:,0:3]


#生成batch数据
def get_batch_data(batch_size=batch_size):
    # 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.train.slice_input_producer([hrrp, labels], shuffle=False)
    hrrp_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    train_in_x1 = hrrp_batch[:,0:n_input_1]
    train_in_x2 = hrrp_batch[:, n_input-n_input_2 : n_input]
    train_out_y = label_batch
    return train_in_x1, train_in_x2, train_out_y

[train_in_x1, train_in_x2, train_out_y] = get_batch_data(batch_size=batch_size)

def get_test_data(batch_size=batch_size):
    # 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.train.slice_input_producer([test_hrrp, test_labels], shuffle=False)
    hrrp_test, label_test = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    test_in_x1 = hrrp_test[:,0:n_input_1]
    test_in_x2 = hrrp_test[:,n_input-n_input_2 : n_input]
    test_out_y = label_test

    return test_in_x1, test_in_x2, test_out_y

[test_in_x1, test_in_x2, test_out_y] = get_batch_data(batch_size=test_batch_size)


#将输入的数据划分成X1、X2两部分，原理是pca之后前128位数据方差大，为X1,   后128位为X2.


# 占位符输入
with tf.name_scope('inputs'):
    x1 = tf.placeholder(tf.float32, [None, n_input_1],name='x1_in')
    x2 = tf.placeholder(tf.float32, [None, n_input_2],name='x2_in')
    y = tf.placeholder(tf.float32, [None, n_classes],name='y_in')

    keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')

# 卷积操作
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,1,1,1], padding='SAME'),b), name=name)
# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1,k,k,1], padding='SAME', name=name)
# 平均下采样操作
def avg_pool(name, l_input,k):
    return tf.nn.avg_pool(l_input, ksize=[1, k, k, 1], strides=[1,k,k,1], padding='SAME', name=name)
# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# 定义整个网络
def alex_net( _X1 , _X2 , _weights, _biases, _dropout):
    # 向量转为矩阵
    _X1 = tf.reshape(_X1, shape=[-1, 1, n_input_1, 1])
    _X2 = tf.reshape(_X2, shape=[-1, 1, n_input_2, 1])

    with tf.name_scope('channel1'): # 通道1，对应大方差
        with tf.name_scope('layer1_1'):
        # 卷积层
            conv1_1 = conv2d('conv1_1', _X1, _weights['wc1_1'], _biases['bc1_1'])
            # 下采样层
            pool1_1 = avg_pool('pool1_1', conv1_1, k=2)
            # 归一化层
            norm1_1 = norm('norm1_1', pool1_1, lsize=4)
            # Dropout
            norm1_1 = tf.nn.dropout(norm1_1, _dropout)

        # 卷积
        with tf.name_scope('layer1_2'):
            conv1_2 = conv2d('conv1_2', norm1_1, _weights['wc1_2'], _biases['bc1_2'])
            # 下采样
            pool1_2 = avg_pool('pool1_2', conv1_2, k=2)
            # 归一化
            norm1_2 = norm('norm1_2', pool1_2, lsize=4)
            # Dropout
            norm1_2 = tf.nn.dropout(norm1_2, _dropout)

        with tf.name_scope('layer1_3'):
            conv1_3 = conv2d('conv1_3', norm1_2, _weights['wc1_3'], _biases['bc1_3'])
            # 下采样
            pool1_3 = avg_pool('pool1_3', conv1_3, k=2)
            # 归一化
            norm1_3 = norm('norm1_3', pool1_3, lsize=4)
            # Dropout
            norm1_3 = tf.nn.dropout(norm1_3, _dropout)

        # 全连接层，先把特征图转为向量
        with tf.name_scope('fc_1'):
            dense1 = tf.reshape(norm1_3, [-1, _weights['wd_1'].get_shape().as_list()[0]])
            dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd_1']) + _biases['bd_1'], name='fc_1')
            # 全连接层

    with tf.name_scope('channel2'): # 通道1，对应大方差
        with tf.name_scope('layer2_1'):
        # 卷积层
            conv2_1 = conv2d('conv2_1', _X2, _weights['wc2_1'], _biases['bc2_1'])
            # 下采样层
            pool2_1 = max_pool('pool2_1', conv2_1, k=2)
            # 归一化层
            norm2_1 = norm('norm2_1', pool2_1, lsize=4)
            # Dropout
            norm2_1 = tf.nn.dropout(norm2_1, _dropout)

        # 卷积
        with tf.name_scope('layer2_2'):
            conv2_2 = conv2d('conv2_2', norm2_1, _weights['wc2_2'], _biases['bc2_2'])
            # 下采样
            pool2_2 = max_pool('pool2_2', conv2_2, k=2)
            # 归一化
            norm2_2 = norm('norm2_2', pool2_2, lsize=4)
            # Dropout
            norm2_2 = tf.nn.dropout(norm2_2, _dropout)

        with tf.name_scope('layer2_3'):
            conv2_3 = conv2d('conv2_3', norm2_2, _weights['wc2_3'], _biases['bc2_3'])
            # 下采样
            pool2_3 = max_pool('pool2_3', conv2_3, k=2)
            # 归一化
            norm2_3 = norm('norm2_3', pool2_3, lsize=4)
            # Dropout
            norm2_3 = tf.nn.dropout(norm2_3, _dropout)

        # 全连接层，先把特征图转为向量
        with tf.name_scope('fc_2'):
            dense2 = tf.reshape(norm2_2, [-1, _weights['wd_2'].get_shape().as_list()[0]])
            dense2 = tf.nn.relu(tf.matmul(dense2, _weights['wd_2']) + _biases['bd_2'], name='fc_2')
            # 全连接层


    # 网络输出层
    with tf.name_scope('outs'):
        dense = tf.concat([dense1, dense2], 1)
        dense_2 = tf.nn.relu(tf.matmul(dense, _weights['wd']) + _biases['bd'] )
        out = tf.add(tf.matmul(dense_2, _weights['out']), _biases['out'],name='output')
    return out

# 存储所有的网络参数
with tf.name_scope('Weights'):
    weights = {
        'wc1_1': tf.Variable(tf.random_normal([1, 24, 1, 64])),
        'wc1_2': tf.Variable(tf.random_normal([1, 24, 64, 128])),
        'wc1_3': tf.Variable(tf.random_normal([1, 24, 128, 256])),
        'wd_1': tf.Variable(tf.random_normal([1*16*256, 1024])),

        'wc2_1': tf.Variable(tf.random_normal([1, 48, 1, 64])),
        'wc2_2': tf.Variable(tf.random_normal([1, 48, 64, 128])),
        'wc2_3': tf.Variable(tf.random_normal([1, 48, 128, 256])),
        'wd_2': tf.Variable(tf.random_normal([1*16*256, 1024])),

        'wd': tf.Variable(tf.random_normal([2048, 2048])),
        'out': tf.Variable(tf.random_normal([2048, n_classes]))
    }
with tf.name_scope('biases'):
    biases = {
        'bc1_1': tf.Variable(tf.random_normal([64])),
        'bc1_2': tf.Variable(tf.random_normal([128])),
        'bc1_3': tf.Variable(tf.random_normal([256])),
        'bd_1': tf.Variable(tf.random_normal([1024])),

        'bc2_1': tf.Variable(tf.random_normal([64])),
        'bc2_2': tf.Variable(tf.random_normal([128])),
        'bc2_3': tf.Variable(tf.random_normal([256])),
        'bd_2': tf.Variable(tf.random_normal([1024])),

        'bd': tf.Variable(tf.random_normal([2048])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

# 构建模型
pred = alex_net(x1,x2, weights, biases, keep_prob)

# 定义损失函数和学习步骤
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.global_variables_initializer()

# train_writer = tf.summary.FileWriter(log_dir+'/train',sess.graph)
# test_writer = tf.summary.FileWriter(log_dir+'/test')

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    print('doing1')
    # Keep training until reach max iterations
    try:
        while step * batch_size <= training_iters:
            # print('doing2')
            # train_hrrp, train_label = sess.run([hrrp_batch, label_batch])
            # train_x1 = sess.run(train_in_x1);
            train_x2 = sess.run(train_in_x2);
            # train_y = sess.run(train_out_y);
            [train_x1, train_x2, train_y] = sess.run([train_in_x1, train_in_x2, train_out_y])
            # print(batch_xs.shape)
            # print(batch_ys.shape)
            # 获取批数据
            sess.run(optimizer, feed_dict={x1: train_x1, x2: train_x2, y: train_y, keep_prob: dropout})
            if step % display_step == 0:
                # 计算精度
                acc = sess.run(accuracy, feed_dict={x1: train_x1, x2: train_x2, y: train_y, keep_prob: 1.})
                # 计算损失值
                loss = sess.run(cost, feed_dict={x1: train_x1, x2: train_x2, y: train_y, keep_prob: 1.})
                
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
            step += 1
        print ("Optimization Finished!")
        step = 1;
        # 计算测试精度
        while step * test_batch_size <= testing_iters:
            # hrrp_test, label_test = sess.run([hrrp_test, label_test])
            # test_x1 = sess.run(test_in_x1);
            # test_x2 = sess.run(test_in_x2);
            # test_y = sess.run(test_out_y);
            [test_x1, test_x2, test_y] = sess.run([test_in_x1, test_in_x2, test_out_y])

            bcc = sess.run(accuracy, feed_dict={x1: test_x1, x2: test_x2, y: test_y, keep_prob: 1.})
            print ("Iter " + str(step*test_batch_size) + ", Testing Accuracy = " + "{:.5f}".format(bcc), )
            step += 1;
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

    # pb_file_path = 'D:/HRRP data/data/2channel'
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['outs.output'])
    # with tf.gfile.FastGFile(pb_file_path+'model.pb', mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())



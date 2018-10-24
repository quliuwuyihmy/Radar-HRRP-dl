
#coding: utf-8
from __future__ import division, print_function, absolute_import
import scipy.io as scio
import numpy as np
import tensorflow as tf
from DAE_def import AdditiveGaussianNoiseAutoencoder


#定义训练参数
input_size = 256
label_size = 3
training_epochs = 50
training_iters = 200
ae_batch_size = 50
train_batch_size = 5
display_step = 2
stack_size = 5  #栈中包含5个ae
hidden_size = [256,128,64,16,3]
learning_rate = 0.0085 #预测网络的学习率
ae_learning_rate = 0.00285  
test_batch_size = 200
testing_iters = 200
noise_scale = 0 #高斯噪声程度(噪声会影响识别率，改参数可测试算法的鲁棒性)
# 导入HRRP数据
file_name = './Train_hrrp.mat'    
traindata_base =scio.loadmat(file_name)['aa']
train_hrrp = traindata_base[:training_iters,3:]
train_labels = traindata_base[:training_iters,0:3]

file_name2 = './Test_hrrp.mat'    
testdata_base =scio.loadmat(file_name2)['bb']
test_hrrp = testdata_base[0:testing_iters,3:]
test_labels = testdata_base[0:testing_iters,0:3]

noise_train_hrrp =train_hrrp + noise_scale * tf.random_normal((input_size,))
noise_test_hrrp =test_hrrp + noise_scale * tf.random_normal((input_size,))

def get_train_batch(train_hrrp,train_labels,batch_size):
    # 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.train.slice_input_producer([train_hrrp, train_labels], shuffle=False)
    hrrp_train, label_train = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    return hrrp_train, label_train
 
[hrrp_batch,labels_batch] = get_train_batch(noise_train_hrrp,train_labels,batch_size = train_batch_size)
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    # print(start_index)
    return data[start_index:(start_index + batch_size)]
 

def get_test_batch(test_hrrp,test_labels,batch_size):
# 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.train.slice_input_producer([test_hrrp, test_labels], shuffle=False)
    hrrp_train, label_train = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    return hrrp_train, label_train
 
[hrrp_test,labels_test] = get_train_batch(noise_test_hrrp,test_labels,batch_size = test_batch_size)

#建立sdae图
sdae = []
for i in range(stack_size):
    if i == 0:
        ae = AdditiveGaussianNoiseAutoencoder(n_input = 256,
                                               n_hidden = hidden_size[i],
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate = ae_learning_rate),
                                               scale = 0.01)
        ae._initialize_weights()
        sdae.append(ae)
    else:
        ae = AdditiveGaussianNoiseAutoencoder(n_input=hidden_size[i-1],
                                              n_hidden=hidden_size[i],
                                              transfer_function=tf.nn.softplus,
                                              optimizer=tf.train.AdamOptimizer(learning_rate=ae_learning_rate),
                                              scale=0.01)
        ae._initialize_weights()
        sdae.append(ae)
        
W = []
b = []
Hidden_feature = [] #保存每个ae的特征
X_train = np.array([0])
for j in range(stack_size):
    #输入
    if j == 0:
        X_train = train_hrrp
        X_test = train_labels
    else:
        X_train_pre = X_train
        # print(X_train_pre.shape)
        X_train = sdae[j-1].transform(X_train_pre)
        print (X_train.shape)
        Hidden_feature.append(X_train)
    
    #贪婪训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0] / ae_batch_size)
        # print(total_batch)
        # Loop over all batches
        for k in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, ae_batch_size)
 
            # Fit training using batch data
            cost = sdae[j].partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / X_train.shape[0] * ae_batch_size
 
        # Display logs per epoch step
        #if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    
    #保存每个ae的参数
    weight = sdae[j].getWeights()

    W.append(weight)
    b.append(sdae[j].getBiases())

#搭建SAE预测网络(手动搭建)
x = tf.placeholder(tf.float32, [None, input_size],name='x_in')
y = tf.placeholder(tf.float32, [None, label_size],name='y_out')

weights = {
        'w_0': tf.Variable(W[0]),
        'w_1': tf.Variable(W[1]),
        # 'w_0': tf.Variable(tf.random_normal([256,128])),
        # 'w_1': tf.Variable(tf.random_normal([128,3]))
        'w_2': tf.Variable(W[2]),
        'w_3': tf.Variable(W[3]),
        'w_4': tf.Variable(W[4])
}

biases = {
        'b_0': tf.Variable(b[0]),
        'b_1': tf.Variable(b[1]),
        # 'b_0': tf.Variable(tf.random_normal([128])),
        # 'b_1': tf.Variable(tf.random_normal([3]))
        'b_2': tf.Variable(b[2]),
        'b_3': tf.Variable(b[3]),
        'b_4': tf.Variable(b[4])
}

hid_1 = tf.nn.softplus(tf.add(tf.matmul(x,weights['w_0']),biases['b_0']))
hid_2 = tf.nn.softplus(tf.add(tf.matmul(hid_1,weights['w_1']),biases['b_1']))
hid_3 = tf.nn.softplus(tf.add(tf.matmul(hid_2,weights['w_2']),biases['b_2']))
hid_4 = tf.nn.softplus(tf.add(tf.matmul(hid_3,weights['w_3']),biases['b_3']))
hid_5 = tf.nn.softplus(tf.add(tf.matmul(hid_4,weights['w_4']),biases['b_4']))
out = hid_5
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(weights['w_0']) - W[0])
    step = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    # Keep training until reach max iterations
    try:
        while step * train_batch_size <= training_iters:
            #运行“获取训练数据”
            [batch_hrrp,batch_labels]=sess.run([hrrp_batch,labels_batch])
            sess.run(optimizer, feed_dict={x:batch_hrrp, y:batch_labels})
            if step % display_step == 0:
                # 计算精度
                acc = sess.run(accuracy, feed_dict={x:batch_hrrp, y:batch_labels})
                # 计算损失值
                los = sess.run(loss, feed_dict={x:batch_hrrp, y:batch_labels})

                print ("Iter " + str(step*train_batch_size) + ", Minibatch Loss= " + "{:.6f}".format(los) + ", Training Accuracy = " + "{:.5f}".format(acc))
            step += 1
        print ("Optimization Finished!")
        # print(sess.run(weights['w_0']) - W[0])
        # print(W[0])

        # 测试集验证部分
        step = 1
        while step * test_batch_size <= testing_iters:
            [test_hrrp,test_labels]=sess.run([hrrp_test,labels_test])
            # outputreal = sess.run(out,feed_dict={x:test_hrrp})
            # print(step)
            # print(outputreal)
            # print(test_labels)
            # 计算精度
            acc2 = sess.run(accuracy, feed_dict={x:test_hrrp, y:test_labels})
            los2 = sess.run(loss, feed_dict={x:test_hrrp, y:test_labels})
            print ("Iter " + str(step*test_batch_size) + ", Minibatch Loss= " + "{:.6f}".format(los2) + ", Training Accuracy = " + "{:.5f}".format(acc2))
            step += 1

    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

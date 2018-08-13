#coding: utf-8
import tensorflow as tf
import numpy as np
import utils 

def xavier_init(n_input,n_output,constant=1):
    """
    Xavier初始化器 让权重被初始化调整合理的分布 mean=0 std=2/（n_input+n_output）
    :param n_input:输入节点数量
    :param n_output:输出节点数量
    """
    low=-constant * np.sqrt(6.0/(n_input+n_output))
    high=constant * np.sqrt(6.0/(n_input+n_output))
    return tf.random_uniform((n_input,n_output), minval=low, maxval=high, 
                             dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):  
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1): 
        self.n_input = n_input 
        self.n_hidden = n_hidden 
        self.transfer = transfer_function 
        self.scale = tf.placeholder(tf.float32) 
        self.training_scale = scale 
        network_weights = self._initialize_weights() 
        self.weights = network_weights 
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        #编码
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                self.weights['w1']),
                self.weights['b1']))
        #解码
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
 
        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
 
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
 
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
    #优化参数
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X,
                                                                            self.scale: self.training_scale
                                                                            })
        return cost
 
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,
                                                     self.scale: self.training_scale
                                                     })
 
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,
                                                       self.scale: self.training_scale
                                                       })
 
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
 
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
                                                               self.scale: self.training_scale
                                                               })
 
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
 
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
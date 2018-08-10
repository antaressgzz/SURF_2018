import numpy as np
from core.memory.memory import Memory
from core.action_dis.action_discretization import action_discretization
from .graph_builder import Graph_builder
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
from core.config import network_config
from time import time

class Dqn_agent:
    def __init__(self, asset_num, division, feature_num, gamma,
                 network_topology=network_config['cnn_fc'],
                 epsilon=1, epsilon_Min=0.1, epsilon_decay_period=100000,
                 learning_rate_decay_step=10000, update_tar_period=1000,
                 history_length=50,
                 memory_size=10000, batch_size=32, log_freq=50,
                 save_period=100000, name='dqn', save=False,
                 GPU=False):

        self.epsilon = epsilon
        self.epsilon_min = epsilon_Min
        self.epsilon_decay_period = epsilon_decay_period
        self.asset_num = asset_num
        self.division = division
        self.gamma = gamma
        self.name = name
        self.update_tar_period = update_tar_period
        self.log_freq = log_freq
        self.history_length = history_length
        self.feature_num = feature_num
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate=0.01, global_step=self.global_step,
                                             decay_steps=learning_rate_decay_step, decay_rate=0.9)
        # self.lr = 0.01
        self.action_num, self.actions = action_discretization(self.asset_num, self.division)
        
        config = tf.ConfigProto()

        if GPU == False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        network_topology['output_num'] = self.action_num
        self.initialize_graph(network_topology)
        self.sess.run(tf.global_variables_initializer())

        if save == True:
            self.save = save
            self.save_period = save_period
            self.name = name
            self.saver = tf.train.Saver()
        else:
            self.save = False

        self.memory = Memory(self.action_num, self.actions, memory_size=memory_size, batch_size=batch_size)

    def initialize_graph(self, config):
        self.price_his = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num - 1, self.history_length, self.feature_num], name="ob")
        self.price_his_ = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num - 1, self.history_length, self.feature_num], name="ob_")
        self.addi_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num], name='addi_inputs')
        self.addi_inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num], name='addi_inputs_')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, ], name='r')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, ], name='a')

        g_b = Graph_builder(config=config)

        # Training network
        with tf.variable_scope('estm_net'):
            self.q_estm = g_b.build_graph(self.price_his, self.addi_inputs, 'training')
            tf.summary.histogram('action_values', self.q_estm)

        # Target network
        with tf.variable_scope('target_net'):
            self.q_tar = g_b.build_graph(self.price_his_, self.addi_inputs_, 'target')


        with tf.variable_scope('q_tar'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_tar, axis=1, name='q_tar_max')
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_estm'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_estm_wa = tf.gather_nd(params=self.q_estm, indices=a_indices)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_estm_wa))
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train'):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='estm_net')

        with tf.name_scope('update_target'):
            # training_params = tf.get_collection('training_params')
            # target_params = tf.get_collection('target_params')
            self.update_target = [tf.assign(t, l) for t, l in zip(t_params, e_params)]

        # self.grad = []
        # for v in tf.trainable_variables():
        #     self.grad.append(tf.gradients(self.loss, [v]))

    def initialize_tb(self):
        for v in tf.trainable_variables():
            tf.summary.histogram(v.name, v)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs/train/" + self.name, self.sess.graph)
        self.tensorboard = True

    def replay(self):

        # s = time()
        obs, as_idx, rs, obs_ = self.memory.sample()
        # e = time()
        # print('Sample use:', e-s)

        _, global_step = self.sess.run([self.train, self.global_step],
                                       feed_dict={self.price_his: obs['history'],
                                                  self.price_his_: obs_['history'],
                                                  self.addi_inputs: obs['weights'],
                                                  self.addi_inputs_: obs_['weights'],
                                                  self.a: as_idx,
                                                  self.r: rs})

        # g = self.sess.run(self.grad[0], feed_dict={self.inputs:obs['history'],self.targets:targets})
        # print(g)

        if global_step % self.update_tar_period == 0:
            self.sess.run(self.update_target)

        if self.tensorboard == True and global_step % self.log_freq == 0:
            s = self.sess.run(self.merged, feed_dict={self.price_his: obs['history'],
                                                      self.price_his_: obs_['history'],
                                                      self.addi_inputs: obs['weights'],
                                                      self.addi_inputs_: obs_['weights'],
                                                      self.a: as_idx,
                                                      self.r: rs})
            self.writer.add_summary(s, global_step)

        # if global_step % 1000 == 0:
            # print('global_step:', global_step)
            # print('save_period:', self.save_period)

        if self.save == True and global_step == self.save_period:
            self.saver.save(self.sess, 'logs/checkpoint/' + self.name, global_step=global_step)

    def choose_action(self, observation, test=False):

        def action_max():
            action_values = self.sess.run(self.q_estm,
                                          feed_dict={self.price_his: observation['history'][np.newaxis, :, :, :],
                                                     self.addi_inputs: observation['weights'][np.newaxis, :]})  # fctur
            return np.argmax(action_values)

        if test == False:
            if np.random.rand() > self.epsilon:
                action_idx = action_max()
                # print('max   ',action_idx)
            else:
                action_idx = np.random.randint(0, self.action_num)  # keyerror: 126
                # print('else   ',action_idx)
        else:
            action_idx = action_max()

        action_weights = self.actions[action_idx]

        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1 - self.epsilon_min) / self.epsilon_decay_period

        return action_idx, action_weights

    def store(self, ob, a, r, ob_):
        self.memory.store(ob, a, r, ob_)

    def get_training_step(self):
        a = self.sess.run(self.global_step)
        return a

    def get_ave_reward(self):
        return self.memory.get_ave_reward()

    def get_lr(self):
        return self.sess.run(self.lr)
        # return self.lr

    def restore(self, name):
        self.saver.restore(self.sess, 'logs/checkpoint/'+name)

    def start_replay(self):
        return self.memory.start_replay()

    def memory_cnt(self):
        return self.memory.memory_pointer

    def network_state(self):
        l = {}
        for v in tf.trainable_variables():
            print(v.name)
            l[v.name] = self.sess.run(v)

        return l

    def action_values(self, o):
        action_values = self.sess.run(self.q_estm,
                                      feed_dict={self.price_his: o['history'][np.newaxis, :, :, :],
                                                 self.addi_inputs: o['weights'][np.newaxis, :]})
        return action_values

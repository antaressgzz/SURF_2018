import numpy as np
from model.memory.memory import Memory
from model.action_dis.action_discretization import action_discretization
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
from model.config import network_config
from config import abspath, fix_random_seed

if fix_random_seed:
    tf.set_random_seed(123)



class Dqn_agent:
    def __init__(self, asset_num, division, feature_num, gamma,
                 network_topology=network_config,
                 learning_rate=0.00025,
                 epsilon=1, epsilon_Min=0.1,
                 dropout=0.5,
                 epsilon_decay_period=100000,
                 update_tar_period=3000,
                 history_length=50,
                 memory_size=10000,
                 batch_size=32,
                 save_period=100000,
                 name='dqn',
                 save=False,
                 GPU=False):

        self.epsilon = epsilon
        self.epsilon_min = epsilon_Min
        self.epsilon_decay_period = epsilon_decay_period
        self.dropout = dropout
        self.asset_num = asset_num
        self.division = division
        self.gamma = gamma
        self.name = name
        self.update_tar_period = update_tar_period
        self.history_length = history_length
        self.feature_num = feature_num
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = learning_rate

        self.action_num, self.actions = action_discretization(self.asset_num, self.division)
        config = tf.ConfigProto()

        if not GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        network_topology['output_num'] = self.action_num

        self.network_config = network_topology
        # print(network_topology)
        self.initialize_graph()
        self.sess.run(tf.global_variables_initializer())

        if save:
            self.save = save
            self.save_period = save_period
            self.name = name
            self.saver = tf.train.Saver()
        else:
            self.save = False

        self.memory = Memory(self.action_num, self.actions, memory_size=memory_size, batch_size=batch_size)

    def initialize_graph(self):
        self.price_his = tf.placeholder(dtype=tf.float32,
                                        shape=[None, self.asset_num - 1, self.history_length, self.feature_num],
                                        name="ob")
        self.price_his_ = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.asset_num - 1, self.history_length, self.feature_num],
                                         name="ob_")
        self.addi_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num], name='addi_inputs')
        self.addi_inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num], name='addi_inputs_')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, ], name='r')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, ], name='a')
        self.keep_prob = tf.placeholder(tf.float32)


        # Training network
        with tf.variable_scope('estm_net'):
            self.fc_input, self.q_pred = self.build_graph(self.price_his, self.addi_inputs)

        # Target network
        with tf.variable_scope('target_net'):
            _, self.tar_pred = self.build_graph(self.price_his_, self.addi_inputs_)

        with tf.variable_scope('q_tar'):
            self.q_target = tf.placeholder(dtype=tf.float32, shape=[None], name='q_target')

        with tf.variable_scope('q_estm_wa'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_estm_wa = tf.gather_nd(params=self.q_pred, indices=a_indices)

        with tf.name_scope('loss'):
            # error = tf.clip_by_value(self.q_target-self.q_estm_wa, -1, 1)
            error = self.q_target-self.q_estm_wa
            square = tf.square(error)
            self.loss = tf.reduce_mean(square)

        with tf.name_scope('train'):
            self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.95, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net'+self.name)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='estm_net'+self.name)
        self.update_target = [tf.assign(t, l) for t, l in zip(t_params, e_params)]


    def build_graph(self, price_his, addi_input):
        kernels = self.network_config['kernels']
        strides = self.network_config['strides']
        filters = self.network_config['filters']
        fc1_size = self.network_config['fc1_size']
        # padding = self.network_config['padding']

        def set_activation(activation):
            if activation == 'relu':
                activation = tf.nn.relu
            elif activation == 'selu':
                activation = tf.nn.selu
            else:
                activation = tf.nn.leaky_relu
            return activation

        cnn_activation = set_activation(self.network_config['cnn_activation'])
        fc_activation = set_activation(self.network_config['fc_activation'])

        w_initializer = tf.truncated_normal_initializer(stddev=self.network_config['w_initializer'])
        b_initializer = tf.constant_initializer(self.network_config['b_initializer'])
        regularizer = layers.l2_regularizer(self.network_config['regularizer'])

        conv = price_his

        for i in range(len(kernels)):
            conv = tf.layers.conv2d(conv, filters=filters[i], kernel_size=kernels[i], strides=strides[i],
                                     trainable=True, activation=cnn_activation,
                                     kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                     kernel_initializer=w_initializer, bias_initializer=b_initializer,
                                     padding='same', name=self.name+'conv'+str(i))

        fc_input = tf.layers.flatten(conv)
        # print('fc_input:', fc_input.shape)

        fc1 = layers.fully_connected(fc_input, num_outputs=fc1_size, activation_fn=fc_activation,
                                     weights_regularizer=regularizer,
                                     weights_initializer=w_initializer,
                                     biases_regularizer=regularizer,
                                     biases_initializer=b_initializer,
                                     trainable=True, scope=self.name+'fc1')
        fc1 = tf.nn.dropout(fc1, self.keep_prob)

        output = layers.fully_connected(fc1, num_outputs=self.action_num, activation_fn=None,
                                        weights_regularizer=regularizer,
                                        biases_regularizer=regularizer,
                                        weights_initializer=w_initializer,
                                        trainable=True, scope=self.name+'output')

        return fc_input, output

    def initialize_tb(self):
        for v in tf.trainable_variables():
            tf.summary.histogram(v.name, v)
        tf.summary.scalar("loss", self.loss),
        tf.summary.histogram("q_values_hist", self.q_pred),
        tf.summary.scalar("max_q_value", tf.reduce_max(self.q_pred))
        tf.summary.histogram('fc_input', self.fc_input)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(abspath+"logs/train/" + self.name, self.sess.graph)
        self.tensorboard = True

    def replay(self):
        obs, action_batch, reward_batch, obs_ = self.memory.sample()

        q_values_next = self.sess.run(self.q_pred, feed_dict={self.price_his: obs_['history'],
                                                              self.addi_inputs:obs_['weights'],
                                                              self.keep_prob: self.dropout})
        best_actions = np.argmax(q_values_next, axis=1)
        q_values_next_target = self.sess.run(self.tar_pred, feed_dict={self.price_his_: obs_['history'],
                                                                       self.addi_inputs_:obs_['weights'],
                                                                       self.keep_prob: self.dropout})
        targets_batch = reward_batch + self.gamma * q_values_next_target[np.arange(len(action_batch)), best_actions]

        # Train
        fd = {self.q_target: targets_batch,
              self.price_his: obs['history'],
              self.addi_inputs: obs['weights'],
              self.a : action_batch,
              self.keep_prob: self.dropout}
        _, global_step = self.sess.run([self.train_op, self.global_step], feed_dict=fd)

        if global_step % self.update_tar_period == 0:
            self.sess.run(self.update_target)

        if self.tensorboard and global_step % 75 == 0:
            s = self.sess.run(self.merged, feed_dict=fd)
            self.writer.add_summary(s, global_step)

        if self.save and global_step % self.save_period == 0:
            self.saver.save(self.sess, abspath+'logs/checkpoint/' + self.name, global_step=global_step)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1 - self.epsilon_min) / self.epsilon_decay_period

    def choose_action(self, observation, test=False):

        def action_max():
            action_values = self.sess.run(self.q_pred,
                        feed_dict={self.price_his: observation['history'][np.newaxis, :, :, :],
                                   self.addi_inputs: observation['weights'][np.newaxis, :],
                                   self.keep_prob: 1})  # fctur
            return np.argmax(action_values)

        if not test:
            if np.random.rand() > self.epsilon:
                action_idx = action_max()
            else:
                action_idx = np.random.randint(0, self.action_num)
        else:
            action_idx = action_max()

        action_weights = self.actions[action_idx]

        return action_idx, action_weights


    def store(self, ob, a, r, ob_):
        self.memory.store(ob, a, r, ob_)

    def get_training_step(self):
        a = self.sess.run(self.global_step)
        return a

    def get_ave_reward(self):
        return self.memory.get_ave_reward()

    def restore(self, name):
        self.saver.restore(self.sess, abspath+'logs/checkpoint/' + name)

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
        action_values = self.sess.run(self.q_pred,
                     feed_dict={self.price_his: o['history'][np.newaxis, :, :, :]})
        return action_values
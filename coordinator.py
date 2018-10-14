import sys
import os
current_path = os.path.abspath(__file__)
OLPS_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + 'OLPS/OLPS_modified')
sys.path.append(OLPS_path)
from OLPS.olps import OLPS
from core.dqn_algo.dqn_agent import Dqn_agent
import matplotlib.pyplot as plt
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf
import pandas as pd
import numpy as np
# %matplotlib inline

df_train = pd.read_hdf('./data/data_raw/JPYGBPEURCAD_4f_1015_30m.hf', key='train')
df_val = pd.read_hdf('./data/data_raw/JPYGBPEURCAD_4f_1015_30m.hf', key='val')
df_test = pd.read_hdf('./data/data_raw/JPYGBPEURCAD_4f_1015_30m.hf', key='test')


class Coordinator:
    def __init__(self, param, name):
        # param to be automatically tuned
        learning_rate = param['learning_rate']
        activation = param['activation']
        fc_size = param['fc_size']
        window_length = int(param['window_length'])

        # model params
        division = 4
        gamma = 0
        name = name
        save_period = 30000
        batch_size = 32
        asset_num = 5
        feature_num = 4
        # window_length = 50

        network_config = {
            'type': 'cnn_fc',
            'kernels': [[1, 3], [1, 3]],
            'strides': [[1, 1], [1, 1]],
            'filters': [3, 3],
            'regularizer': None,
            'activation': activation,
            'fc_size': fc_size,
            'b_initializer': tf.constant_initializer(0),
            'w_initializer': tf.truncated_normal_initializer(stddev=0.01, seed=0),
        }

        self.total_training_step = 10000
        self.replay_period = 4
        self.agent = Dqn_agent(asset_num,
                              division,
                              feature_num,
                              gamma,
                              learning_rate=learning_rate,
                              network_topology=network_config,
                              update_tar_period=3000,
                              dropout=1,
                              epsilon_decay_period=self.total_training_step/5,
                              memory_size=10000,
                              batch_size=batch_size,
                              history_length=window_length,
                              save=True,
                              save_period=save_period,
                              name=name)

        self.env_train = PortfolioEnv(df_train,
                                 steps=1000,
                                 trading_cost=0.0,
                                 window_length=window_length,
                                 talib=False,
                                 augment=0.0,
                                 input='rf',
                                 norm='latest_close',
                                 random_reset=True)

        self.env_val = PortfolioEnv(df_val,
                               steps=1000,
                               trading_cost=0.0,
                               window_length=window_length,
                               talib=False,
                               augment=0.0,
                               input='rf',
                               norm='latest_close',
                               random_reset=False)


    def evaluate(self):

        val_rs, tr_rs = self.train()

        return val_rs, tr_rs

    def train(self, mode='auto', tensorboard=False):

        if tensorboard == True:
            self.agent.initialize_tb()
        else:
            self.agent.tensorboard = False

        training_step = 0
        self.rewards = []
        self.train_rs = []
        val_rs = []

        def get_val_reward():
            val_rewards = []
            for i in range(23):
                ob = self.env_val.reset()
                while True:
                    action_idx, action = self.agent.choose_action(ob, test=True)
                    ob, reward, done, _ = self.env_val.step(action)
                    val_rewards.append(reward)
                    if done:
                        # print(env_val.src.idx)
                        break

            mean = np.mean(val_rewards)
            return mean

        while training_step < self.total_training_step:
            observation = self.env_train.reset()
            while True:
                action_idx, action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env_train.step(action)
                self.rewards.append(reward)
                reward *= 1000
                # reward = np.clip(reward, -1, 1)
                self.agent.store(observation, action_idx, reward, observation_)
                # print('-----------------')
                # print('observation', observation['history'][0, 1:10, :])
                # print('observation_', observation_['history'][0, 1:10, :])

                observation = observation_
                if self.agent.start_replay():
                    # if training_step % 500 == 0:
                    #     print(reward)
                    if self.agent.memory_cnt() % self.replay_period == 0:
                        self.agent.replay()  # update target
                        training_step = self.agent.get_training_step()
                        if (training_step - 1) % 10000 == 0:
                            num_r = 50000
                            train_r = np.sum(self.rewards[-num_r:]) / num_r
                            self.train_rs.append(train_r)
                            val_r = get_val_reward()
                            val_rs.append(val_r)
                            # print('training_step: {}, epsilon: {:.2f}, train_r: {:.2e}, val_r:{:.2e}'.format(
                            # training_step, self.agent.epsilon, train_r, val_r))
                if done:
                    break

        if mode != 'auto':
            print("Successfully trained.")
            x = np.arange(len(self.train_rs))
            fig, ax = plt.subplots()
            ax.plot(x, self.train_rs)
            ax.plot(x, val_rs)
            ax.grid(True, which='both')
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')
            plt.show()
        else:
            return val_rs, self.train_rs


    def back_test(self, env_test, render_mode='usual'):
        ob = env_test.reset()
#         print(ob['history'][0, -10:, 2])
#         print(self.agent.action_values(ob))
        rewards = 0
        while True:
            action_idx, action = self.agent.choose_action(ob, test=True)
            ob, reward, done, _ = env_test.step(action)
            rewards += reward
            if done:
                break

        print('total rewards:', rewards)
        df_info = env_test.return_df()
        df_olps = env_test.OLPS_data()
        olps = OLPS(df_olps=df_olps, df_info=df_info, algo="BAH,BestSoFar")
        olps.plot()
        env_test.render(render_mode)
        sharp, maxDD = env_test.return_SD()
        C_Sharp, C_MDD = olps.IndexComparison(sharp, maxDD)
        print('Sharp ratio Comparison: ', C_Sharp, ' % \nMDD Comparison: ', C_MDD, ' %')
        print('Test is end.')

    def open_tb(self, port):
        path = os.path.abspath('logs/train/'+self.agent.name+' --port='+port)
        os.system('tensorboard --logdir='+path)

    def restore(self, name):
        self.agent.restore(name)

    def network_state(self):
        return  self.agent.network_state()

    def action_values(self, o):
        return self.agent.action_values(o)
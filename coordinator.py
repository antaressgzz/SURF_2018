import sys
import os
current_path = os.path.abspath(__file__)
OLPS_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + 'OLPS/OLPS_modified')
sys.path.append(OLPS_path)
from OLPS.olps import OLPS
import matplotlib.pyplot as plt
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import pandas as pd
import numpy as np
import pprint
from config import FEE, group
if FEE : from model.dqn_algo.dqn_agent_fee import Dqn_agent
else: from model.dqn_algo.dqn_agent import Dqn_agent


df_train = pd.read_hdf('./data/data_raw/'+group+'_4f_1015_30m.hf', key='train')
df_val = pd.read_hdf('./data/data_raw/'+group+'_4f_1015_30m.hf', key='val')
df_test = pd.read_hdf('./data/data_raw/'+group+'_4f_1015_30m.hf', key='test')


class Coordinator:
    def __init__(self, config, name):
        # fixed value
        name = name
        asset_num = 5
        feature_num = 4

        pprint.pprint(config)
        print(group)

        # env config
        window_length = int(config['env']['window_length'])
        input = config['env']['input']
        norm = config['env']['norm']
        talib = config['env']['talib']
        if talib == True:
            feature_num += 2
        trading_cost = config['env']['trading_cost']
        trade_period = config['env']['trading_period']


        # net config
        network_config = config['net']


        # training config
        self.total_training_step = config['train']['steps']
        self.replay_period = config['train']['replay_period']
        self.reward_scale = config['train']['reward_scale']
        learning_rate = config['train']['learning_rate']
        epsilon = config['train']['epsilon']
        division = config['train']['division']
        gamma = config['train']['discount']
        batch_size = config['train']['batch_size']
        memory_size = config['train']['memory_size']
        upd_tar_prd = config['train']['upd_tar_prd']
        dropout = config['train']['dropout']
        save = config['train']['save']
        save_period = config['train']['save_period']
        GPU = config['train']['GPU']

        self.config = config


        self.agent = Dqn_agent(asset_num,
                              division,
                              feature_num,
                              gamma,
                              epsilon=epsilon,
                              learning_rate=learning_rate,
                              network_topology=network_config,
                              update_tar_period=upd_tar_prd,
                              dropout=dropout,
                              epsilon_decay_period=int(self.total_training_step/5),
                              memory_size=memory_size,
                              batch_size=batch_size,
                              history_length=window_length,
                              save=save,
                              save_period=save_period,
                              name=name,
                              GPU=GPU)

        self.env_train = PortfolioEnv(df_train,
                                 steps=1000,
                                 trading_cost=trading_cost,
                                 window_length=window_length,
                                 trade_period=trade_period,
                                 talib=talib,
                                 input=input,
                                 norm=norm,
                                 random_reset=True)

        self.env_val = PortfolioEnv(df_val,
                               steps=1000,
                               trading_cost=trading_cost,
                               window_length=window_length,
                               trade_period=trade_period,
                               talib=talib,
                               input=input,
                               norm=norm,
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
                # print('ob:', observation['history'][:, -4:, :])
                # action = np.ones(5) /5
                observation_, reward, done, info = self.env_train.step(action)
                # print('ob_:', observation_['history'][:, -4:, :])
                # y = 1 / ( observation_['history'][:, -2, 2] / 1000 + 1)
                # y = np.concatenate([[1], y])  # add cash price
                # r = np.log(np.dot(y, action))
                # print('r should be:', r)
                # print('env gives:', reward)
                self.rewards.append(reward)
                reward *= self.reward_scale
                # reward = np.clip(reward, -1, 1)
                self.agent.store(observation, action_idx, reward, observation_)
                observation = observation_
                if self.agent.start_replay():
                    if self.agent.memory_cnt() % self.replay_period == 0:
                        self.agent.replay()  # update target
                        training_step = self.agent.get_training_step()
                        if (training_step - 1) % 10000 == 0:
                            num_r = 50000
                            train_r = np.sum(self.rewards[-num_r:]) / num_r
                            self.train_rs.append(train_r)
                            val_r = get_val_reward()
                            if mode != 'auto':
                                print('training_step: {}, epsilon: {:.2f}, train_r: {:.2e}, val_r:{:.2e}'.format(
                                    training_step, self.agent.epsilon, train_r, val_r))
                            val_rs.append(val_r)
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


    def back_test(self, data_set, steps=8000, random_reset=False):
        if data_set == 'val':
            df = df_val
        else:
            df = df_test
        env_test = PortfolioEnv(df,
                               steps=steps,
                               trading_cost=self.config['env']['trading_cost'],
                                # trading_cost=0.0,
                                trade_period=self.config['env']['trade_period'],
                                window_length=self.config['env']['window_length'],
                               talib=self.config['env']['talib'],
                               input=self.config['env']['input'],
                               norm=self.config['env']['norm'],
                               random_reset=random_reset)
        ob = env_test.reset()

        rewards = 0
        while True:
            action_idx, action = self.agent.choose_action(ob, test=True)
            # print('_ob:', ob['history'][:, -2:, :])
            # action = np.random.rand(5)
            # print(action_idx, action)
            ob, reward, done, _ = env_test.step(action)
            # print('ob:', ob['history'][:, -2:, :])
            # print(reward)
            # print()

            # print(action)
            # print(reward)
            rewards += reward
            if done:
                break

        print('total rewards:', rewards)
        print('ave rewards:', rewards / steps)
        df_info = env_test.return_df()
        df_olps = env_test.OLPS_data()
        olps = OLPS(df_olps=df_olps, df_info=df_info, algo="BAH,BestSoFar")
        olps.plot()
        env_test.render('usual')
        sharp, maxDD = env_test.return_SD()
        C_Sharp, C_MDD = olps.IndexComparison(sharp, maxDD)
        print('Sharp ratio Comparison: ', C_Sharp, ' % \nMDD Comparison: ', C_MDD, ' %')
        print('Test is end.')

    def open_tb(self, port):
        path = os.path.abspath('logs/train/'+self.agent.name+' --port='+port)
        os.system('tensorboard --logdir='+path)

    def restore(self, name):
        self.agent.restore(name)

    def restore_price_predictor(self, name):
        self.agent.restore_price_predictor(name)

    def network_state(self):
        return  self.agent.network_state()

    def action_values(self, o):
        return self.agent.action_values(o)
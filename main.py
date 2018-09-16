from coordinator import Coordinator
from core.dqn_algo.dqn_agent import Dqn_agent
import pandas as pd
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf
import numpy as np

df_train = pd.read_hdf('./data/forex_3f_4.hf', key='train')
df_test = pd.read_hdf('./data/forex_3f_4.hf', key='test')

division = 4
gamma = 0.2
name = '0606'
total_training_step = 400000
replay_period = 4
save_period = 50000
batch_size = 32
GPU = False
asset_num = 5
feature_num = 3
window_length = 500
trade_period = 1

network_config = {
        'type':'cnn_fc',
        'kernels':[[1, 8], [1, 4]],
        'strides':[[1, 4], [1, 2]],
        'filters':[6, 4],
        'cnn_bias': True,
        'regularizer': tf.contrib.layers.l2_regularizer(0.01),
        'activation': tf.nn.selu,
        'fc_size': 256,
        'initializer':tf.truncated_normal_initializer(stddev=0.001),
        'weights_pos': None
    }


agent = Dqn_agent(asset_num,
                  division,
                  feature_num,
                  gamma,
                  network_topology=network_config,
                  update_tar_period=3000,
                  epsilon=1,
                  epsilon_Min=0.1,
                  epsilon_decay_period=total_training_step/5,
                  memory_size=10000,
                  batch_size=batch_size,
                  history_length=window_length,
                  save=True,
                  save_period=save_period,
                  name=name,
                  GPU=GPU)
#
coo = Coordinator(agent)
#

# test input 'price'
env = PortfolioEnv(df_train,
                   steps=100,
                   trading_cost=0.0,
                   trade_period=trade_period,
                   window_length=window_length,
                   input='price',
                   norm=None,
                   random_reset=True)

ob = env.reset()

price = np.concatenate([[1.0], ob['history'][:, -1, 2]])
print('current_price:', price)

action = np.array([0.1,0.2,0.3,0.4,0])
ob_, r, d, i = env.step(action)

price_ = np.concatenate([[1.0], ob_['history'][:, -1, 2]])
print('next_price:', price_)

reward = np.log(np.dot((price_/price), action))

print('reward should be', reward, 'and the env gives', r)

# test input 'rf'
env_2 = PortfolioEnv(df_train,
                   steps=100,
                   trading_cost=0.0,
                   trade_period=trade_period,
                   window_length=window_length,
                   input='rf',
                   norm=None,
                   random_reset=True)

ob = env_2.reset()

# price = np.concatenate([[1.0], ob['history'][:, -1, 2]])
# print('current_price:', price)

action = np.array([0.1,0.2,0.3,0.4,0])
ob_, r, d, i = env_2.step(action)

rf = np.concatenate([[0], ob_['history'][:, -1, 2]])
print('rise and fall:', rf)

reward = np.log(np.dot(rf+1, action))

print('reward should be', reward, 'and the env gives', r)

# coo.back_test(env_test1, render_mode='usual')

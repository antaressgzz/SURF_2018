from coordinator import Coordinator
from core.dqn_algo.dqn_agent import Dqn_agent
import pandas as pd
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf
import numpy as np

df_train = pd.read_hdf('./data/forex_3f_4.hf', key='train')
df_test = pd.read_hdf('./data/forex_3f_4.hf', key='test')

division = 4
gamma = 0
name = '2204'
total_training_step = 200000
replay_period = 4
save_period = 50000
batch_size = 32
GPU = False
asset_num = 5
feature_num = 3
window_length = 50
trade_period = 1

network_config = {
        'type':'cnn_fc',
        'kernels':[[1, 5], [1, 4]],
        'strides':[[1, 3], [1, 2]],
        'filters':[8, 8],
        'cnn_bias': True,
        'regularizer': None,
        'activation': tf.nn.selu,
        'fc_size': 256,
        'b_initializer':tf.constant_initializer(0),
        'w_initializer':tf.truncated_normal_initializer(stddev=0.01),
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

coo = Coordinator(agent)

env = PortfolioEnv(df_train,
                   steps=100,
                   trading_cost=0.0,
                   trade_period=trade_period,
                   window_length=window_length,
                   input='rf',
                   norm=None,
                   random_reset=True)

coo.train(env, total_training_step, replay_period, True)

env_test = PortfolioEnv(df_test,
                   steps=1000,
                   trading_cost=0.0,
                   trade_period=trade_period,
                   window_length=window_length,
                   input='rf',
                   norm=None,
                   random_reset=False)

coo.back_test(env_test, 'usual')
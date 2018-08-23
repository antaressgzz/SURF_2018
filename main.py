from coordinator import Coordinator
from core.dqn_algo.dqn_agent import Dqn_agent
import pandas as pd
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf
import numpy as np

df_train = pd.read_hdf('./data/forex_3f_2.hf', key='train')
df_test = pd.read_hdf('./data/forex_3f_2.hf', key='test')


division = 3
gamma = 0.9
name = 'test'
total_training_step = 1000000
replay_period = 4
save_period = 499999
batch_size = 32
GPU = False
asset_num = 5
feature_num = 3
window_length = 1500
trade_period = 1

network_config = {
        'type':'cnn_fc',                                     #
        'activation': tf.nn.selu,
        'cnn_bias': False,
        'fc_bias': False,
        'initializer':tf.truncated_normal_initializer(stddev=0.01)

    }


agent = Dqn_agent(asset_num,
                  division,
                  feature_num,
                  gamma,
                  network_topology=network_config,
                  update_tar_period=5000,
                  epsilon=1,
                  epsilon_Min=0.1,
                  epsilon_decay_period=total_training_step/5,
                  memory_size=50000,
                  batch_size=batch_size,
                  history_length=window_length,
                  save=True,
                  save_period=save_period,
                  name=name,
                  GPU=GPU)
#
coo = Coordinator(agent, trade_period)
#
env = PortfolioEnv(df_train,
                   steps=256,
                   trading_cost=0.0,
                   window_length=window_length,
                   input='rf',
                   norm=None,
                   random_reset=False)


coo.train(env, total_training_step=total_training_step, replay_period=replay_period, tensorboard=True)
#

# env_test = PortfolioEnv(df_test,
#                         steps=5000,
#                         trading_cost=0.0,
#                         window_length=window_length,
#                         norm=2,
#                         scale=True,
#                         random_reset=False)


# coo.restore('dqn-test-49998')

# coo.back_test(env_test, render_mode='usual')
# coo.open_tb(port='6006')
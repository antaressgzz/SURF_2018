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
name = 'dqn-test'
save = True
total_training_step = 10000
replay_period = 4
save_period = int((total_training_step - 2) / 3)
batch_size = 32
GPU = False
asset_num = 5
feature_num = 3
window_length = 100
trade_period = 8

# network_config = {
#         'type': 'fc',
#         'fc_layer': [50],
#         'fc_activation': tf.nn.leaky_relu,
#         'additional_input': 'weights',
#         'output_num': None,
#         'output_activation': None,
#         'weights_regularization': None,
#         'weights_initializer': tf.truncated_normal_initializer(stddev=0.1), # tf.constant_initializer(0.005),
#         'bias_regularization': tf.keras.regularizers.l2(l=0.1)
#     }

network_config = {
        'type':'cnn_fc',                                     #
        'cnn_layer': {'kernel':(3, 98), 'filter':(2, 10)},   # kernal size, number of kernels
        'cnn_activation': tf.nn.selu,
        'cnn_bias': False,
        'fc_layer': [100],                                   # 0 to 2 hidden layers, size 1000
        'fc_activation': tf.nn.relu,
        'additional_input': None,                            # last_weights or None
        'output_activation': None,
        'weights_regularization': None,
        'weights_initializer': tf.truncated_normal_initializer(stddev=0.01), # tf.constant_initializer(0.005),
        'bias_regularization': None
    }
#
#
agent = Dqn_agent(asset_num,
                  division,
                  feature_num,
                  gamma,
                  network_topology=network_config,
                  learning_rate_decay_step=int(total_training_step/30),
                  update_tar_period=2000,
                  epsilon=1,
                  epsilon_Min=0.1,
                  epsilon_decay_period=total_training_step*replay_period/4,
                  memory_size=500*batch_size,
                  batch_size=batch_size,
                  history_length=window_length,
                  log_freq=50,
                  save_period=save_period,
                  save=save,
                  name=name,
                  GPU=GPU)
#
coo = Coordinator(agent, trade_period)
#
env = PortfolioEnv(df_train,
                   steps=256,
                   trading_cost=0.0,
                   window_length=window_length,
                   input_rf=False,
                   norm=1,
                   scale=True,
                   random_reset=False)


# coo.train(env, total_training_step=total_training_step, replay_period=replay_period, tensorboard=True)



env_test = PortfolioEnv(df_test,
                        steps=5000,
                        trading_cost=0.0,
                        window_length=window_length,
                        norm=2,
                        scale=True,
                        random_reset=False)


# coo.restore('dqn-test-49998')



coo.back_test(env_test, render_mode='usual')
# coo.open_tb(port='6666')
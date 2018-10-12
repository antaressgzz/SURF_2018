from coordinator import Coordinator
from core.dqn_algo.dqn_agent_fee import Dqn_agent
import pandas as pd
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf
import numpy as np
# %matplotlib inline

df_train = pd.read_hdf('./data/data_raw/forex_3f_1.hf', key='train')
df_test = pd.read_hdf('./data/data_raw/forex_3f_1.hf', key='test')

division = 4
gamma = 0.99
name = 'test'
print('model:',name)
total_training_step = 150000
replay_period = 4
save_period = 30000
batch_size = 32
GPU = False
asset_num = 5
feature_num = 3
window_length = 50
trade_period = 1

network_config = {
        'type':'cnn_fc',
        'freeze_cnn': False,
        'kernels':[[1, 3], [1, 3], [1, 3]],
        'strides':[[1, 1], [1, 1], [1, 1]],
        'filters':[3, 4, 5],
        'fc1_size':64,
        'fc2_size':64,
        'regularizer': None,
        'activation': tf.nn.selu,
        'b_initializer':tf.constant_initializer(0.1),
        'w_initializer':tf.truncated_normal_initializer(stddev=0.1),
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
                  process_cost=False,
                  save=True,
                  save_period=save_period,
                  name=name,
                  GPU=GPU)

coo = Coordinator(agent)

env = PortfolioEnv(df_train,
                   steps=1000,
                   trading_cost=0.001,
                   trade_period=trade_period,
                   window_length=window_length,
                   talib=False,
                   augment=0.0,
                   input='rf',
                   norm='latest_close',
                   random_reset=True)

env_test = PortfolioEnv(df_test,
                   steps=1000,
                   trading_cost=0.001,
                   trade_period=trade_period,
                   window_length=window_length,
                   talib=False,
                   augment=0.0,
                   input='rf',
                   norm='latest_close',
                   random_reset=False)


coo.train(env, env_test, total_training_step, replay_period, True)
# coo.restore('0903-30000')

# coo.back_test(env_test, 'usual')
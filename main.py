from coordinator import Coordinator
from core.dqn_algo.dqn_agent import Dqn_agent
import pandas as pd
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf

df_train = pd.read_hdf('./data/forex_30m_CLOSE_1.hf', key='train')
df_test = pd.read_hdf('./data/forex_30m_CLOSE_1.hf', key='test')

division = 3
gamma = 0.7
name = 'dqn-haha'
tensorboard = False
save = True
total_training_step = 2000
replay_period = 4
save_period = total_training_step-1
batch_size = 32
GPU = False
asset_num = 5
feature_num = 1

network_config = {
        'type': 'fc',
        'fc_layer': (1000, 1000),
        'fc_activation': tf.nn.tanh,
        'additional_input': 'portfolio_value',
        'output_num': None,
        'output_activation': None,
        'weights_regularization': None,
        'bias_regularization': None
    }

agent = Dqn_agent(asset_num, division, feature_num, gamma,
                  network_topology=network_config,
                  learning_rate_decay_step=int(total_training_step/30), update_tar_period=1000,
                  epsilon=1, epsilon_Min=0.1, epsilon_decay_period=total_training_step*replay_period/5,
                  memory_size=500*batch_size, batch_size=batch_size,
                  history_length=50,
                  tensorboard=tensorboard, log_freq=50,
                  save_period=save_period, save=save,
                  name=name, GPU=GPU)

env = PortfolioEnv(df_train,
                   steps=256,
                   trading_cost=0.0,
                   window_length=50,
                   scale=False,
                   random_reset=False)

coo = Coordinator(agent, env, total_training_step=total_training_step, replay_period=replay_period)

coo.train()

env_test = PortfolioEnv(df_train,
                   steps=2500,
                   trading_cost=0.0,
                   window_length=50,
                   scale=False,
                   random_reset=False)

# coo.restore('dqn-haha-1990')

coo.back_test(env_test, render_mode='usual')

# coo.open_tb(port='8008')
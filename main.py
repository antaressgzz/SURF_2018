from coordinator import Coordinator
from core.dqn_algo.dqn_agent import Dqn_agent
import pandas as pd
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf
import numpy as np

df_train = pd.read_hdf('./data/forex_30m_CLOSE_1.hf', key='train')
df_test = pd.read_hdf('./data/forex_30m_CLOSE_1.hf', key='test')

division = 1
gamma = 0
name = 'dqn-ha'
save = False
total_training_step = 50000
replay_period = 2
save_period = total_training_step-1
batch_size = 16
GPU = False
asset_num = 5
feature_num = 1
window_length = 50

network_config = {
        'type': 'fc',
        'fc_layer': [1000],
        'fc_activation': tf.nn.selu,
        'additional_input': 'weights',
        'output_num': None,
        'output_activation': None,
        'weights_regularization': None,
        'weights_initializer': tf.constant_initializer(0.005),
        'bias_regularization': tf.keras.regularizers.l2(l=0.1)
    }

agent = Dqn_agent(asset_num, division, feature_num, gamma,
                  network_topology=network_config,
                  learning_rate_decay_step=int(total_training_step/30), update_tar_period=1000,
                  epsilon=1, epsilon_Min=0.1, epsilon_decay_period=total_training_step*replay_period/5,
                  memory_size=500*batch_size, batch_size=batch_size,
                  history_length=window_length,
                  log_freq=50, save_period=save_period, save=save,
                  name=name, GPU=GPU)

env = PortfolioEnv(df_train,
                   steps=256,
                   trading_cost=0.00007,
                   window_length=window_length,
                   scale=False,
                   random_reset=False)

coo = Coordinator(agent, env)

ob = env.reset()
for i in range(5):
    print(coo.action_values(ob))
    ob, a, r, ob_ = env.step(np.ones(5))

coo.train(total_training_step=total_training_step, replay_period=replay_period, tensorboard=True)


ob = env.reset()
for i in range(10):
    print(coo.action_values(ob))
    print(np.argmax(coo.action_values(ob)))
    ob, a, r, ob_ = env.step(np.ones(5))

env_test = PortfolioEnv(df_train,
                        steps=2500,
                        trading_cost=0.0,
                        window_length=window_length,
                        scale=False,
                        random_reset=False)


# coo.restore('dqn-consini-199995')

# l = coo.network_state()
# print(l['training_network/output/weights:0'])

coo.back_test(env_test, render_mode='usual')

# coo.open_tb(port='8009')
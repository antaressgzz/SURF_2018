from coordinator import Coordinator
from core.dqn_algo.dqn_agent import Dqn_agent
import pandas as pd
from rl_portfolio_Env_Modified.environments import PortfolioEnv
import tensorflow as tf
import numpy as np

df_train = pd.read_hdf('./data/forex_C_1.hf', key='train')
df_test = pd.read_hdf('./data/forex_C_1.hf', key='test')

division = 3
gamma = 0.9
name = 'dqn-sss'
save = True
total_training_step = 20000
replay_period = 4
save_period = total_training_step - 2
batch_size = 32
GPU = True
asset_num = 5
feature_num = 1
window_length = 100

# network_config = {
#         'type': 'fc',
#         'fc_layer': [50, 50],
#         'fc_activation': tf.nn.leaky_relu,
#         'additional_input': 'weights',
#         'output_num': None,
#         'output_activation': None,
#         'weights_regularization': None,
#         'weights_initializer': tf.truncated_normal_initializer(stddev=0.1), # tf.constant_initializer(0.005),
#         'bias_regularization': tf.keras.regularizers.l2(l=0.1)
#     }

network_config = {
        'type':'cnn_fc',                                      #
        'cnn_layer': {'kernel':(10, 10), 'filter':(5, 5)},    # kernal size, number of kernels
        'cnn_activation': tf.nn.leaky_relu,
        'fc_layer': [100, 100],                              # 0 to 2 hidden layers, size 1000
        'fc_activation': tf.nn.leaky_relu,
        'additional_input': 'weights',                        # last_weights or None                                #
        'output_activation': None,
        'weights_regularization': None,
        'weights_initializer': tf.truncated_normal_initializer(stddev=0.01), # tf.constant_initializer(0.005),
        'bias_regularization': None
    }


agent = Dqn_agent(asset_num, division, feature_num, gamma,
                  network_topology=network_config,
                  learning_rate_decay_step=int(total_training_step/30), update_tar_period=2000,
                  epsilon=1, epsilon_Min=0.1, epsilon_decay_period=total_training_step*replay_period/4,
                  memory_size=500*batch_size, batch_size=batch_size,
                  history_length=window_length,
                  log_freq=50, save_period=save_period, save=save,
                  name=name, GPU=GPU)

coo = Coordinator(agent)

env = PortfolioEnv(df_train,
                   steps=256,
                   trading_cost=0.00007,
                   window_length=window_length,
                   input_rf=True,
                   norm='sig_0.5',
                   scale=False,
                   random_reset=False)

# ob = env.reset()
# for i in range(5):
#     # print(coo.action_values(ob))
#     print(ob['history'])
#     # print(ob['history'])
#     ob, a, r, ob_ = env.step(np.ones(5))



# l = coo.network_state()
# print(l['training_network/h2/weights_0'])

coo.train(env, total_training_step=total_training_step, replay_period=replay_period, tensorboard=False)


env_test = PortfolioEnv(df_test,
                        steps=10000,
                        trading_cost=0.00007,
                        window_length=window_length,
                        input_rf=True,
                        norm='sig_0.5',
                        scale=False,
                        random_reset=False)
# ob = env_test.reset()
# print(ob['weights'])
# for i in range(10):
#     print(ob['history'])
#     # print(coo.action_values(ob))
#     print(np.argmax(coo.action_values(ob)))
#     ob, r, done, info = env_test.step(np.ones(5)/5)

# coo.restore('')

# l = coo.network_state()
# print(l['training_network/h2/weights_0'])

coo.back_test(env_test, render_mode='usual')

# coo.open_tb(port='6666')
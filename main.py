import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
from hypertune import start_commander, start_workers
from coordinator import Coordinator
from config import get_config
from hypertune import construct_config
# %matplotlib inline


mode = 'single'
name = '2200'

if mode == 'auto':
    start_commander()
    workers = start_workers(6)
else:
    param_space = {
        # train
        'steps': 240000,
        'learning_rate': 0.0001270642752036939,
        'reward_scale': 1200,
        # 'discount': hp.uniform('discount', 0, 1),
        'epsilon': 0.1,
        'batch_size': 128,
        'replay_period': 8,
        'division': 3,
        'dropout': 0.5646002767503104,
        # net
        'activation': 'selu',
        'fc_size': 64,
        'kernels': [[1, 10],
                   [4, 5]],
        'filters': [8, 2],
        'strides': [[1, 2], [1, 3]],
        'regularizer': 1.0081378980963206e-05,
        # env
        'padding': 'valid',
        'window_length': 250,
        'input': 'rf',
        'norm': 'latest_close',
    }
    config = get_config(False)
    # config['train'] = {
    #         'save': True,
    #         'period': 40000
    # }
    config = construct_config(config, param_space)
    model = Coordinator(config, name)
    # model.restore('2102-160000')
    model.train('single', True)
    model.back_test('test', 12000, True)

# from rl_portfolio_Env_Modified.environments.portfolio import PortfolioEnv
# import pandas as pd
# import numpy as np
#
# df_train = pd.read_hdf('./data/data_raw/JPYGBPEURCAD_4f_1015_30m.hf', key='train')
# df_val = pd.read_hdf('./data/data_raw/JPYGBPEURCAD_4f_1015_30m.hf', key='val')
# df_test = pd.read_hdf('./data/data_raw/JPYGBPEURCAD_4f_1015_30m.hf', key='test')
#
# env = PortfolioEnv(df_train,
#                    trading_cost=0,
#                    window_length=100,
#                    input='rf',
#                    norm='latest_close'
#                    )
#
# ob = env.reset()
#
# for ii in range(10):
#     ob, r1, d, i = env.step(np.ones(5))
#     print(ob['history'][:, -1, 2])
#     y = ob['history'][:, -1, 2] / 1000 + 1
#     y = np.concatenate([[1], y])  # add cash price
#     print(y)
#     r2 = np.log(np.dot(np.ones(5) / 5, y))
#     print('r should be:', r2)
#     print('env gives:', r1)


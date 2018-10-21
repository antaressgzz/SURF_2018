import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
from hypertune import start_commander, start_workers
from coordinator import Coordinator
from config import get_config
from hypertune import construct_config


mode = 'auto'

if mode == 'auto':
    start_commander()
    workers = start_workers(6)
else:
    param_space = {
        # train
        'steps': 150000,
        'learning_rate': 0.00025,
        'reward_scale': 1000,
        # 'discount': hp.uniform('discount', 0, 1),
        'batch_size': 32,
        'replay_period': 4,
        'division': 4,
        'dropout': 0.7261497109138413,
        # net
        'activation': 'selu',
        'fc_size': 128,
        'kernels': [[1, 3],
                   [1, 3]],
        'filters': [3, 3],
        'strides': [[1, 2], [1, 2]],
        'regularizer': 0.000697715451566933,
        # env
        'window_length': 100,
        'input': 'rf',
        'norm': 'latest_close',
    }
    config = get_config(False)
    # config['train'] = {
    #         'save': True,
    #         'period': 40000
    # }
    config = construct_config(config, param_space)
    model = Coordinator(config, '2101')
    model.train('single', True)
    # model.restore('1700-120000')
    # model.back_test('val', 1000, False)

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


FEE = False
EXP_KEY = 2700
MAX_EVALS = 100
number_workers = 12
mode = 'single'
asset_group = 0
# name = 'test_ag'+str(asset_group)
# name = '-5.28'
abspath = '/Users/zhangziyang/PycharmProjects/SURF_2019/'
fix_random_seed = True
# abspath = '/home/ubuntu/documents/SURF_2018/'
set_group = ['JPYGBPEURCAD', 'JPYCHFGBPCAD', 'CHFGBPCADEUR', 'JPYCHFCADEUR', 'JPYCHFGBPEUR']
group = set_group[asset_group]
save = (mode == 'single')

tuned_config = {"env": {"window_length": 200,
                         "input": "rf",
                         "norm": "previous",
                         "talib": False,
                         "trading_period": 8,
                         ################## change this ###############
                         "trading_cost": 0.0},
                         ############### change these #################
     "train": {"learning_rate": 0.00015309751147495794,
               "division": 5,
               "epsilon": 1,
               "reward_scale": 1200.0,
               "batch_size": 32,
               "replay_period": 8,
               "memory_size": 20000,
               "dropout": 0.4893817909286,
               ############### change these #################
               "upd_tar_prd": 3000,
               "steps": 200000,
               "save": save,
               "save_period": 40000,
               "GPU": False,
               "discount": 0.0},
                ############### change these #################
     "net": {"kernels": [[1, 5], [4, 5]],
             "strides": [[1, 3], [1, 3]],
             "filters": [6, 7],
             "padding": "same",
             "regularizer": 0.003781580642373421,
             "fc1_size": 64,
             "b_initializer": 0,
             "w_initializer": 0.01,
             "cnn_activation": "relu",
             "fc_activation": "leaky_relu",
             "output_num": 210,
             #################### change this ###################
             "fc2_size": 64,
             "process_cost": True,
             "freeze_cnn": False
             ############### change these #################
             }}

def get_config(fee):
        config = {}
        config['env'] = {
                'window_length': 100,
                'trading_period': 1,
                'input': 'rf',
                'norm': 'lastest_close',
                'talib': False,
                }
        if mode == 'parallel':
            save = False
        else:
            save = True
        config['train'] = {
                'learning_rate': 0.0003891689485800337,
                'division': 4,
                'epsilon': 1,
                'reward_scale': 1000,
                'batch_size': 16,
                'steps': 100000,
                'replay_period': 2,
                'memory_size': 20000,
                'upd_tar_prd': 2000,
                'dropout': 0.5,
                'save': save,
                'save_period': 40000,
                ##########################
                'GPU': False
                ##########################
        }
        if fee is True:
                config['net'] =     {
                                        'freeze_cnn': False,
                                        'kernels':[[1, 3], [1, 3]],
                                        'strides':[[1, 1], [1, 1]],
                                        'filters':[3, 3],
                                        'fc1_size':64,
                                        'fc2_size':64,
                                        'padding': 'same',
                                        'regularizer': 0.000697715451566933,
                                        'b_initializer': 0,
                                        'w_initializer': 0.01,
                                        'cnn_activation': 'selu',
                                        'fc_activation': 'selu'
                                    }
                config['env']['trading_cost'] = 0.0001
                config['train']['discount'] = 0.9
        else:
                config['net'] =    {
                                        'kernels':[[1, 3], [1, 3]],
                                        'strides':[[1, 1], [1, 1]],
                                        'filters':[3, 3],
                                        'padding': 'same',
                                        'regularizer': 0.000697715451566933,
                                        'fc_size': 32,
                                        'b_initializer': 0,
                                        'w_initializer': 0.01,
                                        'cnn_activation': 'selu',
                                        'fc_activation': 'selu'
                                    }
                config['env']['trading_cost'] = 0
                config['train']['discount'] = 0
        return config
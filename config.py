FEE = False
EXP_KEY = 2300
MAX_EVALS = 200
number_workers = 12
mode = 'auto'
name = '2300'

config_fee = {
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

config_nofee = {
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

def get_config(fee):
        config = {}
        config['env'] = {
                'window_length': 100,
                'input': 'rf',
                'norm': 'lastest_close',
                'talib': False,
                }
        if mode == 'auto':
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
                'GPU': True
                ##########################
        }
        if fee is True:
                config['net'] = config_fee
                config['env']['trading_cost'] = 0.0001
                config['train']['discount'] = 0.9
        else:
                config['net'] = config_nofee
                config['env']['trading_cost'] = 0
                config['train']['discount'] = 0
        return config

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
    'cnn_activation': 'selu',
    'fc_activation': 'selu',
    # env
    'padding': 'valid',
    'window_length': 250,
    'input': 'rf',
    'norm': 'latest_close',
}

param_space_fee = {
    # train
    'steps': 240000,
    'learning_rate': 0.0001270642752036939,
    'reward_scale': 1200,
    'process_cost': False,
    'discount': 0.9,
    'epsilon': 1,
    'batch_size': 128,
    'replay_period': 8,
    'division': 3,
    'dropout': 0.5646002767503104,
    # net
    'freezn_cnn': False,
    'fc1_size': 64,
    'fc2_size': 64,
    'kernels': [[1, 10],
                [4, 5]],
    'filters': [8, 2],
    'strides': [[1, 2], [1, 3]],
    'regularizer': 1.0081378980963206e-05,
    'cnn_activation': 'selu',
    'fc_activation': 'selu',
    # env
    'padding': 'valid',
    'window_length': 250,
    'input': 'rf',
    'norm': 'latest_close',
}
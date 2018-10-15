config_fee = {
        'freeze_cnn': False,
        'kernels':[[1, 3], [1, 3]],
        'strides':[[1, 1], [1, 1]],
        'filters':[3, 3],
        'fc1_size':64,
        'fc2_size':64,
        'regularizer': None,
        'activation': 'selu',
        'b_initializer': 0,
        'w_initializer': 0.01,
    }

config_nofee = {
        'kernels':[[1, 3], [1, 3]],
        'strides':[[1, 1], [1, 1]],
        'filters':[3, 3],
        'regularizer': None,
        'activation': 'selu',
        'fc_size': 128,
        'b_initializer': 0,
        'w_initializer': 0.01,
    }

def get_config(fee):
        config = {}
        config['env'] = {
                'window_length': 50,
                'input': 'rf',
                'norm': 'lastest_close',
                'talib': False,
                'argument': 0
                }
        config['train'] = {
                'learning_rate': 0.00025,
                'division': 4,
                'reward_scale': 1000,
                'batch_size': 32,
                'steps': 100000,
                'replay_period': 4,
                'memory_size': 20000,
                'upd_tar_prd': 2000,
                'dropout': 0.5,
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
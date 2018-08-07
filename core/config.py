import tensorflow as tf

network_config = {
    'cnn_fc':{
        'type':'cnn_fc',
        'cnn_layer': {'kernel':(3, 48), 'filter':(2, 10)},
        'cnn_activation': tf.nn.relu,
        'fc_layer': (1000, 1000),
        'fc_activation': tf.nn.tanh,
        'additional_input': 'portfolio_value',
        'output_num': None,
        'output_activation': None,
        'weights_regularization': None,
        'bias_regularization': None
    },

    'fc':{
        'type': 'fc',
        'fc_layer': (1000, 1000), # 0 to 2 hidden layers
        'fc_activation': tf.nn.tanh,
        'additional_input': 'portfolio_value', # portfolio_value, last_weights, None
        'output_num': None,
        'output_activation': None,
        'weights_regularization': None,
        'bias_regularization': None
    },

    'dilated_cnn':{

    },
    
    'lstm':{

    }

}
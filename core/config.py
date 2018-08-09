import tensorflow as tf

network_config = {
    'cnn_fc':{
        'type':'cnn_fc',                                      #
        'cnn_layer': {'kernel':(3, 48), 'filter':(2, 10)},    # kernal size, number of kernels
        'cnn_activation': tf.nn.relu,
        'fc_layer': [1000, 1000],                             # 0 to 2 hidden layers, size 1000
        'fc_activation': tf.nn.tanh,
        'additional_input': 'weights',                        # last_weights or None
        'output_num': None,                                   #
        'output_activation': None,
        'weights_regularization': None,
        'bias_regularization': None
    },

    'fc':{
        'type': 'fc',
        'fc_layer': [1000, 1000],                             # 0 to 2 hidden layers, size 1000
        'fc_activation': tf.nn.tanh,
        'additional_input': 'weights',                        # last_weights or None
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
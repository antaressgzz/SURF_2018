import tensorflow as tf
import tensorflow.contrib.layers as layers

class Graph_builder:

    def __init__(self, config):
        self.config = config

    def build_graph(self, price_his, addi_inputs, scope):

        type = self.config['type']
        output = None

        if type == 'cnn_fc':
            output = self.cnn_fc_builder(price_his, addi_inputs, scope)

        elif type == 'fc':
            output = self.fc_builder(price_his, addi_inputs, scope)

        elif type == 'dilated_cnn':
            pass

        elif type == 'lstm':
            pass

        else:
            print('Invalid Network Topology.')

        return output

    def cnn_fc_builder(self, price_his, addi_inputs, scope):
        output = None

        initializer = self.config['weights_initializer']
        additional_input = self.config['additional_input']
        filter = self.config['cnn_layer']['filter']
        kernel = self.config['cnn_layer']['kernel']
        cnn_activation = self.config['cnn_activation']
        layers_sizes = self.config['fc_layer']
        num_layer = len(layers_sizes)
        activation = self.config['fc_activation']
        num_output = self.config['output_num']
        output_activation = self.config['output_activation']
        weights_regularization = self.config['weights_regularization']
        bias_regularization = self.config['bias_regularization']

        conv1 = layers.conv2d(price_his, num_outputs=filter[0], kernel_size=(1, kernel[0]), stride=1, trainable=True,
                              weights_initializer=initializer,activation_fn=cnn_activation,
                              weights_regularizer=weights_regularization, biases_regularizer=bias_regularization,
                              padding="VALID", scope='conv1')
        conv2 = layers.conv2d(conv1, num_outputs=filter[1], kernel_size=(1, kernel[1]), stride=1, trainable=True,
                              weights_regularizer=weights_regularization, biases_regularizer=bias_regularization,
                              weights_initializer=initializer, activation_fn=cnn_activation,
                              padding="VALID", scope='conv2')

        if additional_input is None:
            fc_input = tf.layers.flatten(conv2)
        else:
            fc_input = tf.concat([tf.layers.flatten(conv2), addi_inputs], axis=1)

        if num_layer == 0:
            output = layers.fully_connected(fc_input, num_outputs=num_output, activation_fn=output_activation,
                                             trainable=True,
                                             weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                             weights_regularizer=weights_regularization,
                                             biases_regularizer=bias_regularization,
                                             scope='output')
        elif num_layer == 1:
            h1 = layers.fully_connected(fc_input, num_outputs=layers_sizes[0], activation_fn=activation, trainable=True,
                                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                        weights_regularizer=weights_regularization,
                                        biases_regularizer=bias_regularization,
                                        scope='h1')
            output = layers.fully_connected(h1, num_outputs=num_output, activation_fn=output_activation, trainable=True,
                                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='output')
        elif num_layer == 2:
            h1 = layers.fully_connected(fc_input, num_outputs=layers_sizes[0], activation_fn=activation, trainable=True,
                                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                        weights_regularizer=weights_regularization,
                                        biases_regularizer=bias_regularization,
                                        scope='h1')
            h2 = layers.fully_connected(h1, num_outputs=layers_sizes[1], activation_fn=activation, trainable=True,
                                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                        weights_regularizer=weights_regularization,
                                        biases_regularizer=bias_regularization,
                                        scope='h2')
            output = layers.fully_connected(h2, num_outputs=num_output, activation_fn=output_activation, trainable=True,
                                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='output')
        else:
            print('Too much layers.')

        return output

    def fc_builder(self, price_his, addi_inputs, scope):
        output = None

        initializer = self.config['weights_initializer']
        layers_sizes = self.config['fc_layer']
        num_layer = len(layers_sizes)
        additional_input = self.config['additional_input']
        activation = self.config['fc_activation']
        num_output = self.config['output_num']
        output_activation = self.config['output_activation']
        weights_regularization = self.config['weights_regularization']
        bias_regularization = self.config['bias_regularization']

        if additional_input is None:
            fc_input = tf.layers.flatten(price_his)
        else:
            fc_input = tf.concat([tf.layers.flatten(price_his), addi_inputs], axis=1)

        if num_layer == 0:
            output = layers.fully_connected(fc_input, num_outputs=num_output, activation_fn=output_activation,
                                            trainable=True,
                                            weights_initializer=initializer,
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='output')
        elif num_layer == 1:
            h1 = layers.fully_connected(fc_input, num_outputs=layers_sizes[0], activation_fn=activation, trainable=True,
                                            weights_initializer=initializer,
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='h1')
            output = layers.fully_connected(h1, num_outputs=num_output, activation_fn=output_activation, trainable=True,
                                            weights_initializer=initializer,
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='output')
        elif num_layer == 2:
            h1 = layers.fully_connected(fc_input, num_outputs=layers_sizes[0], activation_fn=activation, trainable=True,
                                            weights_initializer=initializer,
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='h1')
            h2 = layers.fully_connected(h1, num_outputs=layers_sizes[1], activation_fn=activation, trainable=True,
                                            weights_initializer=initializer,
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='h2')
            output = layers.fully_connected(h2, num_outputs=num_output, activation_fn=output_activation, trainable=True,
                                            weights_initializer=initializer,
                                            weights_regularizer=weights_regularization,
                                            biases_regularizer=bias_regularization,
                                            scope='output')
        else:
            print('Too much layers.')

        return output
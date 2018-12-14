import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import functools


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=False):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights


class CNNSRModel:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _placeholder(self):
        pass

    def kernel(self):
        layer_conv1, weights_conv1 = new_conv_layer(input=x_ph, num_input_channels=3, filter_size=5, num_filters=64)
        layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=64, filter_size=1,
                                                    num_filters=16)
        layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=16, filter_size=5,
                                                    num_filters=3)

        return [layer_conv1, layer_conv2, layer_conv3], [weights_conv1, weights_conv1, weights_conv1]
"""
Tensorflow based Value Functions
"""

import tensorflow as tf


class ValueFunctionBuilder(object):

    def __init__(self,
                 input_shape,
                 hidden_shape,
                 output_shape
                ):

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        weights = [weight_variable((input_shape, hidden_shape[0]))]
        weights += [weight_variable([hidden_shape[i], hidden_shape[i+1]]) for i in range(len(hidden_shape)-1)]
        weights += [weight_variable((hidden_shape[-1], output_shape))]
        self.weights = weights

        biases = [bias_variable([s]) for s in hidden_shape]
        biases += [bias_variable([output_shape])]
        self.biases = biases

    def build(self, x):

        yi = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            yi = tf.nn.relu(tf.matmul(yi, w) + b)

        y = tf.matmul(yi, self.weights[-1]) + self.biases[-1]
        return y


def squared_loss(y0, y1):
    squared_deltas = tf.square(y1-y0)
    loss = tf.reduce_sum(squared_deltas)
    return loss


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

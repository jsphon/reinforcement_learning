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

    def train_loop(self,
                   x,
                   y,
                   num_steps=10,
                   learning_rate=0.01
                   ):

        i = tf.constant(0)

        def cond(counter):
            return tf.less(counter, num_steps)

        def body(counter):
            body_y = self.build(x)
            body_loss = squared_loss(body_y, y)
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=body_loss)

            with tf.control_dependencies([train_step]):
                return counter+1
        return tf.while_loop(cond, body, [i])



    def train_op(self, x, y, *args, **kwargs):
        loss = self.squared_loss(x, y)
        op = tf.train.GradientDescentOptimizer(*args, **kwargs).minimize(loss=loss)
        return op

    def squared_loss(self, x, y):
        y_hat = self.build(x)
        loss = squared_loss(y_hat, y)
        return loss


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

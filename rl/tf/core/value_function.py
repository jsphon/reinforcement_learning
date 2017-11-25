"""
Tensorflow based Value Functions
"""

import tensorflow as tf


class ValueFunctionBuilder(object):

    def __init__(self,
                 input_shape,
                 hidden_shape,
                 output_shape,
                 use_one_hot_input_transform=False,
                 custom_input_transform=None
                ):

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.use_one_hot_input_transform = use_one_hot_input_transform

        if use_one_hot_input_transform:
            assert custom_input_transform is None
            self.custom_input_transform = None
        else:
            self.custom_input_transform = custom_input_transform

        weights = [weight_variable((input_shape, hidden_shape[0]))]
        weights += [weight_variable([hidden_shape[i], hidden_shape[i+1]]) for i in range(len(hidden_shape)-1)]
        weights += [weight_variable((hidden_shape[-1], output_shape))]
        self.weights = weights

        biases = [bias_variable([s]) for s in hidden_shape]
        biases += [bias_variable([output_shape])]
        self.biases = biases

    def vectorized(self, states):
        results = tf.map_fn(self.calculate, states, dtype=tf.float32)
        shape = (tf.shape(states)[0], self.output_shape)
        results = tf.reshape(results, shape)
        return results

    def calculate(self, x):
        yi = x
        if self.use_one_hot_input_transform:
            yi = tf.one_hot(x, depth=self.input_shape, dtype=tf.float32)
        elif self.custom_input_transform:
            yi = self.custom_input_transform(yi)

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
            train_op = self.train_op(x, y, learning_rate=learning_rate)
            with tf.control_dependencies([train_op]):
                return counter+1
        return tf.while_loop(cond, body, [i])

    def train_op(self, x, y, *args, **kwargs):
        loss = self.squared_loss(x, y)
        op = tf.train.GradientDescentOptimizer(*args, **kwargs).minimize(loss=loss)
        return op

    def squared_loss(self, x, y):
        y_hat = self.calculate(x)
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
